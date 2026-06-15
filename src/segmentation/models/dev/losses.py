import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from config import TrainingConfig
from monai.losses.dice import DiceCELoss, DiceFocalLoss, GeneralizedDiceFocalLoss, DiceLoss
from monai.losses import FocalLoss, TverskyLoss

class CustomDiceFocalLoss(nn.Module):
    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        
        self.dice = DiceLoss(
            include_background=False, 
            to_onehot_y=True, 
            softmax=True,
            weight=torch.tensor(cfg.class_weights, dtype=torch.float32) 
                   if cfg.class_weights is not None else None,
        )
        
        self.focal = FocalLoss(
            include_background=True, 
            to_onehot_y=True,
            gamma=cfg.focal_gamma, 
            weight=torch.tensor(cfg.class_weights, dtype=torch.float32) 
                   if cfg.class_weights is not None else None,
            use_softmax=True,
        )
        
        self.w_dice = cfg.dice_weight
        self.w_focal = 1 - cfg.dice_weight

    def forward(self, outputs, masks):
        logits = outputs
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
            
        loss_dice = self.dice(logits, masks)
        loss_focal = self.focal(logits, masks)
        
        total_loss = (self.w_dice * loss_dice) + (self.w_focal * loss_focal)
        return total_loss

class WeightedTverskyLoss(nn.Module):
    """
    Tversky Loss avec pondération par classe.
    
    Remplace TverskyLoss MONAI qui ne supporte pas `weight`.
    
    alpha : poids des FP (< 0.5 → tolère plus les FP)
    beta  : poids des FN (> 0.5 → pénalise les FN → meilleur recall)
    
    Pour necrosis (classe rare) : alpha=0.3, beta=0.7 est un bon départ.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        class_weights = None,
        smooth: float = 1e-5,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index

        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", w)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [B, C, H, W]  — logits bruts (pas de softmax avant)
        targets : [B, H, W]     — indices de classe (long)
                  ou [B, 1, H, W] (squeeze automatique)
        """
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)            # [B, C, H, W]

        # Masque pixels valides (ignore_index)
        valid = (targets != self.ignore_index)          # [B, H, W]

        # One-hot : [B, H, W] → [B, C, H, W]
        one_hot = F.one_hot(
            targets.clamp(min=0), num_classes
        ).permute(0, 3, 1, 2).float()

        # Appliquer le masque
        valid_4d = valid.unsqueeze(1).float()
        probs   = probs   * valid_4d
        one_hot = one_hot * valid_4d

        # Calcul par classe : somme sur B, H, W
        dims = (0, 2, 3)
        TP = (probs * one_hot).sum(dim=dims)            # [C]
        FP = (probs * (1 - one_hot)).sum(dim=dims)      # [C]
        FN = ((1 - probs) * one_hot).sum(dim=dims)      # [C]

        tversky_per_class = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )                                               # [C]

        # Pondération par classe
        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)
            loss = 1 - (tversky_per_class * w).sum() / w.sum()
        else:
            loss = 1 - tversky_per_class.mean()

        return loss

class SegmentationLoss(nn.Module):

    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.loss_type == "ce+dice":
            self.main_loss = DiceCELoss(
                include_background=True,
                to_onehot_y=True,
                softmax=True,
                squared_pred=False,
                reduction='mean',
                lambda_dice=cfg.dice_weight,
                lambda_ce= 1 - cfg.dice_weight,
                weight=torch.tensor(cfg.class_weights, dtype=torch.float32) 
                if cfg.class_weights is not None else None 
            )
        elif cfg.loss_type == "focal+dice":
            self.main_loss = DiceFocalLoss(
                include_background=True,
                to_onehot_y=True,
                softmax=True,
                sigmoid=False,
                reduction='mean',
                lambda_dice=cfg.dice_weight,
                lambda_focal= 1 - cfg.dice_weight,
                weight=torch.tensor(cfg.class_weights, dtype=torch.float32) 
                if cfg.class_weights is not None else None 
            )
        elif cfg.loss_type == "tversky":
            self.main_loss = WeightedTverskyLoss(
                alpha=0.3,
                beta=0.7,
                class_weights=cfg.class_weights,
            )
        else:
            raise ValueError(f"Loss inconnue : {cfg.loss_type}")

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        gt = torch.unsqueeze(targets, 1)
        loss = self.main_loss(outputs, gt)

        return loss

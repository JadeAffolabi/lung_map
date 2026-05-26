"""
utils/losses.py — Fonctions de perte pour la segmentation
==========================================================
Implémentation des losses les plus utilisées en segmentation médicale/histologie.

🔬 EXPÉRIMENTATION :
   - loss_type : "ce+dice" est le standard robuste
   - "focal+dice" : si déséquilibre de classes sévère
   - "tversky"    : si faux négatifs plus coûteux que faux positifs (tumeurs rares)
   - dice_weight  : 0.3 (focus CE) → 0.7 (focus forme)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from config import TrainingConfig


class DiceLoss(nn.Module):
    """
    Dice Loss : mesure le chevauchement entre prédiction et vérité terrain.
    Robuste aux déséquilibres de classes.

    🔬 EXPÉRIMENTATION :
       - smooth : 1.0 (standard) vs 0.0 (strict) vs 1e-5
       - Essayer Generalized Dice Loss (pondère par l'inverse de la fréquence)
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C, H, W) — logits non normalisés
        targets : (B, H, W)   — indices de classe entiers
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode les targets
        targets_oh = F.one_hot(
            targets.clamp(0), num_classes
        ).permute(0, 3, 1, 2).float()   # (B, C, H, W)

        # Masque les pixels ignorés
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs = probs * mask
            targets_oh = targets_oh * mask

        dims = (0, 2, 3)  # Moyenne sur B, H, W — séparé par classe
        intersection = (probs * targets_oh).sum(dim=dims)
        cardinality = (probs + targets_oh).sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss : atténue la contribution des pixels faciles pour se concentrer
    sur les cas difficiles (pixels de contour, petites structures).

    🔬 EXPÉRIMENTATION :
       - gamma = 0 → CE classique | gamma = 2 → standard | gamma = 4 → fort focus
       - alpha (class_weights) : pondère par l'inverse de la fréquence de chaque classe
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


class TverskyLoss(nn.Module):
    """
    Généralisation du Dice avec contrôle des faux positifs/négatifs.

    alpha = 0.5, beta = 0.5 → Dice classique
    alpha < beta             → pénalise davantage les faux négatifs (recall)

    🔬 EXPÉRIMENTATION :
       - alpha=0.3, beta=0.7 : pour détecter les petites structures (tumeurs)
       - Comparer avec Dice standard sur votre dataset
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        tp = (probs * targets_oh).sum(dim=dims)
        fp = (probs * (1 - targets_oh)).sum(dim=dims)
        fn = ((1 - probs) * targets_oh).sum(dim=dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return (1 - tversky).mean()


class SegmentationLoss(nn.Module):
    """
    Loss composite avec support de la deep supervision.

    🔬 EXPÉRIMENTATION :
       - Pondérations aux sorties auxiliaires : [1.0, 0.4, 0.2, 0.1]
       - Désactiver la deep supervision après N epochs (curriculum)
       - Essayer loss_type = "tversky" pour les structures rares
    """

    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        self.cfg = cfg
        dw = cfg.dice_weight
        cw = 1.0 - dw

        alpha = (
            torch.tensor(cfg.class_weights) if cfg.class_weights else None
        )

        if cfg.loss_type == "ce+dice":
            self.main_loss = lambda p, t: (
                cw * F.cross_entropy(p, t, weight=alpha)
                + dw * DiceLoss()(p, t)
            )
        elif cfg.loss_type == "focal+dice":
            focal = FocalLoss(gamma=cfg.focal_gamma, alpha=alpha)
            dice = DiceLoss()
            self.main_loss = lambda p, t: cw * focal(p, t) + dw * dice(p, t)
        elif cfg.loss_type == "tversky":
            tversky = TverskyLoss()
            self.main_loss = lambda p, t: tversky(p, t)
        else:
            raise ValueError(f"Loss inconnue : {cfg.loss_type}")

        # Poids des sorties auxiliaires (deep supervision)
        # 🔬 EXPÉRIMENTATION : Modifier ces pondérations
        self.aux_weights = [0.4, 0.2, 0.1]

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        outputs : dict avec clés "logits" + optionnellement "aux1", "aux2", "aux3"
        targets : (B, H, W) — masques de segmentation
        """
        loss = self.main_loss(outputs["logits"], targets)
        losses = {"main": loss}

        # Supervision profonde : chaque sortie auxiliaire est upsampled vers la taille cible
        for i, key in enumerate(["aux1", "aux2", "aux3"]):
            if key in outputs:
                aux_logits = F.interpolate(
                    outputs[key],
                    size=targets.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                )
                aux_loss = self.main_loss(aux_logits, targets)
                losses[key] = aux_loss
                loss = loss + self.aux_weights[i] * aux_loss

        losses["total"] = loss
        return losses

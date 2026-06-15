"""
training/lightning_module.py — LightningModule de segmentation
===============================================================
Encapsule le modèle, l'optimisation et les métriques dans l'interface
standard de PyTorch Lightning.

Responsabilités :
  • configure_optimizers : AdamW avec LR différenciés encodeur/décodeur
                           + scheduler cosine avec warmup linéaire
  • training_step        : forward + loss + log
  • validation_step      : forward + métriques (Dice, IoU) + log
  • on_validation_epoch_end : agrégation des métriques par époque

Ce que Lightning gère automatiquement (vs train.py manuel) :
  • Mixed precision       → Trainer(precision="16-mixed")
  • Gradient clipping     → Trainer(gradient_clip_val=...)
  • Boucles train/val     → plus de for epoch in range(...)
  • Sauvegarde            → ModelCheckpoint callback
  • Logging               → self.log() → WandB / TensorBoard / CSV
  • Reproductibilité      → seed_everything()
  • Multi-GPU             → Trainer(devices=N, strategy="ddp")
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR, ReduceLROnPlateau, PolynomialLR
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import pytorch_lightning as pl
from torchmetrics import JaccardIndex
from torchmetrics.segmentation import DiceScore
from torchmetrics.functional.segmentation import dice_score
from monai.metrics.hausdorff_distance import compute_hausdorff_distance

from unet_model import UNet
from segmentation_model import SegmentationModel, SegmentationModel2
import segmentation_models_pytorch as smp

from losses import SegmentationLoss
from config import Config
from src.segmentation.constants import LABELS_TO_CLASSES

class SegmentationModuleFM(pl.LightningModule):
    """
    LightningModule pour la segmentation histologique.

    Usage :
        module = SegmentationModule(cfg)
        trainer = pl.Trainer(...)
        trainer.fit(module, datamodule=datamodule)

    🔬 EXPÉRIMENTATION :
        Tous les hyperparamètres sont dans cfg (config.py).
        Modifier cfg puis relancer — Lightning logge tout automatiquement.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()  # sauvegarde cfg dans le checkpoint

        # ── Modèle ──────────────────────────────────────────────────────────
        self.model = SegmentationModel2(cfg)

        # ── Loss ────────────────────────────────────────────────────────────
        self.loss_fn = SegmentationLoss(cfg.training)

        self.val_metric_accumul = []
        self.test_metric_accumul = []


    # ═══════════════════════════════════════════════════════════════════════
    # Forward
    # ═══════════════════════════════════════════════════════════════════════

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["logits"]

    # ═══════════════════════════════════════════════════════════════════════
    # Étapes d'entraînement
    # ═══════════════════════════════════════════════════════════════════════

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.model(images, target_size=masks.shape[1:])
        losses = self.loss_fn(outputs, masks)

        # self.log → Lightning route automatiquement vers WandB/TB/CSV
        # on_step=True  : valeur du step courant dans les courbes
        # on_epoch=True : moyenne sur l'époque (pour les courbes par époque)
        self.log("train/train_loss", losses["total"], on_step=False, on_epoch=True,
                logger=True)
        self.log("train/train_loss_main", losses["main"],  on_step=False, on_epoch=True,
                logger=True)

        # Log des pertes auxiliaires si deep supervision active
        for key in ["aux1", "aux2", "aux3"]:
            if key in losses:
                self.log(f"train/train_{key}_loss", losses[key], on_step=False,
                        on_epoch=True, logger=True)

        # Métrique train (optionnel — coûteux, désactiver si besoin)
        preds = outputs["logits"].argmax(dim=1)

        """ self.train_dice(preds, masks)
        self.log('train_dice', self.train_dice, on_step=False, on_epoch=True, prog_bar=True) """

        return losses["total"]

    def _log_raw_scores(self, scores, score_name, prefix):
        classes_score = scores.nanmean(dim=0)
        
        # 1. On prépare un dictionnaire pour stocker les métriques
        metrics_to_log = {}
        
        for i, score in enumerate(classes_score):
            # 2. Utilisation du "/" pour que le logger regroupe les courbes sur un seul graphique
            class_name = LABELS_TO_CLASSES[i+1]
            metrics_to_log[f'{prefix}/{prefix}_{score_name}_{class_name}'] = score
            
        global_score = classes_score.nanmean()
        metrics_to_log[f'{prefix}/{prefix}_{score_name}'] = global_score

        # 3. On log tout en une seule fois
        self.log_dict(
            metrics_to_log,
            on_step=False, 
            on_epoch=True, 
            prog_bar=False,
            logger=True
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Validation
    # ═══════════════════════════════════════════════════════════════════════

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.model(images, target_size=masks.shape[1:])
        losses = self.loss_fn(outputs, masks)

        self.log("val/val_loss", losses["total"], on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        preds = outputs["logits"].argmax(dim=1)

        batch_dice = dice_score(
            preds, masks, num_classes=4, average="none", input_format='index',
            aggregation_level='samplewise', include_background=False,
        )

        self.val_metric_accumul.append(
            {"dice": batch_dice}
        )

        return losses['total']
    
    def on_validation_epoch_end(self):
        all_dice = torch.cat([x["dice"] for x in self.val_metric_accumul], dim=0)
        self._log_raw_scores(scores=all_dice, score_name="dice", prefix="val")
        self.val_metric_accumul.clear()

    def test_step(self, batch, batch_idx):
        images, masks = batch
        
        # 1. Passage dans le modèle
        outputs = self.model(images)
        preds = outputs["logits"].argmax(dim=1)
        
        # 2. Calcul de la loss (optionnel en test, mais toujours utile à observer)
        losses = self.loss_fn(outputs, masks)

        batch_dice = dice_score(
            preds, masks, num_classes=4, average="none", input_format='index',
            aggregation_level='samplewise', include_background=False,
        )        
        preds_one_hot = F.one_hot(preds, num_classes=4).permute(0, 3, 1, 2).float() 
        masks_one_hot = F.one_hot(masks, num_classes=4).permute(0, 3, 1, 2).float()

        batch_hd95 = compute_hausdorff_distance(
                y_pred=preds_one_hot, 
                y=masks_one_hot, 
                include_background=False, 
                percentile=95
            )
        
        self.test_metric_accumul.append({
            'dice': batch_dice,
            'hausdorff': batch_hd95,
        })

        return losses['total']
    
    def on_test_epoch_end(self):
        all_dice = torch.cat([x["dice"] for x in self.test_metric_accumul], dim=0)
        all_hausdorff = torch.cat([x["hausdorff"] for x in self.test_metric_accumul], dim=0)

        self._log_raw_scores(scores=all_dice, score_name="dice", prefix="test")
        self._log_raw_scores(scores=all_hausdorff, score_name="hd95", prefix="test")

        self.test_metric_accumul.clear()

    # ═══════════════════════════════════════════════════════════════════════
    # Optimiseur et scheduler
    # ═══════════════════════════════════════════════════════════════════════

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        LR différenciés encodeur / décodeur.

        Scheduler : warmup linéaire sur `warmup_epochs` epochs,
                    puis CosineAnnealingLR jusqu'à `max_epochs`.

        🔬 [EXP] :
          - Ratio lr_decoder / lr_encoder : 10× standard, essayer 5× et 20×
          - Remplacer CosineAnnealing par PolynomialLR (power=0.9)
            → standard SegFormer, DeepLab
          - Essayer OneCycleLR (warmup + cosine + cool-down en un seul scheduler)
        """
        cfg = self.cfg.training

        param_groups = [
            {
                "params": [p for p in self.model.encoder.parameters()
                           if p.requires_grad],
                "lr": cfg.lr_encoder,
                "name": "encoder",
            },
            {
                "params": list(self.model.decoder.parameters()),
                "lr": cfg.lr_decoder,
                "name": "decoder",
            },
        ]
        # Retirer les groupes vides (encodeur fully frozen)
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        optimizer = AdamW(
            param_groups,
            weight_decay=cfg.weight_decay,
        )

        # ── Warmup linéaire ────────────────────────────────────────────────
        # LambdaLR multiplie le LR de base par le facteur retourné par la lambda.
        warmup_epochs = cfg.warmup_epochs
        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            return 1.0

        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

        # ── Cosine après warmup ────────────────────────────────────────────
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.max_epochs - warmup_epochs,
            eta_min=cfg.lr_encoder * 1e-2,  # [EXP] floor du LR
        )

        # SequentialLR : warmup_epochs étapes de warmup, puis cosine
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   # step ou epoch
                "monitor": "val_dice", # utilisé par ReduceLROnPlateau uniquement
            },
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Callbacks helpers
    # ═══════════════════════════════════════════════════════════════════════

    def on_train_start(self):
        """Log le nombre de paramètres entraînables au début."""
        stats = self.model.count_parameters()
        self.log_dict({
            "params/total":     float(stats["total"]),
            "params/trainable": float(stats["trainable"]),
        })

    def on_save_checkpoint(self, checkpoint):
        """
        Intercepte la sauvegarde juste avant l'écriture sur le disque.
        Retire dynamiquement tous les paramètres gelés pour gagner de l'espace.
        """
        state_dict = checkpoint["state_dict"]

        # 1. On scanne le modèle pour trouver les paramètres gelés (non-trainables)
        frozen_params = [
            name for name, param in self.named_parameters() 
            if not param.requires_grad
        ]

        for name in frozen_params:
            if name in state_dict:
                del state_dict[name]


    def on_load_checkpoint(self, checkpoint):
        """
        Intercepte le chargement juste avant que Lightning n'applique les poids.
        Comble les trous laissés par les paramètres gelés non-sauvegardés.
        """
        current_state = self.state_dict()
    
        for name, param in current_state.items():
            if name not in checkpoint["state_dict"]:
                checkpoint["state_dict"][name] = param

    @classmethod
    def load_from_checkpoint_with_config(cls, ckpt_path: str) -> "SegmentationModule":
        """
        Rechargement complet : poids + config depuis un checkpoint Lightning.

        Usage :
            module = SegmentationModule.load_from_checkpoint_with_config(
                "checkpoints/best.ckpt"
            )
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["seg_config"]
        module = cls(cfg)
        module.load_state_dict(ckpt["state_dict"])
        return module


class SegmentationModule(pl.LightningModule):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()  # sauvegarde cfg dans le checkpoint

        self.model = UNet(
            n_channels=3,
            n_classes=4,
            bilinear=False,
            dropout=cfg.training.dropout,
        )

        """ self.model = smp.Unet(
            encoder_name='resnet50',
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_interpolation='nearest',
            in_channels=3,
            classes=4,
            activation=None, #after final conv layerq
        ) """

        self.loss_fn = SegmentationLoss(cfg.training)

        self.val_metric_accumul = []
        self.test_metric_accumul = []


    # ═══════════════════════════════════════════════════════════════════════
    # Forward
    # ═══════════════════════════════════════════════════════════════════════

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ═══════════════════════════════════════════════════════════════════════
    # Étapes d'entraînement
    # ═══════════════════════════════════════════════════════════════════════

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, masks)

        self.log("train/train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def _log_raw_scores(self, scores, score_name, prefix):
        classes_score = scores.nanmean(dim=0)
        
        metrics_to_log = {}
        
        for i, score in enumerate(classes_score):
            class_name = LABELS_TO_CLASSES[i+1]
            metrics_to_log[f'{prefix}/{prefix}_{score_name}_{class_name}'] = score
            
        global_score = classes_score.nanmean()
        metrics_to_log[f'{prefix}/{prefix}_{score_name}'] = global_score

        self.log_dict(
            metrics_to_log,
            on_step=False, 
            on_epoch=True, 
            prog_bar=False,
            logger=True
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Validation
    # ═══════════════════════════════════════════════════════════════════════

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, masks)

        self.log("val/val_loss", loss, on_step=False, on_epoch=True)

        preds = outputs.argmax(dim=1)

        batch_dice = dice_score(
            preds, masks, num_classes=4, average="none", input_format='index',
            aggregation_level='samplewise', include_background=False,
        )

        self.val_metric_accumul.append(
            {"dice": batch_dice}
        )

        return loss
    
    def on_validation_epoch_end(self):
        all_dice = torch.cat([x["dice"] for x in self.val_metric_accumul], dim=0)
        self._log_raw_scores(scores=all_dice, score_name="dice", prefix="val")
        self.val_metric_accumul.clear()

    def test_step(self, batch, batch_idx):
        images, masks = batch
        
        outputs = self.model(images)
        preds = outputs.argmax(dim=1)
        
        loss = self.loss_fn(outputs, masks)

        batch_dice = dice_score(
            preds, masks, num_classes=4, average="none", input_format='index',
            aggregation_level='samplewise', include_background=False,
        )        
        preds_one_hot = F.one_hot(preds, num_classes=4).permute(0, 3, 1, 2).float() 
        masks_one_hot = F.one_hot(masks, num_classes=4).permute(0, 3, 1, 2).float()

        batch_hd95 = compute_hausdorff_distance(
                y_pred=preds_one_hot, 
                y=masks_one_hot, 
                include_background=False, 
                percentile=95
            )
        
        self.test_metric_accumul.append({
            'dice': batch_dice,
            'hausdorff': batch_hd95,
        })

        return loss
    
    def on_test_epoch_end(self):
        all_dice = torch.cat([x["dice"] for x in self.test_metric_accumul], dim=0)
        all_hausdorff = torch.cat([x["hausdorff"] for x in self.test_metric_accumul], dim=0)

        self._log_raw_scores(scores=all_dice, score_name="dice", prefix="test")
        self._log_raw_scores(scores=all_hausdorff, score_name="hd95", prefix="test")

        self.test_metric_accumul.clear()

    # ═══════════════════════════════════════════════════════════════════════
    # Optimiseur et scheduler
    # ═══════════════════════════════════════════════════════════════════════

    def configure_optimizers(self) -> OptimizerLRScheduler:

        cfg = self.cfg.training

        param_groups = [
            {
                "params": self.model.parameters(),
                "lr": cfg.lr,
                "name": "unet",
            },
        ]

        if cfg.optimizer == 'sgd':
            optimizer = SGD(
                param_groups,
                momentum=0.99,
                nesterov=True,
                weight_decay=cfg.weight_decay,
            )
        else:
            optimizer = AdamW(
                param_groups,
                weight_decay=cfg.weight_decay,
            )

        # ── Warmup linéaire ────────────────────────────────────────────────
        # LambdaLR multiplie le LR de base par le facteur retourné par la lambda.
        list_scheduler = []
        warmup_epochs = cfg.warmup_epochs
        if warmup_epochs > 0:
            def warmup_lambda(epoch):
                if epoch < warmup_epochs:
                    return float(epoch + 1) / warmup_epochs
                return 1.0

            warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
            list_scheduler.append(warmup_scheduler)

        if cfg.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cfg.max_epochs - warmup_epochs,
                eta_min=cfg.lr_encoder * 1e-2,  # [EXP] floor du LR
            )
                
        elif cfg.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min'
            )
            
        elif cfg.scheduler == 'poly':
            scheduler = PolynomialLR(
                optimizer,
                total_iters = cfg.max_epochs - warmup_epochs,
                power=0.9
            )
            
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cfg.max_epochs - warmup_epochs,
                eta_min=cfg.lr_encoder * 1e-2,  # [EXP] floor du LR
            )
    
        list_scheduler.append(scheduler)

        schedulers = SequentialLR(
            optimizer,
            schedulers=list_scheduler,
            milestones=[warmup_epochs] if warmup_epochs > 0 else [],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": schedulers,
                "interval": "epoch",   # step ou epoch
                "monitor": "val_dice", # utilisé par ReduceLROnPlateau uniquement
            },
        }


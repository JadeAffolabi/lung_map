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
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import pytorch_lightning as pl
from torchmetrics import JaccardIndex
from torchmetrics.segmentation import DiceScore

from segmentation_model import SegmentationModel
from losses import SegmentationLoss
from config import Config


class SegmentationModule(pl.LightningModule):
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
        self.save_hyperparameters(ignore=["cfg"])  # sauvegarde cfg dans le checkpoint

        # ── Modèle ──────────────────────────────────────────────────────────
        self.model = SegmentationModel(cfg)

        # ── Loss ────────────────────────────────────────────────────────────
        self.loss_fn = SegmentationLoss(cfg.training)

        # ── Métriques (torchmetrics — thread-safe, gère l'accumulation) ─────
        # torchmetrics.Dice et JaccardIndex accumulent sur toute l'époque
        # avant de calculer la métrique finale (évite la moyenne de moyennes)
        num_classes = cfg.decoder.num_classes
        task = "binary" if num_classes == 2 else "multiclass"

        self.val_dice = DiceScore(num_classes)
        self.val_iou  = JaccardIndex('multiclass', num_classes=num_classes)
        self.train_dice = DiceScore(num_classes)

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
        self.log("train/loss",      losses["total"], on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("train/loss_main", losses["main"],  on_step=False, on_epoch=True,
                 sync_dist=True)

        # Log des pertes auxiliaires si deep supervision active
        for key in ["aux1", "aux2", "aux3"]:
            if key in losses:
                self.log(f"train/{key}_loss", losses[key], on_step=False,
                         on_epoch=True, sync_dist=True)

        # Métrique train (optionnel — coûteux, désactiver si besoin)
        preds = outputs["logits"].argmax(dim=1)
        self.train_dice.update(preds, masks)

        return losses["total"]

    def on_train_epoch_end(self):
        self.log("train/dice", self.train_dice.compute(),
                 prog_bar=False, sync_dist=True)
        self.train_dice.reset()

    # ═══════════════════════════════════════════════════════════════════════
    # Validation
    # ═══════════════════════════════════════════════════════════════════════

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.model(images, target_size=masks.shape[1:])
        losses = self.loss_fn(outputs, masks)

        self.log("val/loss", losses["total"], on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        preds = outputs["logits"].argmax(dim=1)
        self.val_dice.update(preds, masks)
        self.val_iou.update(preds, masks)

    def on_validation_epoch_end(self):
        # torchmetrics agrège sur tous les batches de validation
        # avant de calculer — c'est la bonne façon (pas mean of means)
        dice = self.val_dice.compute()
        iou  = self.val_iou.compute()

        self.log("val/dice", dice, prog_bar=True, sync_dist=True)
        self.log("val/iou",  iou,  prog_bar=True, sync_dist=True)

        self.val_dice.reset()
        self.val_iou.reset()

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
                "monitor": "val/dice", # utilisé par ReduceLROnPlateau uniquement
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
        """Ajoute la config dans le checkpoint pour la reproductibilité."""
        checkpoint["seg_config"] = self.cfg

    @classmethod
    def load_from_checkpoint_with_config(cls, ckpt_path: str) -> "SegmentationModule":
        """
        Rechargement complet : poids + config depuis un checkpoint Lightning.

        Usage :
            module = SegmentationModule.load_from_checkpoint_with_config(
                "checkpoints/best.ckpt"
            )
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt["seg_config"]
        module = cls(cfg)
        module.load_state_dict(ckpt["state_dict"])
        return module

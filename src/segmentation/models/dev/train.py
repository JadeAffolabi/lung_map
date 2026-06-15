"""
train.py — Boucle d'entraînement principale
============================================
Orchestre l'entraînement avec :
  - LR différentiés encodeur/décodeur
  - Mixed precision (AMP)
  - Métriques de segmentation (Dice, IoU)
  - Checkpointing du meilleur modèle
  - Logging WandB (optionnel)

Usage :
    python train.py

🔬 EXPÉRIMENTATION :
    Modifier config.py puis relancer — toutes les décisions importantes
    sont dans Config.
"""

import os
import random
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR
import torch.nn.functional as F

from config import Config, EncoderConfig, DecoderConfig, TrainingConfig
from models.segmentation_model import SegmentationModel
from utils.losses import SegmentationLoss
from datasets.dataset import build_dataloaders

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════════════

class SegMetrics:
    """
    Calcule Dice et IoU par classe sur toute une époque.

    🔬 EXPÉRIMENTATION : Ajouter Hausdorff Distance pour les contours fins
    """

    def __init__(self, num_classes: int, ignore_index: int = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = logits.argmax(dim=1)  # (B, H, W)
        mask = targets != self.ignore_index

        for c in range(self.num_classes):
            pred_c = (preds == c) & mask
            tgt_c = (targets == c) & mask
            self.tp[c] += (pred_c & tgt_c).sum().float()
            self.fp[c] += (pred_c & ~tgt_c).sum().float()
            self.fn[c] += (~pred_c & tgt_c).sum().float()

    def compute(self) -> Dict[str, float]:
        eps = 1e-7
        dice = (2 * self.tp) / (2 * self.tp + self.fp + self.fn + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        return {
            "dice_mean": dice.mean().item(),
            "iou_mean": iou.mean().item(),
            **{f"dice_c{i}": dice[i].item() for i in range(self.num_classes)},
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. OPTIMISEUR ET SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════

def build_optimizer(model: SegmentationModel, cfg: TrainingConfig):
    """
    LR différencié : faible pour l'encodeur (poids pré-entraînés),
    plus élevé pour le décodeur (aléatoire).

    🔬 EXPÉRIMENTATION :
       - Ratio lr_decoder / lr_encoder : 10x standard, essayer 5x, 20x
       - Essayer Lion optimizer (plus efficace en mémoire qu'AdamW)
       - weight_decay : 0 sur les biais et norms (standard)
    """
    param_groups = model.get_trainable_params()

    if cfg.optimizer == "adamw":
        return AdamW(param_groups, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=cfg.weight_decay)
    raise ValueError(f"Optimiseur inconnu : {cfg.optimizer}")


def build_scheduler(optimizer, cfg: TrainingConfig):
    """
    🔬 EXPÉRIMENTATION :
       - cosine  : décroissance douce, bon défaut
       - poly    : standard en segmentation sémantique (DeepLab, SegFormer)
       - plateau : adaptatif selon la validation loss
    """
    if cfg.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.max_epochs - cfg.warmup_epochs)
    elif cfg.scheduler == "poly":
        return PolynomialLR(
            optimizer, total_iters=cfg.max_epochs, power=0.9
        )
    raise ValueError(f"Scheduler inconnu : {cfg.scheduler}")


def warmup_lr(optimizer, epoch: int, warmup_epochs: int, base_lrs: list):
    """Linear warmup des LR pendant les premières époques."""
    if epoch < warmup_epochs:
        alpha = (epoch + 1) / warmup_epochs
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            param_group["lr"] = base_lr * alpha


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BOUCLES TRAIN / VALIDATE
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    loss_fn: SegmentationLoss,
    scaler: GradScaler,
    device: torch.device,
    cfg: TrainingConfig,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    metrics = SegMetrics(cfg.decoder.num_classes if hasattr(cfg, "decoder") else 2)
    total_loss = 0.0

    for step, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=cfg.amp):
            outputs = model(images)
            losses = loss_fn(outputs, masks)
            loss = losses["total"]

        scaler.scale(loss).backward()
        # Gradient clipping — important avec les grands ViT
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.gradient_clip
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        metrics.update(outputs["logits"].detach(), masks)

        if step % cfg.log_every_n_steps == 0:
            log.info(
                f"Epoch {epoch} | Step {step}/{len(loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    result = metrics.compute()
    result["loss"] = total_loss / len(loader)
    return result


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    loss_fn: SegmentationLoss,
    device: torch.device,
    cfg: TrainingConfig,
) -> Dict[str, float]:
    model.eval()
    metrics = SegMetrics(cfg.decoder.num_classes if hasattr(cfg, "decoder") else 2)
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(enabled=cfg.amp):
            outputs = model(images)
            losses = loss_fn(outputs, masks)

        total_loss += losses["total"].item()
        metrics.update(outputs["logits"], masks)

    result = metrics.compute()
    result["loss"] = total_loss / len(loader)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ENTRAÎNEMENT PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def train(cfg: Config):
    # ── Reproducibilité ────────────────────────────────────────────────────
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    torch.backends.cudnn.deterministic = False   # False = plus rapide
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")

    # ── Données ────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg.training)

    # ── Modèle ────────────────────────────────────────────────────────────
    model = SegmentationModel(cfg).to(device)
    stats = model.count_parameters()
    log.info(
        f"Paramètres totaux    : {stats['total']:,}\n"
        f"Paramètres entraînables : {stats['trainable']:,} "
        f"({100*stats['trainable']/stats['total']:.1f}%)\n"
        f"  dont encodeur     : {stats['encoder_trainable']:,}\n"
        f"  dont décodeur     : {stats['decoder_total']:,}\n"
        f"Paramètres gelés    : {stats['frozen']:,}"
    )

    # ── Optimisation ───────────────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg.training)
    scheduler = build_scheduler(optimizer, cfg.training)
    scaler = GradScaler(enabled=cfg.training.amp)
    loss_fn = SegmentationLoss(cfg.training)

    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    save_dir = Path(cfg.training.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_dice = 0.0

    # ── Boucle d'entraînement ─────────────────────────────────────────────
    for epoch in range(cfg.training.max_epochs):
        # Warmup LR
        warmup_lr(optimizer, epoch, cfg.training.warmup_epochs, base_lrs)

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            scaler, device, cfg.training, epoch,
        )
        val_metrics = validate(model, val_loader, loss_fn, device, cfg.training)

        # Mise à jour du scheduler (après warmup)
        if epoch >= cfg.training.warmup_epochs:
            scheduler.step()

        log.info(
            f"\n{'='*60}\n"
            f"Epoch {epoch:3d}/{cfg.training.max_epochs}\n"
            f"  Train → Loss: {train_metrics['loss']:.4f}  "
            f"Dice: {train_metrics['dice_mean']:.4f}\n"
            f"  Val   → Loss: {val_metrics['loss']:.4f}  "
            f"Dice: {val_metrics['dice_mean']:.4f}\n"
            f"{'='*60}"
        )

        # Sauvegarde du meilleur modèle
        if val_metrics["dice_mean"] > best_dice:
            best_dice = val_metrics["dice_mean"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "config": cfg,
                },
                save_dir / "best_model.pth",
            )
            log.info(f"  ✅ Meilleur modèle sauvegardé — Dice: {best_dice:.4f}")

    log.info(f"\nEntraînement terminé. Meilleur Dice Val : {best_dice:.4f}")
    return best_dice


# ═══════════════════════════════════════════════════════════════════════════════
# 5. POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = Config()

    # ── Exemple : configuration LoRA + UNI ────────────────────────────────
    cfg.encoder.name = "uni"
    cfg.encoder.finetune_strategy = "lora"
    cfg.encoder.lora_rank = 16
    cfg.encoder.out_indices = [5, 11, 17, 23]  # Blocs ViT-L (24 blocs)

    cfg.decoder.num_classes = 2
    cfg.decoder.channels = [512, 256, 128, 64]
    cfg.decoder.deep_supervision = True

    cfg.training.batch_size = 4
    cfg.training.max_epochs = 100
    cfg.training.loss_type = "ce+dice"

    train(cfg)

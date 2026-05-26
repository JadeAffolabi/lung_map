"""
models/segmentation_model.py — Modèle de segmentation complet
===============================================================
Assemble l'encodeur (Foundation Model + fine-tuning) et le décodeur (UNet).

  Image (B, 3, H, W)
       ↓
  FoundationEncoder  →  [f0, f1, f2, f3]
       ↓
  UNetDecoder        →  {"logits": (B, C, H, W), "aux1": ..., ...}
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from config import Config
from encoder import FoundationEncoder
from decoder import UNetDecoder


class SegmentationModel(nn.Module):
    """
    Modèle de segmentation complet Foundation Encoder + UNet Decoder.

    Usage :
        cfg = Config()
        model = SegmentationModel(cfg)
        out = model(images)          # {"logits": tensor}
        out = model(images, labels)  # {"logits": tensor, "loss": tensor}

    🔬 EXPÉRIMENTATION :
       - Voir config.py pour tous les hyperparamètres
       - Essayer UNI puis CONCH comme encodeur
       - Comparer LoRA (rank=8,16,32) vs ViT-Adapter vs Frozen
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.encoder = FoundationEncoder(cfg.encoder)
        self.decoder = UNetDecoder(cfg.decoder, encoder_dim=cfg.encoder.proj_dim)

    def forward(
        self,
        x: torch.Tensor,                              # (B, 3, H, W)
        target_size: Optional[tuple] = None,
    ) -> Dict[str, torch.Tensor]:
        target_size = target_size or x.shape[2:]

        features = self.encoder(x)                   # [f0, f1, f2, f3]
        output = self.decoder(features, target_size)
        return output

    def get_trainable_params(self):
        """
        Retourne les paramètres entraînables groupés par learning rate.
        Utile pour les optimiseurs avec LR différentiés (encoder vs decoder).
        """
        encoder_params = [
            p for p in self.encoder.parameters() if p.requires_grad
        ]
        decoder_params = list(self.decoder.parameters())

        return [
            {"params": encoder_params, "lr": self.cfg.training.lr_encoder},
            {"params": decoder_params, "lr": self.cfg.training.lr_decoder},
        ]

    def count_parameters(self) -> Dict[str, int]:
        """Diagnostique : compte les paramètres entraînables et totaux."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_trainable = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        decoder_total = sum(p.numel() for p in self.decoder.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "encoder_trainable": encoder_trainable,
            "decoder_total": decoder_total,
            "frozen": total - trainable,
        }

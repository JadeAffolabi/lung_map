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

from config import Config, TrainingConfig
from encoder import FoundationEncoder, FoundationEncoder2
from decoder import UNetDecoder, UNetDecoder2, SimpleFPN

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

class SegmentationModel2(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()

        self.encoder = FoundationEncoder2(cfg.encoder)

        # out_channels du FPN = channels du décodeur dans l'ordre spatial croissant
        # cfg.decoder.channels = [512, 256, 128, 64]  (bottleneck → surface)
        # FPN doit produire    = [64, 128, 256, 512]  (H/4 → H/32)
        fpn_out_channels = list(reversed(cfg.decoder.channels))

        self.fpn = SimpleFPN(
            embed_dim=cfg.encoder.embed_dim,
            out_channels=fpn_out_channels,
            norm=cfg.decoder.norm_type,
            act=cfg.decoder.activation,
        )

        self.decoder = UNetDecoder2(
            cfg.decoder,
            encoder_channels=fpn_out_channels,  # [64, 128, 256, 512]
        )

        self.cfg = cfg

    def forward(self, x, target_size=None):
        target_size = target_size or x.shape[2:]

        feat_map = self.encoder(x)              # (B, embed_dim, H/16, W/16)
        pyramid  = self.fpn(feat_map)           # [f0, f1, f2, f3]
        output   = self.decoder(pyramid, target_size)

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

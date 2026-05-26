"""
config.py — Configuration centrale du projet
============================================
Toutes les hyperparamètres sont centralisés ici.
🔬 EXPÉRIMENTATION : Les champs annotés `# [EXP]` sont les premiers leviers à tester.
"""

from dataclasses import dataclass, field
from typing import Literal, List, Optional


@dataclass
class EncoderConfig:
    # ── Choix du backbone ──────────────────────────────────────────────────
    name: Literal["uni", "conch"] = "uni"       # [EXP] Comparer UNI vs CONCH
    pretrained_path: str = "/path/to/weights"   # Chemin vers les poids HuggingFace locaux
    img_size: int = 224                          # Taille d'entrée (UNI/CONCH : 224)
    patch_size: int = 16
    embed_dim: int = 1024                        # ViT-L → 1024 | ViT-H → 1280

    # ── Stratégie de fine-tuning ───────────────────────────────────────────
    finetune_strategy: Literal["lora", "vit_adapter", "full", "frozen"] = "lora"  # [EXP]
    freeze_backbone: bool = True                 # Geler le backbone sauf les adapteurs

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_rank: int = 16              # [EXP] Tester 4, 8, 16, 32, 64
    lora_alpha: float = 32.0         # [EXP] Ratio alpha/rank = scaling factor
    lora_dropout: float = 0.1        # [EXP] 0.0, 0.05, 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["qkv", "proj"]  # [EXP] Ajouter "mlp" ?
    )

    # ── ViT-Adapter ────────────────────────────────────────────────────────
    proj_dim: int = 256        # Dimension de projection de l'encodeur → décodeur           # [EXP] 128, 256, 512
    adapter_depth: int = 4           # Nbre de blocs adapter insérés
    use_cnn_prior: bool = True       # Module spatial prior (CNN léger)

    # ── Extraction multi-échelle ───────────────────────────────────────────
    # Indices des blocs ViT dont on extrait les features (sur 24 blocs pour ViT-L)
    out_indices: List[int] = field(
        default_factory=lambda: [5, 11, 17, 23]  # [EXP] [3,7,15,23] ou [7,11,17,23]
    )


@dataclass
class DecoderConfig:
    # ── Architecture UNet ──────────────────────────────────────────────────
    channels: List[int] = field(
        default_factory=lambda: [512, 256, 128, 64]  # [EXP] Capacité du décodeur
    )
    use_attention_gate: bool = True   # [EXP] Gates d'attention dans les skip connections
    dropout_rate: float = 0.1         # [EXP] 0.0, 0.1, 0.2
    norm_type: Literal["bn", "ln", "in", "gn"] = "bn"  # [EXP] BatchNorm vs LayerNorm
    activation: Literal["relu", "gelu", "silu"] = "relu"  # [EXP]

    # ── Tête de segmentation ───────────────────────────────────────────────
    num_classes: int = 2             # Adapter selon la tâche
    deep_supervision: bool = True    # [EXP]
    use_fpn: bool = True              # [EXP] FPN neck avant le décodeur
    aspp_dilations: tuple = (6, 12, 18)  # [EXP] (3,6,12) si feature map < 10x10


@dataclass
class TrainingConfig:
    # ── Données ────────────────────────────────────────────────────────────
    data_root: str = "/path/to/dataset"
    batch_size: int = 4              # [EXP] Limité par GPU RAM avec ViT-L
    num_workers: int = 4
    img_size: int = 512              # Taille patches histologie

    # ── Optimisation ───────────────────────────────────────────────────────
    optimizer: Literal["adamw", "sgd", "lion"] = "adamw"  # [EXP]
    lr_encoder: float = 1e-5        # [EXP] LR très faible pour le backbone gelé/LoRA
    lr_decoder: float = 1e-4        # [EXP] LR plus élevé pour le décodeur
    weight_decay: float = 0.01      # [EXP] 0.01, 0.05, 0.1
    scheduler: Literal["cosine", "poly", "plateau"] = "cosine"  # [EXP]
    warmup_epochs: int = 5          # [EXP]
    max_epochs: int = 100

    # ── Loss ───────────────────────────────────────────────────────────────
    loss_type: Literal["ce+dice", "focal+dice", "tversky"] = "ce+dice"  # [EXP]
    dice_weight: float = 0.5        # [EXP] Pondération CE vs Dice
    focal_gamma: float = 2.0        # [EXP] si focal loss
    class_weights: Optional[List[float]] = None  # Déséquilibre de classes

    # ── Augmentations ──────────────────────────────────────────────────────
    use_stain_augmentation: bool = True   # [EXP] Crucial en histologie
    use_mixup: bool = False               # [EXP]
    use_cutmix: bool = False              # [EXP]

    # ── Misc ───────────────────────────────────────────────────────────────
    seed: int = 42
    amp: bool = True                      # Mixed precision (économise VRAM)
    gradient_clip: float = 1.0
    save_dir: str = "./checkpoints"
    log_every_n_steps: int = 10
    fast_dev_run = True


@dataclass
class Config:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
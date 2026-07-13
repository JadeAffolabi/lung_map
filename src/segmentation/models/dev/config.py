from dataclasses import dataclass, field
from typing import Literal, List, Generic, TypeVar, Optional

# 1. Define generic type variables
EncoderCfg = TypeVar('EncoderCfg')
DecoderCfg = TypeVar('DecoderCfg')

@dataclass
class EncoderConfig:
    # ── Choix du backbone ──────────────────────────────────────────────────
    name = "uni"       # [EXP] Comparer UNI vs CONCH
    pretrained_path: str = "/path/to/weights"   # Chemin vers les poids HuggingFace locaux
    img_size: int = 224                          # Taille d'entrée (UNI/CONCH : 224)
    patch_size: int = 16
    embed_dim: int = 1024                        # ViT-L → 1024 | ViT-H → 1280

    # ── Stratégie de fine-tuning ───────────────────────────────────────────
    finetune_strategy: Literal["lora", "frozen"] = "lora"  # [EXP]
    freeze_backbone: bool = True                 # Geler le backbone sauf les adapteurs

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_rank: int = 16              # [EXP] Tester 4, 8, 16, 32, 64
    lora_alpha: int = 32        # [EXP] Ratio alpha/rank = scaling factor
    lora_dropout: float = 0.1        # [EXP] 0.0, 0.05, 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["qkv", "proj"]  # [EXP] Ajouter "mlp" ?
    )

    # ── ViT-Adapter ────────────────────────────────────────────────────────
    proj_dim: int = 256        # Dimension de projection de l'encodeur → décodeur           # [EXP] 128, 256, 512
    adapter_depth: int = 4           # Nbre de blocs adapter insérés

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
    num_classes: int = 4             # Adapter selon la tâche
    deep_supervision: bool = True    # [EXP]
    use_fpn: bool = True              # [EXP] FPN neck avant le décodeur
    aspp_dilations: tuple = (6, 12, 18)  # [EXP] (3,6,12) si feature map < 10x10


@dataclass
class TrainingConfig:
    # ── Données ────────────────────────────────────────────────────────────
    batch_size: int = 4              # [EXP] Limité par GPU RAM avec ViT-L
    num_workers: int = 4

    # ── Optimisation ───────────────────────────────────────────────────────
    optimizer: Literal["adamw", "sgd"] = "adamw"
    lr: float = 1e-5
    lr_encoder: float = 0
    lr_decoder: float = 0 
    weight_decay: float = 0.01
    scheduler: Literal["cosine", "poly", "plateau"] = "cosine"
    warmup_epochs: int = 0
    max_epochs: int = 200
    dropout: float = 0

    # ── Loss ───────────────────────────────────────────────────────────────
    loss_type: Literal["ce+dice", "focal+dice", "tversky"] = "ce+dice"
    dice_weight: float = 0.5
    focal_gamma: float = 2.0
    class_weights: Optional[List[float]] = None


    # ── Misc ───────────────────────────────────────────────────────────────
    seed: int = 42
    amp: bool = True                      # Mixed precision (économise VRAM)
    gradient_clip: float = 1.0
    save_dir: str = "./checkpoints"
    log_every_n_steps: int = 10
    fast_dev_run = False

    # Models
    n_channels: int = 3

@dataclass
class Config(Generic[EncoderCfg, DecoderCfg]):
    encoder: EncoderCfg
    decoder: DecoderCfg
    training: TrainingConfig = field(default_factory=TrainingConfig)
    name: str = 'Model'
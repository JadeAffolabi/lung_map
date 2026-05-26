"""
train_lightning.py — Point d'entrée d'entraînement (PyTorch Lightning)
=======================================================================

Usage :
    python train_lightning.py                          # config par défaut
    python train_lightning.py --strategy ddp --devices 4  # multi-GPU

Architecture du script :
    1. Config          → dataclasses (config.py)
    2. Callbacks       → ModelCheckpoint, EarlyStopping, LRMonitor, RichProgressBar
    3. Logger          → WandbLogger (ou CSVLogger en fallback)
    4. Trainer         → assemble tout + gère AMP, gradient clip, DDP
    5. trainer.fit()   → boucle complète
    6. trainer.test()  → évaluation finale sur test set

Comparaison avec train.py (PyTorch pur) :
  ┌─────────────────────┬──────────────────┬──────────────────────────────┐
  │ Fonctionnalité      │ train.py manuel  │ Lightning                    │
  ├─────────────────────┼──────────────────┼──────────────────────────────┤
  │ Mixed precision     │ GradScaler       │ Trainer(precision="16-mixed")│
  │ Gradient clipping   │ clip_grad_norm_  │ Trainer(gradient_clip_val=.) │
  │ Checkpointing       │ torch.save(...)  │ ModelCheckpoint callback     │
  │ Early stopping      │ à coder          │ EarlyStopping callback       │
  │ Multi-GPU           │ DDP manuel       │ Trainer(devices=N)           │
  │ Logging             │ print / wandb    │ self.log() → partout         │
  │ Reproductibilité    │ seed manuel      │ seed_everything()            │
  └─────────────────────┴──────────────────┴──────────────────────────────┘
"""

import os

import argparse
import glob
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
    RichModelSummary,
)
from pytorch_lightning.loggers import WandbLogger, CSVLogger, TensorBoardLogger

from config import Config, EncoderConfig, DecoderConfig, TrainingConfig
from lightning_module import SegmentationModule
from data_module import SegmentationDataModule
from src.segmentation.constants import PATH_SEG_DATA, ACCESS_TOKEN
from huggingface_hub import login

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

def build_callbacks(cfg: Config) -> list:
    """
    Callbacks Lightning. Chacun est indépendant et peut être activé/désactivé.

    🔬 [EXP] :
      ModelCheckpoint : monitor="val/dice" → sauvegarder sur IoU plutôt ?
      EarlyStopping   : patience=15 → réduire pour les runs rapides
      StochasticWeightAveraging (SWA) : à ajouter si plateau en fin d'entraînement
        from pytorch_lightning.callbacks import StochasticWeightAveraging
        SWA(swa_lrs=1e-4, swa_epoch_start=0.8)
    """
    return [
        # ── Sauvegarde du meilleur modèle ──────────────────────────────────
        ModelCheckpoint(
            dirpath=cfg.training.save_dir,
            filename="best-{epoch:03d}-{val/dice:.4f}",
            monitor="val/dice",
            mode="max",
            save_top_k=3,          # [EXP] garder les 3 meilleurs
            save_last=True,        # toujours sauvegarder le dernier checkpoint
            auto_insert_metric_name=False,
        ),

        # ── Arrêt anticipé ─────────────────────────────────────────────────
        # Évite de continuer si la validation stagne.
        EarlyStopping(
            monitor="val/dice",
            mode="max",
            patience=15,           # [EXP] 10 (agressif) → 20 (patient)
            min_delta=1e-4,        # amélioration minimale considérée
            verbose=True,
        ),

        # ── Suivi du learning rate ─────────────────────────────────────────
        # Logue les LR à chaque epoch → visible dans WandB/TensorBoard
        LearningRateMonitor(logging_interval="epoch"),

        # ── Interface terminal ─────────────────────────────────────────────
        RichProgressBar(),
        RichModelSummary(max_depth=3),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOGGER
# ═══════════════════════════════════════════════════════════════════════════════

def build_logger(cfg: Config, use_wandb: bool = True):
    """
    🔬 [EXP] :
      WandbLogger : idéal pour comparer plusieurs runs (sweep, hyperparameter search)
      CSVLogger   : fallback local, toujours actif
      TensorBoardLogger : si vous préférez TensorBoard
    """
    loggers = []

    if use_wandb:
        try:
            wandb_logger = WandbLogger(
                project="histo-segmentation",
                name=f"{cfg.encoder.name}_{cfg.encoder.finetune_strategy}"
                     f"_r{cfg.encoder.lora_rank}",
                log_model="all",    # upload les checkpoints dans WandB
                config={
                    "encoder": cfg.encoder.__dict__,
                    "decoder": cfg.decoder.__dict__,
                    "training": cfg.training.__dict__,
                },
            )
            loggers.append(wandb_logger)
        except Exception as e:
            print(f"WandB non disponible ({e}), fallback CSV.")

    # CSVLogger toujours actif comme fallback
    loggers.append(CSVLogger(save_dir=cfg.training.save_dir, name="csv_logs"))

    return loggers if len(loggers) > 1 else loggers[0]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

def build_trainer(cfg: Config, args: argparse.Namespace) -> Trainer:
    """
    Trainer Lightning : assemble tous les composants.

    Paramètres clés :
      precision       : "16-mixed"    → AMP BFloat16/FP16 (économise ~40% VRAM)
                        "bf16-mixed"  → plus stable que fp16 (recommandé sur A100)
                        "32-true"     → debug ou si instabilité numérique
      devices         : N GPUs, ou "auto"
      strategy        : "ddp" pour multi-GPU, "auto" en single GPU
      accumulate_grad : simule un batch_size × N sans VRAM supplémentaire

    🔬 [EXP] :
      accumulate_grad_batches : 4 → batch effectif = cfg.batch_size × 4
      val_check_interval      : 0.5 → valider 2× par époque (utile sur gros datasets)
      sync_batchnorm          : True si BatchNorm + multi-GPU (pas nécessaire avec GN)
    """
    return Trainer(
        # ── Durée ─────────────────────────────────────────────────────────
        max_epochs=cfg.training.max_epochs,

        # ── Précision ─────────────────────────────────────────────────────
        precision=getattr(args, "precision", "16-mixed"),

        # ── Matériel ──────────────────────────────────────────────────────
        accelerator="gpu",
        devices=getattr(args, "devices", 1),
        strategy=getattr(args, "strategy", "auto"),

        # ── Gradient ──────────────────────────────────────────────────────
        gradient_clip_val=cfg.training.gradient_clip,
        gradient_clip_algorithm="norm",  # "norm" ou "value"
        accumulate_grad_batches=getattr(args, "accumulate_grad", 1),  # [EXP]

        # ── Validation ────────────────────────────────────────────────────
        val_check_interval=10000,     # [EXP] 0.5 pour valider 2× par époque
        check_val_every_n_epoch=1,

        # ── Callbacks et loggers ──────────────────────────────────────────
        callbacks=build_callbacks(cfg),
        logger=build_logger(cfg, use_wandb=getattr(args, "wandb", True)),

        # ── Reproductibilité ──────────────────────────────────────────────
        deterministic=False,       # True → plus lent mais reproductible strictement

        # ── Debug ─────────────────────────────────────────────────────────
        fast_dev_run=cfg.training.fast_dev_run,       # 1 batch train+val → vérifie que le code tourne
        # overfit_batches=0.01,    # sur-apprend 1% des données → test de sanité
        # profiler="simple",       # profile CPU/GPU pour trouver les goulots
        log_every_n_steps=cfg.training.log_every_n_steps,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraînement segmentation histologie")
    parser.add_argument("--devices",          type=int,   default=1)
    parser.add_argument("--strategy",         type=str,   default="auto")
    parser.add_argument("--precision",        type=str,   default="16-mixed")
    parser.add_argument("--accumulate-grad",  type=int,   default=1,
                        dest="accumulate_grad")
    parser.add_argument("--no-wandb",         action="store_false", dest="wandb")
    parser.add_argument("--fast-dev-run",     action="store_true",  dest="fast_dev_run")
    return parser.parse_args()

def get_path_shards(split: str):
    path_to_shards = str(PATH_SEG_DATA / 'data-shards')
    return glob.glob(path_to_shards + f"/dataset-{split}-*.tar")

def main():
    args = parse_args()

    # ── Reproductibilité globale ───────────────────────────────────────────
    pl.seed_everything(42, workers=True)

    # ── Configuration ─────────────────────────────────────────────────────
    cfg = Config()
    cfg.encoder.name               = "uni"
    cfg.encoder.finetune_strategy  = "frozen"
    cfg.encoder.lora_rank          = 16
    cfg.encoder.lora_target_modules = ["qkv", "proj"]
    cfg.encoder.out_indices        = [5, 11, 17, 23]
    cfg.encoder.proj_dim           = 256

    cfg.decoder.num_classes        = 4
    cfg.decoder.channels           = [512, 256, 128, 64]
    cfg.decoder.deep_supervision   = False
    cfg.decoder.use_fpn            = False
    cfg.decoder.norm_type          = "bn"
    cfg.decoder.activation         = "gelu"

    cfg.training.batch_size        = 4
    cfg.training.max_epochs        = 100
    cfg.training.warmup_epochs     = 5
    cfg.training.lr_encoder        = 1e-5
    cfg.training.lr_decoder        = 1e-4
    cfg.training.loss_type         = "ce+dice"
    cfg.training.save_dir          = "./checkpoints"

    # ── Login Hugginface ───────────────────────────────────────────────────
    login(ACCESS_TOKEN)

    # ── Module et DataModule ───────────────────────────────────────────────

    module     = SegmentationModule(cfg)
    datamodule = SegmentationDataModule(
        train_urls=get_path_shards('train'),
        val_urls=get_path_shards('val'),
        test_urls=get_path_shards('test'),
        batch_size=4, 
        num_workers=0 
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = build_trainer(cfg, args)

    # ── Entraînement ──────────────────────────────────────────────────────
    trainer.fit(module, datamodule=datamodule)

    # ── Évaluation finale sur le test set ─────────────────────────────────
    trainer.test(module, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()

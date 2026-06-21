
import os

import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
    RichModelSummary,
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from config import Config, EncoderConfig, DecoderConfig, TrainingConfig
from lightning_module import SegmentationModule, SegmentationModuleFM
from data_module import SegmentationDataModule, SegmentationDataModule2, get_path_shards, get_class_weight
from src.segmentation.constants import ACCESS_TOKEN
from huggingface_hub import login
import torch

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

torch.set_float32_matmul_precision('medium')


def build_callbacks(cfg: Config) -> list:
    return [
        # ── Sauvegarde du meilleur modèle ──────────────────────────────────
        ModelCheckpoint(
            dirpath=cfg.training.save_dir,
            filename="best-{epoch:03d}-{val_dice:.4f}",
            monitor="val/val_dice",
            mode="max",
            save_top_k=1,       
            save_last=False,
            save_weights_only=True,   
            auto_insert_metric_name=False,
        ),

        ModelCheckpoint(
        dirpath=cfg.training.save_dir,
        filename="last",
        save_top_k=0,
        save_last=True,
        save_weights_only=False,
        ),

        LearningRateMonitor(logging_interval="epoch"),

        RichProgressBar(),
        RichModelSummary(max_depth=3),
    ]

""" EarlyStopping(
            monitor="val/val_dice",
            mode="max",
            patience=20,
            min_delta=1e-4,
            verbose=True,
        ) """

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
                log_model=False,
                config={
                    "training": cfg.__dict__,
                },
                save_dir=cfg.training.save_dir
            )
            loggers.append(wandb_logger)
        except Exception as e:
            print(f"WandB non disponible ({e}), fallback CSV.")
    else:
        loggers.append(TensorBoardLogger(
            save_dir=cfg.training.save_dir,
            name="tensorboard",
        ))

    return loggers if len(loggers) > 1 else loggers[0]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

def build_trainer(cfg: Config, args: argparse.Namespace) -> Trainer:
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
        # gradient_clip_val=cfg.training.gradient_clip,
        # gradient_clip_algorithm="norm",  # "norm" ou "value"
        # accumulate_grad_batches=getattr(args, "accumulate_grad", 1),  # [EXP]

        # ── Validation ────────────────────────────────────────────────────
        check_val_every_n_epoch=1,

        # ── Callbacks et loggers ──────────────────────────────────────────
        callbacks=build_callbacks(cfg),
        logger=build_logger(cfg, use_wandb=getattr(args, "wandb", True)),

        # ── Reproductibilité ──────────────────────────────────────────────
        deterministic=False,       # True → plus lent mais reproductible strictement

        # ── Debug ─────────────────────────────────────────────────────────
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
    parser.add_argument("--save_dir", type=str)
    return parser.parse_args()

def main():
    args = parse_args()

    # ── Reproductibilité globale ───────────────────────────────────────────
    pl.seed_everything(42, workers=True)

    # ── Configuration ─────────────────────────────────────────────────────
    cfg = Config(
        encoder=None,
        decoder=None,
    )

    cfg.training.num_workers       = 4
    cfg.training.batch_size        = 64
    cfg.training.max_epochs        = 100
    cfg.training.warmup_epochs     = 0
    cfg.training.optimizer         = 'adamw'
    cfg.training.lr                = 1e-3
    cfg.training.weight_decay      = 0.0001
    cfg.training.scheduler         = 'cosine'

    cfg.training.dropout           = 0.5

    cfg.training.loss_type         = "ce+dice"
    cfg.training.dice_weight       = 0.5
    cfg.training.focal_gamma       = 0

    cfg.training.save_dir          = args.save_dir

    # ── Login Hugginface ───────────────────────────────────────────────────
    #login(ACCESS_TOKEN)

    # ── Module et DataModule ───────────────────────────────────────────────

    datamodule = SegmentationDataModule2(
        train_urls=get_path_shards('train'),
        val_urls=get_path_shards('val'),
        test_urls=get_path_shards('test'),
        rare_train_urls=get_path_shards('train-rare'),
        common_train_urls=get_path_shards('train-common'),
        batch_size=cfg.training.batch_size, 
        num_workers=cfg.training.num_workers
    )

    """ datamodule = SegmentationDataModule(
        train_urls=get_path_shards('train-full'),
        val_urls=get_path_shards('val'),
        test_urls=get_path_shards('test'),
        batch_size=cfg.training.batch_size, 
        num_workers=cfg.training.num_workers
    ) """

    """ cfg.training.class_weights = get_class_weight(
        datamodule.train_dataloader(),
        norm=True,
        square_root=False,    
    ).tolist() """

    module = SegmentationModule(cfg)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = build_trainer(cfg, args)

    # ── Entraînement ──────────────────────────────────────────────────────
    trainer.fit(
        module, 
        datamodule,
    )

    # ── Évaluation finale sur le test set ─────────────────────────────────
    torch.serialization.add_safe_globals([
        Config, EncoderConfig, 
        DecoderConfig, TrainingConfig,
    ])
    trainer.test(module, datamodule=datamodule, ckpt_path="best")
    #trainer.test(module, datamodule=datamodule)
 

if __name__ == "__main__":
    main()

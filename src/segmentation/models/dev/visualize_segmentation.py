
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms
from matplotlib.colors import ListedColormap
import time

from lightning_module import iou_per_sample_per_class
from config import Config, EncoderConfig, DecoderConfig, TrainingConfig
from data_module import SegmentationDataModule, get_path_shards
from lightning_module import SegmentationModule


def load_model(checkpoint_path: str, device: torch.device):
    torch.serialization.add_safe_globals([
        Config, EncoderConfig, 
        DecoderConfig, TrainingConfig,
    ])
    model = SegmentationModule.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    print("Modèle chargé avec succès.")
    return model


def predict_one(model: SegmentationModule, image_tensor: torch.Tensor,
             device: torch.device) -> np.ndarray:

    #image_tensor = image_tensor.unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        logits = model(image_tensor)  # (1, n_classes, H, W)
        pred_mask = logits.argmax(dim=1)
        #.squeeze(0).cpu().numpy().astype(np.uint8)

    return pred_mask

def predict_batch(model: SegmentationModule, images: torch.Tensor,
             device: torch.device) -> np.ndarray:

    #image_tensor = image_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
    images_tensor = images.to(device)
    with torch.no_grad():
        logits = model(images_tensor)  # (1, n_classes, H, W)
        pred_mask = logits.argmax(dim=1)

    return pred_mask



# ─────────────────────────────────────────────────────────────────────────────
# 5.  Visualisation
# ─────────────────────────────────────────────────────────────────────────────

COLORS = [
    [0, 0, 0],        # classe 0 – fond (noir)
    [255, 0, 0],      # classe 1 – rouge
    [0, 255, 0],      # classe 2 – vert
    [0, 0, 255],      # classe 3 – bleu
    [255, 255, 0],    # classe 4 – jaune
    [255, 0, 255],    # classe 5 – magenta
    [0, 255, 255],    # classe 6 – cyan
    [128, 0, 0],      # classe 7 – bordeaux
    [0, 128, 0],      # classe 8 – vert foncé
    [0, 0, 128],      # classe 9 – bleu marine
]


def mask_to_rgb(mask: np.ndarray, n_classes: int) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(n_classes if n_classes > 1 else 2):
        color = COLORS[cls_idx % len(COLORS)]
        rgb[mask == cls_idx] = color
    return rgb


def overlay_mask(image_np: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image_np = image_np.astype(np.float32)
    mask_rgb = mask_rgb.astype(np.float32)
    overlay = (1 - alpha) * image_np + alpha * mask_rgb
    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_segmentation(images, gt_masks, pred_masks, batch_iou, output_dir, fig_name):

    class_dict = {
        0: ("other", "c"),
        1: ("tumeur", "r"),      
        2: ("necrosis", "b"), 
        3: ("stroma", "g"),
    }

    # Extraction des couleurs pour créer la Colormap fixe
    colors = [class_dict[i][1] for i in sorted(class_dict.keys())]
    custom_cmap = ListedColormap(colors)

    # Valeurs min et max pour forcer l'alignement des couleurs
    vmin = 0
    vmax = len(class_dict) - 1

    fig, axes = plt.subplots(len(images), 4, figsize=(8, 4 * len(images) + 1))

    for i in range(len(images)):

        # 4. Affichage visuel du premier élément du batch
        # PyTorch utilise le format [Canal, Hauteur, Largeur]. 
        # Matplotlib veut [Hauteur, Largeur, Canal]. On utilise .permute() pour réarranger.
        img_display = images[i].permute(1, 2, 0).numpy()
        gt_display = gt_masks[i].numpy()
        pred_display = pred_masks[i].numpy()
        iou = batch_iou[i, :]

        # --- ATTENTION : GESTION DE LA NORMALISATION ---
        # Si vous avez utilisé A.Normalize (ImageNet) dans Albumentations, 
        # l'image aura des couleurs bizarres et des valeurs négatives.
        # Pour l'afficher correctement, on doit "dé-normaliser" visuellement :
        if images.min() < 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_display = std * img_display + mean
            img_display = np.clip(img_display, 0, 1)

        axes[i, 0].set_title("Image")
        axes[i, 0].imshow(img_display)
        axes[i, 0].axis('off')

        axes[i, 1].set_title("Ground truth")
        axes[i, 1].imshow(gt_display, cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[i, 1].axis('off')

        axes[i, 2].set_title(f"Prediction, IoU\n tumor: {iou[0]:.2f} | necrose: {iou[1]:.2f}\n | stroma: {iou[2]:.2f}")
        axes[i, 2].imshow(pred_display, cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[i, 2].axis('off')

        axes[i, 3].set_title(f"Superposition \nde la prediction")
        axes[i, 3].imshow(img_display)
        axes[i, 3].imshow(pred_display, cmap=custom_cmap, vmin=vmin, vmax=vmax, alpha=0.5)
        axes[i, 3].axis('off')

    patches = []
    n = 4
    for i in range(n):
        c = class_dict[i][1]
        label = class_dict[i][0]
        patches.append(mpatches.Patch(color=c, label=label))
    fig.legend(handles=patches, loc="lower center", ncol=min(n, 5),
               bbox_to_anchor=(0.5, -0.05), fontsize=9)

    plt.tight_layout()
    save_path = f"{output_dir}/{fig_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



# ─────────────────────────────────────────────────────────────────────────────
# 6.  Boucle principale
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def get_image_paths(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir()
                  if p.suffix.lower() in IMAGE_EXTENSIONS)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    model = load_model(args.checkpoint, device)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    testloader = SegmentationDataModule(
        train_urls=get_path_shards('train'),
        val_urls=get_path_shards('val'),
        test_urls=get_path_shards('shard512', shard_dir='shards_chu'),
        batch_size=4, 
        num_workers=0
    ).test_dataloader()

    print('Prediction')

    i = 0
    iou_accuml = []
    for images, masks in testloader:

        pred_masks = predict_batch(model, images, device)
        pred_masks = pred_masks.cpu()
        
        batch_iou = iou_per_sample_per_class(pred_masks, masks, 4, ignore_index=0)
        iou_accuml.append(batch_iou)

        fig_name = f'out_{i}'
        plot_segmentation(
            images,
            masks,
            pred_masks,
            batch_iou,
            args.out,
            fig_name
        )
        i += 1
    all_iou = torch.cat([x for x in iou_accuml], dim=0)
    classes_score = all_iou.nanmean(dim=0)
    classes_score_std = np.nanstd(all_iou.numpy(), axis=0)

    print(
        f"tumor iou: {classes_score[0]} +/- {classes_score_std[0]}\n"
        f"necrosis iou: {classes_score[1]} +/- {classes_score_std[1]}\n"
        f"stroma iou: {classes_score[2]} +/- {classes_score_std[2]}\n"
    )

    print("Terminé.")



def parse_args():
    p = argparse.ArgumentParser(description="Visualisation des segmentations UNet")
    p.add_argument("-ckpt", "--checkpoint",  required=True, help="Chemin vers le fichier .ckpt")
    p.add_argument("--out",  default="./segmentation_results",
                   help="Dossier de sortie pour les visualisations")
    return p.parse_args()

if __name__ == "__main__":

    start = time.time()

    args = parse_args()
    run(args)

    end = time.time()
    print(f"Execution time : {(end - start)/60} min")

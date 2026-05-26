from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image, ImageDraw
import random
from scipy.ndimage import binary_erosion
import os
from pathlib import Path 


# ── Constantes ────────────────────────────────────────────────────────────────

CLASS_OTHER      = 0
CLASS_MEDIUM     = 1
CLASS_SMALL_CAT1 = 2
CLASS_SMALL_CAT2 = 3

IMG_COLORS = {
    "large":  [0.71, 0.87, 0.96],
    "medium": [0.27, 0.64, 0.42],
    "cat1":   [0.88, 0.22, 0.22],
    "cat2":   [0.97, 0.80, 0.15],
}

MASK_COLORS = ["#AABBCC", "#2EA854", "#DC2626", "#EAB308"]

MASK_LABELS = [
    "Classe 0 – Autre (fond + grand objet)",
    "Classe 1 – Objet moyen (étoile organique)",
    "Classe 2 – Petits objets cat. 1 (croix organiques)",
    "Classe 3 – Petits objets cat. 2 (blobs compacts)",
]


# ═══════════════════════════════════════════════════ Formes organiques ════════

def _smooth_noise_radii(angles, r_base, roughness, n_freqs, freq_start, rng):
    radii = np.ones_like(angles) * r_base
    for freq in range(freq_start, freq_start + n_freqs):
        amp   = roughness * r_base * rng.uniform(0.4, 1.0) / (freq - freq_start + 1)
        phase = rng.uniform(0, 2 * np.pi)
        radii += amp * np.cos(freq * angles + phase)
    return radii


def _amoeba_blob(cx, cy, r_base, n_pts=120, roughness=0.38, n_freqs=9, rng=None):
    """Grand blob amiboïde très irrégulier (style dessin à main levée)."""
    if rng is None:
        rng = np.random.RandomState()
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    radii  = _smooth_noise_radii(angles, r_base, roughness, n_freqs, 2, rng)
    # Bosses saillantes supplémentaires pour l'aspect amiboïde
    for _ in range(5):
        ca    = rng.uniform(0, 2 * np.pi)
        width = rng.uniform(0.20, 0.50)
        bump  = rng.uniform(0.10, 0.28) * r_base
        for shift in [0, -2 * np.pi, 2 * np.pi]:
            radii += bump * np.exp(-((angles - ca - shift) ** 2) / (2 * width ** 2))
    radii = np.maximum(radii, r_base * 0.25)
    return np.column_stack([cx + radii * np.cos(angles),
                             cy + radii * np.sin(angles)])


def _organic_star(cx, cy, r_outer, r_inner, n_branches=5,
                  n_pts=150, roughness=0.18, rng=None):
    """Étoile organique déformée – branches asymétriques."""
    if rng is None:
        rng = np.random.RandomState()
    rotation = rng.uniform(0, 2 * np.pi)
    angles   = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    a_rot    = angles + rotation
    t        = 0.5 - 0.5 * np.cos(n_branches * a_rot)
    radii    = r_inner + (r_outer - r_inner) * t
    for freq in range(2, 10):
        amp    = roughness * r_outer * rng.uniform(0, 1.0) / freq
        phase  = rng.uniform(0, 2 * np.pi)
        radii += amp * np.sin(freq * angles + phase)
    # Asymétrie par branche
    for k in range(n_branches):
        ba      = rotation + k * 2 * np.pi / n_branches
        stretch = rng.uniform(-0.18, 0.24) * r_outer
        width   = rng.uniform(0.18, 0.42)
        for shift in [0, -2 * np.pi, 2 * np.pi]:
            radii += stretch * np.exp(-((angles - ba - shift) ** 2) / (2 * width ** 2))
    radii = np.maximum(radii, r_inner * 0.35)
    return np.column_stack([cx + radii * np.cos(angles),
                             cy + radii * np.sin(angles)])


def _organic_cross(cx, cy, r, t_ratio=0.35, roughness=0.22, n_pts=100, rng=None):
    """Croix organique déformée (non convexe)."""
    if rng is None:
        rng = np.random.RandomState()
    rotation = rng.uniform(0, np.pi / 4)
    angles   = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    a_rot    = angles + rotation
    cross    = np.maximum(np.abs(np.cos(a_rot)), np.abs(np.sin(a_rot)))
    r_min    = r * t_ratio
    radii    = r_min + (r - r_min) * (cross ** 1.8)
    for freq in range(3, 11):
        amp    = roughness * r * rng.uniform(0, 1.0) / (freq - 1)
        phase  = rng.uniform(0, 2 * np.pi)
        radii += amp * np.sin(freq * angles + phase)
    for arm in range(4):
        ba      = rotation + arm * np.pi / 2
        stretch = rng.uniform(-0.15, 0.25) * r
        width   = rng.uniform(0.15, 0.35)
        for shift in [0, -2 * np.pi, 2 * np.pi]:
            radii += stretch * np.exp(-((angles - ba - shift) ** 2) / (2 * width ** 2))
    radii = np.maximum(radii, r_min * 0.40)
    return np.column_stack([cx + radii * np.cos(angles),
                             cy + radii * np.sin(angles)])


def _compact_blob(cx, cy, r_base, n_pts=50, roughness=0.28, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    radii  = _smooth_noise_radii(angles, r_base, roughness, 6, 2, rng)
    radii  = np.maximum(radii, r_base * 0.40)
    return np.column_stack([cx + radii * np.cos(angles),
                             cy + radii * np.sin(angles)])


# ════════════════════════════════════════════════════════ Utilitaires ══════════

def _rasterize(vertices: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    H, W = shape
    pil  = Image.new("L", (W, H), 0)
    ImageDraw.Draw(pil).polygon(
        [(float(x), float(y)) for x, y in vertices], fill=1
    )
    return np.asarray(pil, dtype=bool)


def _sample_in_mask(mask, rng, n):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    return [(float(xs[i]), float(ys[i])) for i in idx]


# ═══════════════════════════════════════════════════ generate_image ═══════════

def generate_image(
    image_size: int = 512,
    n_cat1: int = 12,
    n_cat2: int = 9,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère une image synthétique hiérarchique avec formes organiques déformées.

    Parameters
    ----------
    image_size : taille H = W de l'image carrée (pixels)
    n_cat1     : nombre de petits objets catégorie 1 (croix organiques)
    n_cat2     : nombre de petits objets catégorie 2 (blobs compacts)
    seed       : graine aléatoire (None = non déterministe)

    Returns
    -------
    img  : np.ndarray float32 (H, W, 3), valeurs dans [0, 1]
    mask : np.ndarray uint8   (H, W)
               0 = autre | 1 = objet moyen | 2 = cat.1 | 3 = cat.2
    """
    rng  = np.random.RandomState(seed)
    H = W = image_size
    cx, cy = W / 2.0, H / 2.0

    img  = np.ones((H, W, 3), dtype=np.float32)
    mask = np.zeros((H, W),   dtype=np.uint8)

    # 1. Grand objet : blob amiboïde
    r_large = int(0.50 * image_size)
    v_large = _amoeba_blob(cx, cy, r_large, roughness=0.35, n_freqs=9, rng=rng)
    m_large = _rasterize(v_large, (H, W))
    img[m_large] = IMG_COLORS["large"]

    # 2. Objet moyen : étoile organique
    if seed is not None:
        random.seed(seed)
    rand_num = random.random()
    if rand_num > 0.5:
        r_outer    = int(0.25 * image_size)
        r_inner    = int(r_outer * 0.42)
        n_branches = int(rng.choice([4, 5, 6]))
        v_med      = _organic_star(cx, cy, r_outer, r_inner,
                                n_branches=n_branches, roughness=0.20, rng=rng)
        m_med      = _rasterize(v_med, (H, W))
        img[m_med]  = IMG_COLORS["medium"]
        mask[m_med] = CLASS_MEDIUM
    else : 
        r_medium = int(0.20 * image_size)
        v_med      = _amoeba_blob(cx, cy, r_medium, roughness=0.35, n_freqs=9, rng=rng)
        m_med      = _rasterize(v_med, (H, W))
        img[m_med]  = IMG_COLORS["medium"]
        mask[m_med] = CLASS_MEDIUM

    # Zone érodée pour placement sécurisé
    erosion_px = max(4, int(0.032 * image_size))
    safe_mask  = binary_erosion(m_med, iterations=erosion_px)

    # 3. Petits objets cat. 1 : croix organiques
    r1         = max(5, int(0.024 * image_size))
    candidates = _sample_in_mask(safe_mask, rng, n=n_cat1 * 6)
    placed = 0
    for (px, py) in candidates:
        if placed >= n_cat1:
            break
        v = _organic_cross(px, py, r=r1, t_ratio=0.34, roughness=0.22, rng=rng)
        m = _rasterize(v, (H, W))
        img[m]  = IMG_COLORS["cat1"]
        mask[m] = CLASS_SMALL_CAT1
        placed += 1

    # 4. Petits objets cat. 2 : blobs compacts
    r2         = max(4, int(0.018 * image_size))
    candidates = _sample_in_mask(safe_mask, rng, n=n_cat2 * 6)
    placed = 0
    for (px, py) in candidates:
        if placed >= n_cat2:
            break
        v = _compact_blob(px, py, r2, roughness=0.28, rng=rng)
        m = _rasterize(v, (H, W))
        img[m]  = IMG_COLORS["cat2"]
        mask[m] = CLASS_SMALL_CAT2
        placed += 1

    return img, mask


# ══════════════════════════════════════════════════ visualize_image ═══════════

def visualize_grid(
    n_images: int = 6,
    image_size: int = 256,
    seeds: Optional[list] = None,
) -> Figure:
    """
    Grille de n_images paires (image, masque) pour explorer la variété des formes.
    """
    if seeds is None:
        seeds = list(range(n_images))
    n_cls = 4
    cmap  = ListedColormap(MASK_COLORS)
    norm  = BoundaryNorm(np.arange(n_cls + 1) - 0.5, n_cls)

    fig, axes = plt.subplots(2, n_images, figsize=(3.2 * n_images, 7))
    fig.suptitle("Variété des formes organiques – grille multi-seeds",
                 fontsize=13, fontweight="bold")
    for i, seed in enumerate(seeds):
        img, mask = generate_image(image_size=image_size, seed=seed)
        axes[0, i].imshow(img, interpolation="bilinear")
        axes[0, i].set_title(f"seed={seed}", fontsize=9)
        axes[0, i].axis("off")
        axes[1, i].imshow(mask, cmap=cmap, norm=norm, interpolation="nearest")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Image", fontsize=10)
    axes[1, 0].set_ylabel("Masque", fontsize=10)
    fig.tight_layout()
    return fig

# ════════════════════════════════════════════════════════════════ Demo ═════════

if __name__ == "__main__":
    nb_images = 5

    data_dir = Path("synthetic_data")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        os.mkdir(data_dir / "images")
        os.mkdir(data_dir / "masks")

    for i in range(nb_images):
        img, mask = generate_image(image_size=512, n_cat1=12, n_cat2=9)
        plt.imsave(f"{data_dir.name}/images/img_{i :03d}.png", img)
        plt.imsave(f"{data_dir.name}/masks/mask_{i :03d}.png", mask)

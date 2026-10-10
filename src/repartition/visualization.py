import math
import os

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

from src.repartition.constants import CLASS_NAMES


def normalize(mask: np.ndarray) -> np.ndarray:
    arr = mask.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)


def make_composite(masks: dict[str, np.ndarray]) -> np.ndarray:
    """
        Merge the masks into a single image array.

    Parameters
    ----------
    masks : dict

    Returns
    -------
        img : np.ndarray
    """
    stack_img = np.stack([normalize(m) for k, m in masks.items() if k != 'tumor_bed'], axis=-1)
    
    soft_colors = np.array([
        [0.90, 0.10, 0.10],
        [0.10, 0.10, 0.90],
        [0.10, 0.70, 0.10],
    ])
    
    img_rgb = stack_img @ soft_colors
    
    img_rgb[~(stack_img.any(axis=-1))] = 1.0 
    
    return np.clip(img_rgb, 0, 1)

def make_image_masks(masks: dict, slide_name: str = None) -> Figure:
    """
        Create an image in which each mask is shown.

    Parameters
    ----------
    masks : dict
    slide_name : str, optional

    Returns
    -------
        fig : Figure
    """
    colormaps   = ["Blues", "Greens", "Reds", "Purples"]
    n_classes  = len(CLASS_NAMES)
    n_cols     = n_classes + 1

    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4.5))
    img_title = slide_name if slide_name else 'Slide'
    fig.suptitle(img_title, fontsize=13, fontweight="bold", y=1.01)

    composite = make_composite(masks)
    axes[0].imshow(composite)
    axes[0].set_title("Composite\n(RGB fusion)", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    for i, (cname, cmap) in enumerate(zip(CLASS_NAMES, colormaps)):
        ax = axes[i + 1]
        ax.imshow(masks[cname], cmap=cmap, vmin=0, vmax=1)
        ax.set_title(cname, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    return fig

def make_image_furthest_tissue(
        masks: dict, bed_center: np.ndarray,
        tissue_objs: pd.DataFrame, title: str, tissue_name: str
) -> Figure:
    """
        Make a colored image of a slide based of its annotation masks, and show the furthest tissue point.

    Parameters
    ----------
    masks : dict
        the masks for every tissue type (tumor, necrosis, stroma)
    bed_center : np.ndarray
        the coordinates of the bed center.
    tissue_objs : pd.DataFrame
        some information about the slide.
    title : str
        image title
    tissue_name : str
        name of the tissue (tumor, necrosis, stroma)

    Returns
    -------
        fig : Figure
    """
    img_rgb = make_composite(masks)
    bed_mask = masks['tumor_bed']

    rows = np.any(bed_mask, axis=1)
    cols = np.any(bed_mask, axis=0) 
    if rows.any():
        y_min, y_max = np.argmax(rows), len(rows) - 1 - np.argmax(rows[::-1])
        x_min, x_max = np.argmax(cols), len(cols) - 1 - np.argmax(cols[::-1])
    else:
        y_min, y_max, x_min, x_max = 0, bed_mask.shape[0], 0, bed_mask.shape[1]
    h_pad = (y_max - y_min) * 0.30
    w_pad = (x_max - x_min) * 0.30

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(img_rgb)
    ax.set_xlim(x_min - w_pad, x_max + w_pad)
    ax.set_ylim(y_max + h_pad, y_min - h_pad)
    ax.set_title(title, fontsize=14, fontweight='bold', y=0.9)

    ax.plot(bed_center[1], bed_center[0], marker="*", color="black",
             ms=10, mew=0.2, mec='black', zorder=5, label='tumor bed center')

    y, x = tissue_objs['periph-coord']
    max_dist = tissue_objs['dist-center']

    ax.plot(x, y, marker="v", color="yellow", ms=5, markeredgecolor='black')

    info = f"Furthest {tissue_name}\nd={max_dist:.1f}mm"
    ax.annotate(
        info, xy=(x, y), xytext=(12, 12), textcoords="offset points",
        fontsize=7, color="white",
        bbox=dict(boxstyle="round,pad=0.2", fc='black', alpha=0.6),
        arrowprops=dict(arrowstyle="-", color='black', lw=0.8),
    )

    legend_elements = [
        mpatches.Patch(color='red', label='tumeur'),
        mpatches.Patch(color='green', label='stroma'),
        mpatches.Patch(color='blue', label='nécrose'),
        Line2D([0], [0], marker='*', color='w', label='centre du lit',
            markerfacecolor='black', markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='v', color='w', label='tumeur la plus éloignée',
            markerfacecolor='yellow',  markeredgecolor='black', markersize=10),
    ]
    ax.legend(
        handles=legend_elements,
        loc='lower right',      # Position (ex: 'best', 'lower right', 'center')
        fontsize='medium',     # Taille de la police
        frameon=True,          # Afficher ou non le cadre
        shadow=True,           # Ajouter une ombre
        facecolor='white',     # Couleur de fond du bloc
        edgecolor='black',     # Couleur de la bordure
        ncol=2
    )

    ax.axis("off")
    return fig

def make_image(masks: dict, title: str) -> Figure:
    """
        Create a color image by layering the masks.

    Parameters
    ----------
    masks : dict
        the masks for every tissue type (tumor, necrosis, stroma).
    title : str
        image title.

    Returns
    -------
        fig : Figure
    """
    img_rgb = make_composite(masks)
    img_rgb = (img_rgb * 255).astype(np.uint8)

    coords = np.argwhere(masks['tumor_bed'])
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    h_pad = (y_max - y_min) * 0.30
    w_pad = (x_max - x_min) * 0.30

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(img_rgb)
    ax.set_xlim(x_min - w_pad, x_max + w_pad)
    ax.set_ylim(y_max + h_pad, y_min - h_pad)
    ax.set_title(title, fontsize=12, fontweight='bold', y=0.9)

    legend_elements = [
        mpatches.Patch(color='red', label='tumeur'),
        mpatches.Patch(color='green', label='stroma'),
        mpatches.Patch(color='blue', label='nécrose'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='lower right',      # Position (ex: 'best', 'lower right', 'center')
        fontsize='medium',     # Taille de la police
        frameon=True,          # Afficher ou non le cadre
        shadow=True,           # Ajouter une ombre
        facecolor='white',     # Couleur de fond du bloc
        edgecolor='black',     # Couleur de la bordure
        ncol=2
    )

    ax.axis("off")
    return fig

def export_mask_pdf(slide_annotations: dict, output="masques.pdf", saveimg=False) -> None:
    """
        Export for each slide the display its masks.

    Parameters
    ----------
    slide_annotations : dict [str, dict]
    output : str
        name of the output pdf file.
    saveimg : bool
        if True, save the image for each slide.

    Returns
    -------
        None
    """

    plt.ioff() # Disable interactive mode.
    
    if saveimg:
        savedir = f"./{output.split('.')[0]}"
        os.makedirs(savedir, exist_ok=True)
    else: savedir = "."

    with PdfPages(output) as pdf:
        for sld_name in slide_annotations:
            masks = slide_annotations[sld_name]
            fig = make_image_masks(masks, sld_name)
            pdf.savefig(fig, bbox_inches="tight", dpi=150)
            if saveimg:
                fig.savefig(f"{savedir}/img_{sld_name}.png")
            fig.clf()
            plt.close(fig)

        d = pdf.infodict()
        d["Title"]   = "Visualisation masques"

def export_furthest_tissue_pdf(
        slide_annotations: dict, func_find_center, tissues_geometrics: pd.DataFrame,
        tissue_type='tumor', output="furthest_tumors.pdf", saveimg=False
) -> None:
    """
        Export for each slide the colored image with the position of the furthest tissue.

    Parameters
    ----------
    slide_annotations : dict [str, dict]
    func_find_center : callable
        function to estimate the center of the tumor bed.
    tissues_geometrics : pd.DataFrame
        some data about each slide, such as the coordinate and  the distance
        of the furthest tissue from the center of the tumor bed.
    tissue_type : str
    output : str
        name of the output pdf file.
    saveimg : bool
        if True, save the image for each slide.

    Returns
    -------
        None
    """
    plt.ioff()

    if saveimg:
        savedir = f"./{output.split('.')[0]}"
        os.makedirs(savedir, exist_ok=True)
    else: savedir = "."

    with PdfPages(output) as pdf:
        i = 1
        for sld_name in slide_annotations:
            tissues = tissues_geometrics[
                (tissues_geometrics['id-slide']==sld_name)
                & (tissues_geometrics['type']==tissue_type)
            ]
            if tissues.empty:
                continue

            masks = slide_annotations[sld_name]
            tumor_bed = {
                'mask': masks['tumor_bed'],
                'center': func_find_center(masks['tumor_bed'])[0]['center']
            }

            fig = make_image_furthest_tissue(masks, tumor_bed['center'], tissues, "", tissue_type)
            pdf.savefig(
                fig,
                bbox_inches='tight',
                pad_inches=0, 
                dpi=300,            # Increases image quality
                transparent=False
            )
            if saveimg:
                fig.savefig(
                    f"{savedir}/img_{i}.png",
                    transparent=True
                )
            fig.clf()
            plt.close(fig)
            i += 1

        d = pdf.infodict()
        d["Title"]   = "Visualization furthest tumor per slide"

def export_tumor_dispersion_pdf(
        slides: np.ndarray, scores: np.ndarray,
        output="tumors dispersion score.pdf", saveimg=False
) -> None:
    """
        Export for each slide the colored image with the dispersion score.

    Parameters
    ----------
    slides : np.ndarray
        slides annotations with tumor
    scores : np.ndarray
        dispersion scores in the same order as slides
    output : str
        name of the output pdf file.
    saveimg : bool
        if True, save the image for each slide.

    Returns
    -------
        None
    """
    with PdfPages(output) as pdf:
        i = 1
        for (sld_name, masks), score in zip(slides, scores):
    
            if not np.any(masks['tumor']):
                continue

            fig = make_image(masks, f"score : {score:.2f}")
            pdf.savefig(
                fig,
                bbox_inches='tight',
                pad_inches=0, 
                dpi=300,            # Increases image quality
                transparent=False
            )
            if saveimg:
                savedir = f"./{output.split('.')[0]}"
                if not os.path.exists(savedir):
                    os.mkdir(savedir)
                fig.savefig(
                    f"{savedir}/img_{i}.png",
                    transparent=True
                )
            plt.close(fig)
            i += 1

def plot_distribution(
    distrib: dict, show='hist-proportion',
    figsize=(12, 8), suptitle="Titre Général",
    xlabel="Valeurs de x", ylabel="Valeurs de y",
) -> None:
    """
        Plot the distribution of tumor pixels for every slide.

    Parameters
    ----------
    distrib : dict
        data containing a modeling of tumor pixels distribution for each slide.
    show : str
        argument to specify which model of distribution to show (kde or hist-proportion)
    figsize : tuple[int, int]
        size of the figure.
    suptitle : str
        title of the figure.
    xlabel : str
        label of the x-axis.
    ylabel : str
        label of the y-axis.

    Returns
    -------
        None
    """
    n = len(distrib)
    if n == 0: return

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten() if n > 1 else [axes]

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight='bold', color='#222222')

    for i, sld_name in enumerate(distrib.keys()):
        ax = axes[i]
        dist = distrib[sld_name]

        title = sld_name if 'UTC' not in sld_name else sld_name.split('UTC')[0]
        ax.set_title(f"{title}", fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel(xlabel, fontsize=10, color='#444444')
        ax.set_ylabel(ylabel, fontsize=10, color='#444444')
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#888888')
            spine.set_linewidth(1)
        
        ax.grid(True, linestyle='-', alpha=0.3, color='#a0a0a0')
        ax.set_axisbelow(True)
        ax.set_xticks([0, 0.5, 1])

        if show == 'kde':
            x_eval = np.linspace(0, 1.5, num=50)
            ax.plot(x_eval, dist['kde']['center'](x_eval))

        elif show == 'hist-proportion':
            counts = dist['hist-proportion'][0]
            edges = dist['hist-proportion'][1]
            
            ax.stairs(counts, edges, fill=True, alpha=0.65, 
                      edgecolor='black', linewidth=0.8)

        else:
            title = sld_name if 'UTC' not in sld_name else sld_name.split('UTC')[0]
            ax.set_title(f"Lame {title}", fontsize=11, color='#999999', style='italic')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#dddddd')
                spine.set_linestyle('--')
            ax.set_yticks([])
        
            

    for j in range(len(distrib), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

def plot_slide_grid(
        slides: np.ndarray, figsize=(14,15),
        output: str = None, imgtitles: list[str] = None
) -> None:
    """
        Plot the tumor bed images in color in a grid layout.

    Parameters
    ----------
    slides : np.ndarray
        slides annotations with tumor
    figsize : tuple[int, int]
        size of the figure.
    output : str
        name of the output file.
    imgtitles : list[str]
        the list of titles for each slide.

    Returns
    -------
        None
    """
    n = len(slides)

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten() if n > 1 else [axes]


    for i, sld in enumerate(slides):
        _, masks = sld
        ax = axes[i]
        coords = np.argwhere(masks['tumor_bed'])
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        h_pad = (y_max - y_min) * 0.10
        w_pad = (x_max - x_min) * 0.10

        img_rgb = make_composite(masks)
        ax.imshow(img_rgb)

        ax.set_xlim(x_min - w_pad, x_max + w_pad)
        ax.set_ylim(y_max + h_pad, y_min - h_pad)
        ax.set_title(f'{f"slide {i}" if imgtitles is None else imgtitles[i]}', fontdict={'fontsize': 'small', 'fontweight': 'bold'})
        ax.axis('off')

    for k in range(len(slides), len(axes)):
        axes[k].axis('off')
    if output is not  None:
        plt.suptitle(output, fontweight='bold')
        fig.savefig(f'{output}.png')
    else:
        fig.savefig('tumors grid.png')

import math
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

from src.repartition.constants import CLASS_NAMES
from src.repartition.geometry_funcs import find_center_edt_barycentre


def normalize(mask):
    arr = mask.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)

def make_composite(masks):
    stack_img = np.stack([normalize(m) for k, m in masks.items() if k != 'tumor_bed'], axis=-1)
    mask = (stack_img == 0).all(axis=-1)
    for i in range(3):
        stack_img[:,:,i][mask] = 1
    return stack_img

def make_image_figure(masks, slide_name=None):
    COLORMAPS   = ["Blues", "Greens", "Reds", "Purples"]
    n_classes  = len(CLASS_NAMES)
    n_cols     = n_classes + 1   # composite + masques individuels

    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4.5))
    img_title = slide_name if slide_name else 'Slide'
    fig.suptitle(img_title, fontsize=13, fontweight="bold", y=1.01)

    composite = make_composite(masks)
    axes[0].imshow(composite)
    axes[0].set_title("Composite\n(RGB fusion)", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    for i, (cname, cmap) in enumerate(zip(CLASS_NAMES, COLORMAPS)):
        ax = axes[i + 1]
        ax.imshow(masks[cname], cmap=cmap, vmin=0, vmax=1)
        ax.set_title(cname, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    return fig


def make_figure_furthest_tissue(masks, bed_center, tissue_objs, title, tissue_name):
    img_rgb = make_composite(masks)
    coords = np.argwhere(masks['tumor_bed'])
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    h_pad = (y_max - y_min) * 0.30
    w_pad = (x_max - x_min) * 0.30

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(img_rgb)
    ax.set_xlim(x_min - w_pad, x_max + w_pad)
    ax.set_ylim(y_max + h_pad, y_min - h_pad)
    ax.set_title(title, fontsize=14, fontweight='bold', y=0.9)

    ax.plot(bed_center[1], bed_center[0], marker="o", color="white",
             ms=5, mew=0.2, mec='black', zorder=5, label='tumor bed center')

    idx_max = tissue_objs['distance-center'].idxmax()
    y, x = tissue_objs['periph-coord'].loc[idx_max]
    max_dist = tissue_objs['distance-center'].loc[idx_max]    

    ax.plot(x, y, marker="v", color="yellow", ms=5, markeredgecolor='black')

    info = f"Furthest {tissue_name}\nd={max_dist:.1f}mm"
    ax.annotate(
        info, xy=(x, y), xytext=(12, 12), textcoords="offset points",
        fontsize=7, color="white",
        bbox=dict(boxstyle="round,pad=0.2", fc='black', alpha=0.6),
        arrowprops=dict(arrowstyle="-", color='black', lw=0.8),
    )

    legend_elements = [
        mpatches.Patch(color='red', label='tumor'),
        mpatches.Patch(color='green', label='stroma'),
        mpatches.Patch(color='blue', label='necrosis'),
        Line2D([0], [0], marker='o', color='w', label='centre du lit',
            markerfacecolor='white', markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='v', color='w', label='tumeur la plus éloigné',
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

def draw_pie_chart(labels, sizes, colors):    
    fig, ax = plt.subplots()
    ax.pie(sizes, colors = colors,
            labels=labels, autopct='%1.1f%%', 
            startangle=90, textprops={
                'size': 'smaller',
                'rotation': 45,
            }
    )

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    ax.legend()
    plt.tight_layout()
    return fig

def draw_bar_plot(labels, sizes, colors):
    fig, ax = plt.subplots()
    ax.bar(range(len(labels)), sizes, label=labels, color=colors)

    ax.set_ylabel('Pourcentage')
    ax.set_title('Proportion de tissues (nécrose, stroma, tumeur) par lame')
    ax.legend(title='Type')
    return fig

def export_mask_pdf(slide_annotations, output="masques.pdf", saveimg=False):
    with PdfPages(output) as pdf:
        for sld_name in slide_annotations:
            masks = slide_annotations[sld_name]
            fig = make_image_figure(masks, sld_name)
            pdf.savefig(fig, bbox_inches="tight", dpi=150)
            if saveimg:
                savedir = f"./{output.split('.')[0]}"
                if not os.path.exists(savedir):
                    os.mkdir(savedir)
                fig.savefig(f"{savedir}/img_{sld_name}.png")
            plt.close(fig)

        d = pdf.infodict()
        d["Title"]   = "Visualisation masques"

def export_furthest_tissue_pdf(slide_annotations, tissues_geometrics,
                                tissue_type='tumor', output="furthest_tumors.pdf", saveimg=False):
    with PdfPages(output) as pdf:
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
                'center': find_center_edt_barycentre(masks['tumor_bed'])[0]['center']
            }
            
            fig = make_figure_furthest_tissue(masks, tumor_bed['center'], tissues, f"Slide {sld_name}", tissue_type)
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
                    f"{savedir}/img_{sld_name}.png",
                    transparent=True
                )
            plt.close(fig)

        d = pdf.infodict()
        d["Title"]   = "Visualization furthest tumor per slide"

def export_tissues_proportion(all_tissues_areas, output='tissues_proportion.pdf',
                               draw_type='bar', saveimg=False):
     with PdfPages(output) as pdf:
        for sld_name in all_tissues_areas['id-slide'].unique():
            tissue_areas = all_tissues_areas[all_tissues_areas['id-slide']==sld_name]
            label2color = {
                'necrosis': 'blue',
                'stroma': 'green',
                'tumor': 'red'
            }

            sizes, colors, labels = [], [], []
            for l in CLASS_NAMES:
                l_percent = tissue_areas[tissue_areas['type']==l]['area-percentage']
                if not l_percent.empty:
                    sizes.append(l_percent.values[0])
                    colors.append(label2color[l])
                    labels.append(l)

            match draw_type:
                case 'bar':
                    fig = draw_bar_plot(labels, sizes, colors)
                case 'pie':
                    fig = draw_pie_chart(labels, sizes, colors)
                case _:
                    raise NotImplementedError(
                        f"""unknown draw_type '{draw_type}', expected one of:
                        bar, pie."""
                    )
            fig.suptitle(sld_name, fontweight='bold')
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
                fig.savefig(f"{savedir}/img_{sld_name}.png")
            plt.close(fig)

        d = pdf.infodict()
        d["Title"]   = "Visualization tissue proportion per slide"

def plot_distribution(distrib, show='hist-proportion',
                       ref='center', figsize=(12, 8), 
                       suptitle="Titre Général",
                       xlabel="Valeurs de x", 
                       ylabel="Valeurs de y"):
    
    n = len(distrib)
    if n == 0: return
    
    # --- 1. Calcul des échelles globales ---
    global_x_min, global_x_max = float('inf'), float('-inf')
    global_y_max = 0
    
    for sld_name, dist in distrib.items():
        if show == 'hist':
            counts = dist['hist'][ref][0]
            edges = dist['hist'][ref][1]
            if len(counts) > 0:
                global_y_max = max(global_y_max, np.max(counts))
            if len(edges) > 0:
                global_x_min = min(global_x_min, np.min(edges))
                global_x_max = max(global_x_max, np.max(edges))
    # ---------------------------------------
    
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

        if show == 'hist':
            counts = dist['hist'][ref][0]
            edges = dist['hist'][ref][1]
            
            ax.stairs(counts, edges, fill=True, alpha=0.65, 
                      edgecolor='black', linewidth=0.8)
            
            if global_x_min != float('inf'):
                # Marge de 5% sur l'axe X pour aérer
                x_margin = (global_x_max - global_x_min) * 0.05
                x_margin = x_margin if x_margin > 0 else 0.5 
                ax.set_xlim(global_x_min - x_margin, global_x_max + x_margin)
                # Marge de 10% sur l'axe Y au-dessus de la plus haute barre
                #ax.set_ylim(0, global_y_max * 1.1)

        elif show == 'kde':
            x_eval = np.linspace(0, 2, num=100) if ref=='center' else np.linspace(0, 5, num=50)
            ax.plot(x_eval, dist['kde'][ref](x_eval))
        
        elif show == 'hist-proportion':
            counts = dist['hist-proportion'][0]
            edges = dist['hist-proportion'][1]
            
            ax.stairs(counts, edges, fill=True, alpha=0.65, 
                      edgecolor='black', linewidth=0.8)
            
            """ if global_x_min != float('inf'):
                x_margin = (global_x_max - global_x_min) * 0.05
                x_margin = x_margin if x_margin > 0 else 0.5 
                ax.set_xlim(global_x_min - x_margin, global_x_max + x_margin)
                ax.set_ylim(0, global_y_max * 1.1) """
        else:
            title = sld_name if 'UTC' not in sld_name else sld_name.split('UTC')[0]
            ax.set_title(f"Lame {title}", fontsize=11, color='#999999', style='italic')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#dddddd')
                spine.set_linestyle('--')
            ax.set_xticks([])
            ax.set_yticks([])

    for j in range(len(distrib), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

def plot_slide_grid(slides, figsize=(14,14), title=None):
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
        ax.set_title(f'slide {i}', fontweight='bold')
        ax.axis('off')

    for k in range(len(slides), len(axes)):
        axes[k].axis('off')
    if title is None:
        fig.savefig('slides_tumeurs.png')
    else:
        fig.savefig(title+'.png')

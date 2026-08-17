
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
import tifffile

from src.repartition.constants import CLASSES, COLOR2LABEL, MULTI_SLIDES_PATIENTS
from src.repartition.data_analysis import id_patient

def is_valid_mask(mask, num_class=10):
    return np.all(mask < num_class)

def make_valid_mask(mask):
    new_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

    for color, label in COLOR2LABEL.items():
        match = np.all(mask == color, axis=-1)
        new_mask[match] = label
    return new_mask

def smooth_combined_mask(mask, close_radius=10):
    global_mask = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1)
    )
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_CLOSE, kernel)
    global_mask = binary_fill_holes(global_mask)
    return global_mask

def is_jagged(mask, crack_width=15, threshold=0.02):
    bg = mask == False
    mask = (mask * 1).astype(np.uint8)
    bg = (bg * 1).astype(np.uint8)

    r = crack_width

    # Directional kernels: each looks only in ONE direction
    k_left  = np.zeros((1, 2*r+1), np.uint8); k_left[0,  :r]   = 1
    k_right = np.zeros((1, 2*r+1), np.uint8); k_right[0, r+1:] = 1
    k_up    = np.zeros((2*r+1, 1), np.uint8); k_up[  :r,  0]   = 1
    k_down  = np.zeros((2*r+1, 1), np.uint8); k_down[r+1:, 0]  = 1

    fg_left  = cv2.dilate(mask, k_left)
    fg_right = cv2.dilate(mask, k_right)
    fg_up    = cv2.dilate(mask, k_up)
    fg_down  = cv2.dilate(mask, k_down)

    # A crack pixel = background with foreground on BOTH opposite sides
    h_crack = (bg == 1) & (fg_left == 1) & (fg_right == 1)
    v_crack = (bg == 1) & (fg_up   == 1) & (fg_down  == 1)
    crack_pixels = h_crack | v_crack

    crack_ratio = crack_pixels.sum() / max(1, mask.sum())
    return crack_ratio > threshold, crack_ratio

def is_mask_to_substract(mask1, mask2, kernel_size=3):
    intersection = mask1 & mask2
    kernel = np.ones((kernel_size, kernel_size), np.uint8) # A 3x3 kernel looks at the immediate 1-pixel border
    dilated_intersect = cv2.dilate(intersection, kernel, iterations=1)

    surround_zone = cv2.subtract(dilated_intersect, intersection)

    surrounding_pixels = mask1[surround_zone > 0]
    is_surrounded_by_background = np.all(surrounding_pixels == 0)

    return is_surrounded_by_background

def tif_to_real_mask(img_tif):
    masks = dict()
    for cls in CLASSES:
        msk = img_tif[CLASSES[cls],:,:]
        for other_cls in CLASSES:
            other_msk = img_tif[CLASSES[other_cls],:,:]
            if (other_cls != cls) and (not is_mask_to_substract(msk, other_msk)):
                msk = msk & ~other_msk
        masks.update(
            {cls: msk}
        )
    bed_mask = np.any(img_tif[:,:,:], axis=0)
    masks.update(
        {'tumor_bed': smooth_combined_mask(bed_mask.astype(np.uint8))}
    )
    return masks

def tif_to_filled_mask(img_tif):
    masks = dict()
    for cls in CLASSES:
        masks.update(
            {cls: img_tif[CLASSES[cls],:,:]}
        )
    bed_mask = np.any(img_tif[:,:,:], axis=0)
    masks.update(
        {'tumor_bed': smooth_combined_mask(bed_mask) if is_jagged(bed_mask) 
         else bed_mask}
    )
    return masks

def get_annotation(path2annot, tif_to_mask):
    slides_annot = dict()
    puzzle_slides = {p:{} for p in MULTI_SLIDES_PATIENTS}
    for annot_pth in path2annot:
        img = tifffile.imread(annot_pth)
        masks = tif_to_mask(img)

        slide_name = annot_pth.split("/")[-1].split("-")[0]
        slides_annot.update({
            slide_name: masks
        })

        id = id_patient(slide_name)
        if id in MULTI_SLIDES_PATIENTS:
            puzzle_slides[id].update({
                slide_name: img
            })

    return slides_annot, puzzle_slides

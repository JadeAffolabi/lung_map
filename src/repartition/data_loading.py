
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
import tifffile


from src.repartition.constants import CLASSES

def smooth_combined_mask(mask: np.ndarray, close_radius: int = 10) -> np.ndarray:
    """
    Combined the cracked parts of the original mask to make one seamless whole.

    Parameters
    ----------
    mask : np.ndarray
        Cracked mask.
    close_radius : int (optional)
        Radius of the kernel.

    Returns
    -------
    smoothed_mask : np.ndarray
    """
    global_mask = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1)
    )
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_CLOSE, kernel)
    global_mask = binary_fill_holes(global_mask)
    return global_mask

def is_cracked(mask: np.ndarray, crack_width=15, threshold=0.02) -> bool:
    """
    Check if a mask is cracked. A cracked mask is one where the object of interest is supposed to be
    one whole but the mask is in fact has more than one component. Each component is separated
    by a small empty space "crack".

    Parameters
    ----------
    mask : np.ndarray
        mask to be checked
    crack_width : int (optional)
        the maximum width of a crack.
    threshold : float (optional)
        threshold to determine if a mask is cracked.
        The crack pixels ratio is compared to that threshold.

    Returns
    -------
        bool
    """
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
    return crack_ratio > threshold

def is_nested_mask(
        mask1: np.ndarray, mask2: np.ndarray, tol=1e-1, kernel_size=3) -> bool:
    """
    Check if mask1 is contained in mask2.

    Parameters
    ----------
    mask1 : np.ndarray
    mask2 : np.ndarray
    tol : float (optional)
        threshold of the ratio of foreground to background.
        If the ratio is lower than that value, the returned value is True, else False.
    kernel_size : int (optional)
        size of the kernel used to dilate the intersection between mask1 and mask2.

    Returns
    -------
        bool
    """
    intersection = mask1 & mask2 
    kernel = np.ones((kernel_size, kernel_size), np.uint8) 
    dilated_intersect = cv2.dilate(intersection, kernel, iterations=1)

    surround_zone = dilated_intersect & ~intersection

    surrounding_pixels = mask1[surround_zone > 0]
    pxl_val, counts = np.unique(surrounding_pixels, return_counts=True)
    if len(pxl_val) == 2:
        return counts[1]/counts[0] < tol
    elif pxl_val in [255, 1]:
        return False
    else:
        return True

def refine_masks(raw_masks: np.ndarray) -> dict:
    """
    Refine the masks by applying a substraction when a mask from one class
    is contained within a mask from another class.

    Parameters
    ----------
    raw_masks : np.ndarray

    Returns
    -------
        new_masks : dict[str, np.ndarray]
    """
    masks = dict()
    for cls in CLASSES:
        msk = raw_masks[CLASSES[cls], :, :]
        msk = (msk != 0).astype(np.uint8) # security
        for other_cls in CLASSES:
            if other_cls == cls:
                continue
            other_msk = raw_masks[CLASSES[other_cls], :, :]
            other_msk = (other_msk != 0).astype(np.uint8) # security
            if np.any(msk & other_msk) and (not is_nested_mask(msk, other_msk)):
                msk = msk & ~other_msk
        masks.update(
            {cls: msk}
        )
    bed_mask = np.any(raw_masks[:, :, :], axis=0)
    masks.update(
        {'tumor_bed': smooth_combined_mask(bed_mask.astype(np.uint8))}
    )
    return masks

def get_annotation(path2annot: np.ndarray) -> dict:
    """
    Load the images annotation from the list of paths.
    Parameters
    ----------
    path2annot : np.ndarray
        list of annotation paths.

    Returns
    -------
        slides_annot : dict[str, dict]
            dictionary containing the slide name as key and the multiarray masks as value.
    """
    slides_annot = dict()
    for annot_pth in path2annot:
        slide_name = annot_pth.split("/")[-1].split("-")[0]
        raw_masks = tifffile.imread(annot_pth)
        masks = refine_masks(raw_masks)

        slides_annot.update({
            slide_name: masks
        })

    return slides_annot

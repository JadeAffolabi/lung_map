from abc import ABC, abstractmethod
import argparse
from collections import defaultdict
import glob
import os
import random
import shutil
import time
from typing import Union
import tifffile
import cv2
import glob
import webdataset as wds
from zipfile import ZipFile

import numpy as np
from PIL import Image

from src.segmentation.constants import RAW_DATA_DIR, DIR_BCSS, DIR_LUAD, DIR_LCHUR, \
CLASS_CONVERSION_BCSS, CLASS_CONVERSION_LUAD, CLASSES_TO_LABELS, PATH_SEG_DATA
from src.repartition.data_loading import tif_to_real_mask
from src.segmentation.constants import CLASSES_TO_LABELS

class DataPaths(ABC):
    img_paths: list
    mask_paths: list

    @abstractmethod
    def extract_patient_id(self, filepath) -> str:
        pass

class LuadPaths(DataPaths):
    def __init__(self):
        self.img_paths, self.mask_paths = self._get_data_paths_luad()

    def _get_data_paths_luad(self):
        img_paths = []
        mask_paths = []
        for split in ['val', 'test']:
            img_paths.extend(glob.glob(os.path.join(RAW_DATA_DIR, DIR_LUAD, split, "img", "*.png")))
            mask_paths.extend(glob.glob(os.path.join(RAW_DATA_DIR, DIR_LUAD, split, "mask", "*.png")))
        img_paths.sort()
        mask_paths.sort()

        for img_pth, msk_pth in zip(img_paths, mask_paths):
            img_name = os.path.splitext(os.path.basename(img_pth))[0]
            msk_name = os.path.splitext(os.path.basename(msk_pth))[0]
            assert img_name == msk_name, \
            f"Misalignement of image ({img_name}) and mask ({msk_name})."
        return img_paths, mask_paths
    
    def extract_patient_id(self, filepath):
        filename = os.path.basename(filepath)
        id = filename.split('-')[0]
        if id.isdigit():
            return id
        return "UNKNOWN"

class BcssPaths(DataPaths):
    def __init__(self):
        self.img_paths, self.mask_paths = self._get_data_paths_bcss()

    def _get_data_paths_bcss(self):
        img_paths = glob.glob(os.path.join(RAW_DATA_DIR, DIR_BCSS,  "rgbs_colorNormalized", "*.png"))
        mask_paths = glob.glob(os.path.join(RAW_DATA_DIR, DIR_BCSS, "masks", "*.png"))
        img_paths.sort()
        mask_paths.sort()

        for img_pth, msk_pth in zip(img_paths, mask_paths):
            img_name = os.path.splitext(os.path.basename(img_pth))[0]
            msk_name = os.path.splitext(os.path.basename(msk_pth))[0]
            assert img_name == msk_name, \
            f"Misalignement of image ({img_name}) and mask ({msk_name})."
        return img_paths, mask_paths

    def extract_patient_id(self, filepath):
        filename = os.path.basename(filepath)
        tcga_barcode = filename.split('_')[0]
        code_parts = tcga_barcode.split('-')
        if len(code_parts) >= 3 and code_parts[0] == 'TCGA':
            return "-".join(code_parts[:3]) 
        return "UNKNOWN"

class LchurPaths(DataPaths):
    def __init__(self):
        self.img_paths, self.mask_paths = self._get_data_paths_lchur()

    def _get_data_paths_lchur(self):
        img_paths = glob.glob(os.path.join(RAW_DATA_DIR, DIR_LCHUR, "imgs", "*.png"))
        mask_paths = glob.glob(os.path.join(RAW_DATA_DIR, DIR_LCHUR, "masks", "*.tif"))
        img_paths.sort()
        mask_paths.sort()

        for img_pth, msk_pth in zip(img_paths, mask_paths):
            img_name = os.path.splitext(os.path.basename(img_pth))[0]
            msk_name = os.path.splitext(os.path.basename(msk_pth))[0]
            assert img_name == msk_name, \
            f"Misalignement of image ({img_name}) and mask ({msk_name})."
        return img_paths, mask_paths

    def extract_patient_id(self, filepath):
        filename = os.path.basename(filepath)
        patient_id = filename.split('_')[0]
        if 'P' in patient_id:
            return patient_id
        return "UNKNOWN"

def extract_from_zip(zip_dir):
    archives_path = glob.glob(os.path.join(zip_dir, "*.zip")) 
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    for arch_pth in archives_path:
        with ZipFile(arch_pth, 'r') as zObject:
            zObject.extractall(path=RAW_DATA_DIR)

def patient_level_split(data_paths: DataPaths, train_ratio=0.8,
                        val_ratio=None):
    patient_to_indices = defaultdict(list)
    
    for i, path in enumerate(data_paths.img_paths):
        patient_id = data_paths.extract_patient_id(path)
        patient_to_indices[patient_id].append(i)
        
    unique_patients = list(patient_to_indices.keys())
    random.shuffle(unique_patients)

    if train_ratio > 1:
        raise ValueError('train_ratio must be less than 1')
    
    split_idx = int(len(unique_patients) * train_ratio)
    train_patients = set(unique_patients[:split_idx])
    
    if val_ratio:
        if val_ratio > 1:
            raise ValueError('val_ratio must be less than 1')
        split_idx = int(len(train_patients) * val_ratio)
        val_patients = set(list(train_patients)[:split_idx])
    else:
        val_patients = set()
    
    train_patients -= val_patients

    train_dict = {'img_paths': [], 'mask_paths': []}
    val_dict = {'img_paths': [], 'mask_paths': []}
    test_dict = {'img_paths': [], 'mask_paths': []}

    for patient_id, indices in patient_to_indices.items():
        if patient_id in train_patients:
            target_dict = train_dict
        elif patient_id in val_patients:
            target_dict = val_dict
        else:
            target_dict = test_dict

        for i in indices:
            target_dict['img_paths'].append(data_paths.img_paths[i])
            target_dict['mask_paths'].append(data_paths.mask_paths[i])
                
    # Shuffle all patches in order to shuffle patients data
    for final_dict in (train_dict, val_dict, test_dict):
        total_patches = len(final_dict['img_paths'])
        shuffle_indices = list(range(total_patches))
        random.shuffle(shuffle_indices)
        for key in final_dict.keys():
            final_dict[key] = [final_dict[key][i] for i in shuffle_indices]
 
    return train_dict, test_dict, val_dict

def merge_dict_of_list(list_of_dicts):

    if not list_of_dicts:
        return {}, {}

    keys = list_of_dicts[0].keys()
    merged_dict = {key: [] for key in keys}

    for d in list_of_dicts:
        for key in keys:
            merged_dict[key].extend(d[key])

    first_key = list(keys)[0]
    total_samples = len(merged_dict[first_key])
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    shuffled_merge_dict = {
        key: [merged_dict[key][i] for i in indices] 
        for key in keys
    } 
    return shuffled_merge_dict

def merge_split(datapaths, train_ratio=0.8, val_ratio=None):
    train_dict_list = []
    test_dict_list = []
    val_dict_list = []
    
    for d in datapaths:
        train_dict, test_dict, val_dict = patient_level_split(d, train_ratio, val_ratio)
        train_dict_list.append(train_dict)
        test_dict_list.append(test_dict)
        if val_ratio:
            val_dict_list.append(val_dict)
    
    final_train_dict = merge_dict_of_list(train_dict_list)
    final_test_dict = merge_dict_of_list(test_dict_list)
    final_val_dict = merge_dict_of_list(val_dict_list) if val_ratio else {}

    return final_train_dict, final_test_dict, final_val_dict

def img_mask_correct_order(path_dict):
    for img_pth, msk_pth in zip(
        path_dict['img_paths'], 
        path_dict['mask_paths']
        ):
        img_name = os.path.splitext(os.path.basename(img_pth))[0]
        msk_name = os.path.splitext(os.path.basename(msk_pth))[0]
        if img_name != msk_name:
            return False
    return True

def class_mask(cls, img_mask, class_mapping):
    labels = class_mapping[cls]
    mask_list = [img_mask == l for l in labels]

    return np.any(np.array(mask_list), axis=0)

def adapt_mask(mask, cls_map):
    new_mask = np.zeros_like(mask)

    tum_mask = class_mask('tumor', mask, cls_map)
    necr_mask = class_mask('necrosis', mask, cls_map)
    str_mask = class_mask('stroma', mask, cls_map)

    new_mask[tum_mask] = CLASSES_TO_LABELS['tumor']
    new_mask[necr_mask] = CLASSES_TO_LABELS['necrosis']
    new_mask[str_mask] = CLASSES_TO_LABELS['stroma']

    return new_mask

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s           = pts.sum(axis=1)
    rect[0]     = pts[np.argmin(s)]    # top-left     (x+y minimal)
    rect[2]     = pts[np.argmax(s)]    # bottom-right (x+y maximal)
    diff        = np.diff(pts, axis=1)
    rect[1]     = pts[np.argmin(diff)] # top-right    (y-x minimal)
    rect[3]     = pts[np.argmax(diff)] # bottom-left  (y-x maximal)
    return rect

def compute_output_dims(src_pts: np.ndarray) -> tuple[int, int]:
    # tl = top-left, br = bottom-right, etc.
    tl, tr, br, bl = src_pts

    width_top    = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    dst_w        = int(max(width_top, width_bottom))

    height_left  = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    dst_h        = int(max(height_left, height_right))

    return dst_w, dst_h

def extract_annotated_region(
    image: np.ndarray,
    mask: np.ndarray,
    exclude_label: int = 0,
    kernel_size: int = 25,
):

    valid_map = (mask != exclude_label).astype(np.uint8) * 255
    kernel       = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,) * 2)
    valid_closed = cv2.morphologyEx(valid_map, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(valid_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Aucune zone valide trouvée.")

    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea)) # find min enclosing rect
    box  = cv2.boxPoints(rect)  # 4 vertices of the rotated rect

    src_pts       = order_points(box)
    dst_w, dst_h  = compute_output_dims(src_pts)

    dst_pts = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32
    )

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped_mask = cv2.warpPerspective(
        mask, M, (dst_w, dst_h),
        flags=cv2.INTER_NEAREST, # There is no interpolation
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=exclude_label
    )

    warped_image = cv2.warpPerspective(
        image, M, (dst_w, dst_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    warped_valid = cv2.warpPerspective(
        valid_map, M, (dst_w, dst_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Clean borders
    oob = warped_valid == 0
    warped_mask[oob]  = exclude_label
    warped_image[oob] = 0

    original_labels = set(np.unique(mask).tolist())
    warped_labels   = set(np.unique(warped_mask).tolist())
    false_labels        = warped_labels - original_labels
    if false_labels:
        raise RuntimeError(f"Labels parasites créés : {false_labels}")

    return warped_image, warped_mask

def is_luad(img_path):
    return 'luad' in img_path.lower()

def is_bcss(img_path):
    return 'tcga' in img_path.lower()

def is_lchur(img_path):
    return 'lchur' in img_path.lower()

def is_partially_annot(mask):
    return np.any(mask == 0)

def extract_patches(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: Union[int, None]=224,
    other_label: int = 0,
    min_valid_ratio: float = 0.1,
) -> list[dict]:

    H, W = mask.shape
    patches = []
    if patch_size:
        for y in range(0, H - patch_size + 1, patch_size):
            for x in range(0, W - patch_size + 1, patch_size):
                img_patch  = image[y : y+patch_size, x : x+patch_size]
                mask_patch = mask [y : y+patch_size, x : x+patch_size]

                if np.mean(mask_patch != other_label) < min_valid_ratio:
                    continue

                patches.append({
                    "img"   : img_patch,
                    "mask"    : mask_patch,
                    "pos": (y, x),
                })

    return patches

def data_adaptation(img, mask, img_path, patch_size: Union[int, None]=224):
    result = []

    if is_luad(img_path):
        new_mask = adapt_mask(mask, CLASS_CONVERSION_LUAD)
        result.append({
            "img": img,
            "mask": new_mask,
            "pos": None
        })

    if is_bcss(img_path):
        if is_partially_annot(mask):
            img, mask = extract_annotated_region(
                img,
                mask,
                exclude_label=0,
                kernel_size=25
            )
        
        new_mask = adapt_mask(mask, CLASS_CONVERSION_BCSS)

        # BCSS magnification is 40x and LUAD is 10x
        """ reduc_factor = 40 / 10
        old_size = new_mask.shape               # (H, W)
        size_10x = (                            # (W, H) for cv2
            int(old_size[1] / reduc_factor),
            int(old_size[0] / reduc_factor)
        )
        img_10x = cv2.resize(
            img,
            size_10x,
            interpolation=cv2.INTER_CUBIC
        )
        mask_10x = cv2.resize(
            new_mask,
            size_10x,
            interpolation=cv2.INTER_NEAREST
        )

        result = extract_patches(
            img_10x, mask_10x, patch_size=patch_size,
            other_label=CLASSES_TO_LABELS['other'],
            min_valid_ratio=0.20
        ) """
        result.append({
            "img": img,
            "mask": new_mask,
            "pos": None,
        })
    
    if is_lchur(img_path):
        result.append({
            "img": img,
            "mask": mask,
            "pos": None,
        })

    return result



def create_dataset_shards(split_name, data_paths, out_dir, max_size=1e9, patch_size: Union[int, None]=224):

    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, f"{split_name}-%06d.tar")
    
    with wds.writer.ShardWriter(pattern, maxsize=max_size) as sink:
        for img_path, mask_path in zip(
            data_paths['img_paths'],
            data_paths['mask_paths']
        ):
            base_name = os.path.splitext(os.path.basename(img_path))[0].replace('.', '_')
    
            img = np.array(Image.open(img_path).convert("RGB"))

            if is_lchur(mask_path):
                mask_tif = tifffile.imread(mask_path)
                annotations = tif_to_real_mask(mask_tif)
                mask = np.zeros((img.shape[0], img.shape[1]))
                for cls in CLASSES_TO_LABELS:
                    if cls != 'other':
                        mask[annotations[cls] == 255] = CLASSES_TO_LABELS[cls]
            else:
                mask  = np.array(Image.open(mask_path).convert("P"))

            assert mask.shape[:2] == img.shape[:2], \
            f"Dimensions incohérentes image/masque \n{img_path}\n{mask_path}"

            new_data = data_adaptation(img, mask, img_path, patch_size=patch_size)

            for i, patch in enumerate(new_data):
                key_name = f"{base_name}_yx_{patch['pos'][0]}_{patch['pos'][1]}" \
                            if patch['pos'] else base_name

                sample = {
                    "__key__": key_name,
                    "image.png": patch['img'].astype(np.uint8),
                    "mask.png": patch['mask'].astype(np.uint8)
                }

                sink.write(sample)

# The script must run at the root of the project, due to path management

if __name__ == '__main__':

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-z', '--zip_path', help="Path the files of the raw data", type=str)
    args = parser.parse_args()

    path_to_zip = args.zip_path
    print(f"Extracting data from {path_to_zip}")
    extract_from_zip(path_to_zip)

    #luadpaths = LuadPaths()
    bcsspaths = BcssPaths()
    #lchurpaths = LchurPaths()

    print("Merging and splitting datasets")
    #train_dict, test_dict, val_dict = merge_split([luadpaths, bcsspaths], train_ratio=0.9, val_ratio=0.1)
    train_ratio=0.9
    val_ratio=0.1
    train_dict, test_dict, val_dict = patient_level_split(bcsspaths, train_ratio, val_ratio)

    assert img_mask_correct_order(train_dict) \
        & img_mask_correct_order(test_dict) \
        & img_mask_correct_order(val_dict) \
        ,"Images and masks are not in correct order"

    shard_dir = '/shards_bcss'
    patch_size = None

    print("Creating shards")

    create_dataset_shards(
        'train-full',
        train_dict,
        str(PATH_SEG_DATA) + shard_dir,
        patch_size=patch_size,
    )

    create_dataset_shards(
        'test',
        test_dict,
        str(PATH_SEG_DATA) + shard_dir,
        patch_size=patch_size,
    )

    if val_dict:
        create_dataset_shards(
            'val',
            val_dict,
            str(PATH_SEG_DATA) + shard_dir,
            patch_size=patch_size,
        )

    end = time.time()
    shutil.rmtree(RAW_DATA_DIR)
    print("Done.")
    print(f"Execution time : {(end - start)/60} min")

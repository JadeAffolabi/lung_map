
import glob
from typing import Union
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import webdataset as wds
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import io
from PIL import Image
from skimage.color import rgb2hed, hed2rgb
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random

from src.segmentation.constants import PATH_SEG_DATA

def custom_decoder(key, data):
    if key.endswith("image.png"):
        return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    elif key.endswith("mask.png"):
        return np.array(Image.open(io.BytesIO(data)).convert("P"))
    return None

def extract_np(img_np, mask_np, cy, cx, size, base_size, H, W):
    half = size // 2
    top    = cy - half
    left   = cx - half
    bottom = cy + half
    right  = cx + half

    pad_t = max(0, -top)
    pad_b = max(0, bottom - H)
    pad_l = max(0, -left)
    pad_r = max(0, right - W)

    # Pad in numpy — always contiguous by definition
    if any([pad_t, pad_b, pad_l, pad_r]):
        img_np = np.pad(img_np,
                        ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                        mode='reflect')
        if mask_np is not None:
            mask_np = np.pad(mask_np,
                             ((pad_t, pad_b), (pad_l, pad_r)),
                             mode='constant', constant_values=0)

    # Crop
    t = top + pad_t
    l = left + pad_l
    img_crop  = img_np[t:t+size, l:l+size, :]         # [size, size, 3] uint8
    mask_crop = mask_np[t:t+size, l:l+size] if mask_np is not None else None

    # Resize if needed
    if size != base_size:
        img_crop = np.array(
            Image.fromarray(img_crop).resize((base_size, base_size), Image.BILINEAR)
        )
        if mask_crop is not None:
            mask_crop = np.array(
                Image.fromarray(mask_crop).resize((base_size, base_size), Image.NEAREST)
            )

    return img_crop, mask_crop


def multiscale_patch_generator(src, num_patches=5, base_size=224, mode='random',
                               rare_class_ids=None, rare_class_prob=0.5,
                               max_grid_patchs=None):
    for key, img, mask in src:
        img  = np.ascontiguousarray(img)   # HWC uint8
        mask = np.ascontiguousarray(mask)  # HW uint8
        H, W = img.shape[:2]
        max_half = base_size*4 // 2

        # Rare pixels index — in numpy
        rare_pixels = None
        if rare_class_ids:
            ys, xs = np.where(np.isin(mask, rare_class_ids))
            if len(ys) > 0:
                rare_pixels = list(zip(ys.tolist(), xs.tolist()))

        # Compute centers
        if mode == 'random':
            centers = []
            for _ in range(num_patches):
                if rare_pixels and random.random() < rare_class_prob:
                    cy, cx = random.choice(rare_pixels)
                    cy = int(np.clip(cy, max_half, H - max_half))
                    cx = int(np.clip(cx, max_half, W - max_half))
                else:
                    cy = random.randint(max_half, max(max_half, H - max_half))
                    cx = random.randint(max_half, max(max_half, W - max_half))
                centers.append((cy, cx))

        elif mode == 'grid':
            step = base_size
            centers = [
                (cy, cx)
                for cy in range(max_half, H - max_half, step)
                for cx in range(max_half, W - max_half, step)
            ]
            if max_grid_patchs and len(centers) > max_grid_patchs:
                indices = np.linspace(0, len(centers) - 1, max_grid_patchs, dtype=int)
                centers = [centers[i] for i in indices]
        else:
            centers = [(H // 2, W // 2)]

        for cy, cx in centers:
            s1, m = extract_np(img, mask, cy, cx, base_size,     base_size, H, W)
            s2, _ = extract_np(img, None, cy, cx, base_size * 2, base_size, H, W)
            s3, _ = extract_np(img, None, cy, cx, base_size * 4, base_size, H, W)

            yield key, s1, s2, s3, m

class HEDStainJitter(A.ImageOnlyTransform):
    def __init__(self, strength=0.05, p=0.5):
        super().__init__(p=p)
        self.strength = strength
    
    def get_params(self):
        return {"shift": np.random.uniform(-self.strength, self.strength, size=3)}

    def apply(self, img, shift=None, **params):
        if img.dtype == np.uint8:
            img_float = img.astype(np.float32) / 255.0
            return_uint8 = True
        else:
            img_float = img.astype(np.float32)
            return_uint8 = False

        hed = rgb2hed(img_float)
        for i in range(3):
            hed[:, :, i] += shift[i]

        rgb = np.clip(hed2rgb(hed), 0, 1)

        if return_uint8:
            return (rgb * 255).astype(np.uint8)
        return rgb

    def get_transform_init_args_names(self):
        return ("strength",)

def get_wsi_transforms(mode: str):
    assert mode in ("train", "eval")

    shared = [
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std =(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ]
    
    target_mapping = {'image2': 'image', 'image3': 'image'}

    if mode == "eval":
        return A.Compose(shared, additional_targets=target_mapping)

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),                 # rotations à 0/90/180/270°
        A.ElasticTransform(                      # déformation du tissu
            alpha=120, sigma=10,
            p=0.2,
        ),

        # ── Couleur / coloration ───────────────────────────────────────────────
        HEDStainJitter(strength=0.05, p=0.2),   # variation inter-scanner (priorité haute)
        # ── Artefacts scanner ──────────────────────────────────────────────────
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),   # variation de mise au point
        *shared,
    ], additional_targets=target_mapping)

def get_class_weight_from_shards(shard_urls, num_classes, norm=True, square_root=False):

    label_count = np.zeros(num_classes, dtype=np.int64)

    dataset = (
        wds.WebDataset(shard_urls, shardshuffle=False, empty_check=False)
        .decode(custom_decoder)
        .to_tuple("mask.png")
    )

    for (mask,) in dataset:
        labels, counts = np.unique(mask, return_counts=True)
        for l, c in zip(labels, counts):
            if l < num_classes:
                label_count[l] += c

    return _compute_weights(label_count, norm, square_root)


def _compute_weights(label_count, norm=True, square_root=False):
    total = label_count.sum()
    freq = label_count / total
    
    inv_freq = np.where(freq > 0, 1.0 / freq, 0.0)

    if norm:
        mean = inv_freq[inv_freq > 0].mean()
        weights = inv_freq / mean
        if square_root:
            weights = np.sqrt(weights)
    else:
        weights = inv_freq

    return weights.astype(np.float32)

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, train_urls: Union[str, list[str]],
                 val_urls: Union[str, list[str]],
                 test_urls: Union[str, list[str]],
                 n_images: int,
                 batch_size: int = 32, num_workers: int = 4,
                 patches_per_image_train: int = 10,
                 multichannel=False,
                 ):
        super().__init__()
        self.train_urls = train_urls
        self.test_urls = test_urls
        self.val_urls = val_urls
        self.n_images = n_images
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patches_per_image_train = patches_per_image_train
        self.multichannel = multichannel

        self.train_transform = get_wsi_transforms('train')
        self.eval_transform = get_wsi_transforms('eval')

    def _process_sample(self, sample_tuple, is_train=True):
        _, img, mask = sample_tuple
        
        transform = self.train_transform if is_train else self.eval_transform
        augmented = transform(image=img, mask=mask)
        
        img_tensor = augmented['image']
        mask_tensor = augmented['mask'].long() 
        
        return img_tensor, mask_tensor

    
    def _process_sample_2(self, sample_tuple, is_train=True, multichannel=True):
        key, s1_np, s2_np, s3_np, mask_np = sample_tuple

        transform = self.train_transform if is_train else self.eval_transform

        augmented = transform(
            image=s1_np,
            image2=s2_np,
            image3=s3_np,
            mask=mask_np
        )

        img_multiscale_tensor = torch.cat([
            augmented['image'],
            augmented['image2'],
            augmented['image3']
        ], dim=0) if multichannel else augmented['image']

        return img_multiscale_tensor, augmented['mask'].long()

    def setup(self, stage=None):
        # Webdataset take care of it
        pass

    def train_dataloader(self):
        n_train_samples = self.n_images * self.patches_per_image_train

        dataset = (
            wds.WebDataset(
                self.train_urls, 
                nodesplitter=wds.split_by_node, 
                shardshuffle=2,
                empty_check=False
            )
            .decode(custom_decoder)
            .to_tuple("__key__", "image.png", "mask.png")
            # .map(lambda x: self._process_sample(x, is_train=True))
            # .shuffle(100)
            # .with_epoch(n_train_samples)
            .compose(lambda src: multiscale_patch_generator(
                src, 
                num_patches=self.patches_per_image_train, 
                mode='random',
                rare_class_ids=[2],
                rare_class_prob=0.6,
            ))
            .shuffle(500)
            .map(lambda x: self._process_sample_2(x, is_train=True, multichannel=self.multichannel))
            .with_epoch(n_train_samples)
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self):
        dataset = (
            wds.WebDataset(
                self.val_urls, 
                nodesplitter=wds.split_by_node, 
                shardshuffle=False,
                empty_check=False
            )
            .decode(custom_decoder)
            .to_tuple("__key__", "image.png", "mask.png")
            #.map(lambda x: self._process_sample(x, is_train=False))
            .compose(lambda src: multiscale_patch_generator(
                src, 
                num_patches=1, 
                mode='grid',
                max_grid_patchs=5,
            ))
            .map(lambda x: self._process_sample_2(x, is_train=False, multichannel=self.multichannel))
        )

        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=False,
        )
    
    def test_dataloader(self):
        dataset = (
            wds.WebDataset(
                self.test_urls, 
                nodesplitter=wds.split_by_node, 
                shardshuffle=False,
                empty_check=False
            )
            .decode(custom_decoder)
            .to_tuple("__key__", "image.png", "mask.png")
            #.map(lambda x: self._process_sample(x, is_train=False))
            .compose(lambda src: multiscale_patch_generator(
                src, 
                num_patches=1, 
                mode='grid',
                max_grid_patchs=5,
            ))
            .map(lambda x: self._process_sample_2(x, is_train=False, multichannel=self.multichannel))
        )

        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=False,
        )

class SegmentationDataModule2(pl.LightningDataModule):
    def __init__(self, train_urls: Union[str, list[str]],
                 val_urls: Union[str, list[str]],
                 test_urls: Union[str, list[str]],
                 rare_train_urls=None,
                 common_train_urls=None,
                 rare_mix_ratio=0.6,
                 batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.rare_train_urls   = rare_train_urls
        self.common_train_urls = common_train_urls
        self.rare_mix_ratio    = rare_mix_ratio

        self.train_urls = train_urls
        self.test_urls = test_urls
        self.val_urls = val_urls
        
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = get_wsi_transforms('train')
        self.eval_transform = get_wsi_transforms('eval')

    def _process_sample(self, sample_tuple, is_train=True):
        _, img, mask = sample_tuple
        
        transform = self.train_transform if is_train else self.eval_transform
        augmented = transform(image=img, mask=mask)
        
        img_tensor = augmented['image']
        mask_tensor = augmented['mask'].long() 
        
        return img_tensor, mask_tensor

    def setup(self, stage=None):
        # Webdataset take care of it
        pass

    def train_dataloader(self):
        n_train_samples = sum(1 for _ in wds.WebDataset(self.train_urls))
        rare_pipeline = (
            wds.WebDataset(
                self.rare_train_urls,
                shardshuffle=True,
                empty_check=False,
            )
            .shuffle(500)
            .decode(custom_decoder)
            .to_tuple("__key__", "image.png", "mask.png")
        )

        common_pipeline = (
            wds.WebDataset(
                self.common_train_urls,
                shardshuffle=False,
                empty_check=False,
            )
            .shuffle(500)
            .decode(custom_decoder)
            .to_tuple("__key__", "image.png", "mask.png")
        )

        mixed = wds.RandomMix(
            [rare_pipeline, common_pipeline],
            probs=[self.rare_mix_ratio, 1 - self.rare_mix_ratio],
        )

        pipeline = wds.DataPipeline(
            mixed,
            wds.map(lambda x: self._process_sample(x, is_train=True))
        ).with_epoch(n_train_samples)

        return DataLoader(
            pipeline,
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        dataset = (
            wds.WebDataset(
                self.val_urls, 
                nodesplitter=wds.split_by_node, 
                shardshuffle=False,
                empty_check=False
            )
            .decode(custom_decoder)
            .to_tuple("__key__", "image.png", "mask.png")
            .map(lambda x: self._process_sample(x, is_train=False))
        )

        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        dataset = (
            wds.WebDataset(
                self.test_urls, 
                nodesplitter=wds.split_by_node, 
                shardshuffle=False,
                empty_check=False
            )
            .decode(custom_decoder)
            .to_tuple("__key__", "image.png", "mask.png")
            .map(lambda x: self._process_sample(x, is_train=False))
        )

        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=True
        )

def get_path_shards(split: str, shard_dir='shards', path=None):
    if path is None:
        path_to_shards = str(PATH_SEG_DATA / shard_dir)
        return glob.glob(path_to_shards + f"/{split}-*.tar")
    else:
        return glob.glob(path)

if __name__ == "__main__":

    data_module = SegmentationDataModule2(
        train_urls=get_path_shards('train'),
        val_urls=get_path_shards('val'),
        test_urls=get_path_shards('test'),
        rare_train_urls=get_path_shards('train-rare'),
        common_train_urls=get_path_shards('train-common'),
        batch_size=4, 
        num_workers=0 
    )

    train_loader = data_module.train_dataloader()
    images, masks = next(iter(train_loader))

    print("=== VÉRIFICATION DES TENSEURS ===")
    print(f"Images batch shape : {images.shape} (Attendu : [Batch, 3, H, W])")
    print(f"Images dtype       : {images.dtype} (Attendu : torch.float32)")
    print(f"Valeurs Images     : Min={images.min():.2f}, Max={images.max():.2f}\n")

    print(f"Masks batch shape  : {masks.shape} (Attendu : [Batch, H, W])")
    print(f"Masks dtype        : {masks.dtype} (Attendu : torch.int64 ou torch.long)")
    print(f"Classes présentes  : {np.unique(masks)}")

"""
training/data_module.py — LightningDataModule
==============================================
Encapsule la construction des datasets et dataloaders.
Lightning appelle automatiquement setup() et les méthodes *_dataloader()
au bon moment (avant fit, avant test, etc.).

Avantages vs construction manuelle :
  • setup(stage) : chargé une seule fois, partagé entre tous les workers DDP
  • prepare_data  : téléchargement/vérification sur un seul process (rank 0)
  • Compatible avec Trainer(devices=N) sans modification
"""

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

from src.segmentation.constants import PATH_SEG_DATA

def custom_decoder(key, data):
    if key.endswith("image.png"):
        return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    elif key.endswith("mask.png"):
        return np.array(Image.open(io.BytesIO(data)).convert("P"))
    return None

class HEDStainJitter(A.ImageOnlyTransform):
    def __init__(self, strength=0.05, p=0.5):
        super().__init__(p=p)
        self.strength = strength

    def apply(self, img, **params):
        img_float = img.astype(np.float32) / 255.0
        hed        = rgb2hed(img_float)

        for i in range(3):
            hed[:, :, i] = hed[:, :, i] + np.random.uniform(-self.strength, self.strength)

        rgb = hed2rgb(hed)
        rgb = np.clip(rgb, 0, 1)
        return (rgb * 255).astype(np.uint8)

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

    if mode == "eval":
        return A.Compose(shared)

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),                 # rotations à 0/90/180/270°
        A.ElasticTransform(                      # déformation du tissu
            alpha=120, sigma=6,
            p=0.2,
        ),

        # ── Couleur / coloration ───────────────────────────────────────────────
        #HEDStainJitter(strength=0.05, p=0.7),   # variation inter-scanner (priorité haute)
        A.ColorJitter(                           # luminosité / contraste globaux
            brightness=0.15,
            contrast=0.15,
            saturation=0.1,
            p=0.5,
        ),
        # ── Artefacts scanner ──────────────────────────────────────────────────
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),   # variation de mise au point
        *shared,
    ])

def get_nb_elements(loader):
    sample_count = 0
    label_count = dict()
    for images, masks in loader:
        sample_count += len(images)
        for m in masks:
            labels, effs = np.unique(m, return_counts=True)
            for l, eff in zip(labels, effs):
                if l in label_count:
                    label_count[l] += eff
                else:
                    label_count[l] = eff

    return sample_count, label_count

def get_class_weight(loader, norm=True, square_root=False):
    _, label_count = get_nb_elements(loader)
    total_pxl = np.sum(list(label_count.values()))
    freq = {int(label): count/total_pxl for label, count in label_count.items()}
    inv_feq = np.ones(len(freq))
    if norm:
        for label in freq:
            inv_feq[label] = 1/freq[label]
        if square_root:
            weights = np.sqrt(inv_feq) / np.sqrt(inv_feq).mean()
        else:
            weights = inv_feq / inv_feq.mean()
    else:
        weights = inv_feq
    return weights

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, train_urls: Union[str, list[str]],
                 val_urls: Union[str, list[str]],
                 test_urls: Union[str, list[str]],
                 batch_size: int = 32, num_workers: int = 4):
        super().__init__()
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
        dataset = (
            wds.WebDataset(
                self.train_urls, 
                nodesplitter=wds.split_by_node, 
                shardshuffle=False,
                empty_check=False
            )
            .shuffle(1000)
            .decode(custom_decoder)
            .to_tuple("__key__", "image.png", "mask.png")
            .map(lambda x: self._process_sample(x, is_train=True))
            .with_epoch(n_train_samples)
        )

        return DataLoader(
            dataset,
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

def get_path_shards(split: str, shard_dir='/shards2'):
    path_to_shards = str(PATH_SEG_DATA / shard_dir)
    return glob.glob(path_to_shards + f"/dataset-{split}-*.tar")

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

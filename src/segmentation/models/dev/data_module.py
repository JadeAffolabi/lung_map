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
            hed[:, :, i] += np.random.uniform(-self.strength, self.strength)

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
        # HEDStainJitter(strength=0.05, p=0.7),   # variation inter-scanner (priorité haute)
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

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, train_urls: str | list[str],
                 val_urls: str | list[str],
                 test_urls: str | list[str],
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
        dataset = (
            # 1. split_by_node prévient Lightning d'essayer de séparer les données
            wds.WebDataset(self.train_urls, nodesplitter=wds.split_by_node, shardshuffle=False)
            .split_by_worker()
            .shuffle(1000)
            .decode(custom_decoder)
            .to_tuple("__key__", "image.png", "mask.png")
            .map(lambda x: self._process_sample(x, is_train=True))
            .with_epoch(10000) # Force l'epoch à s'arrêter après 10 000 batchs
        )

        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        # Pipeline de validation (Plus simple : pas de shuffle, pas d'augmentation)
        dataset = (
            wds.WebDataset(self.val_urls, nodesplitter=wds.split_by_node, shardshuffle=False)
            .split_by_worker()
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
            wds.WebDataset(self.test_urls, nodesplitter=wds.split_by_node, shardshuffle=False)
            .split_by_worker()
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

"""
dataset.py
----------
Branche votre archive WebDataset (.tar) sur la construction de graphes.

Chaque graphe porte, par patch :
  - data.x       : le masque INITIAL (sortie du U-Net), utilisé pour la
                    correction résiduelle (base_logits) ET comme partie de
                    l'entrée de l'encodeur.
  - data.context : les FEATURES du U-Net (couche juste avant sa conv finale),
                    plus riches que le masque -- c'est ce qui permet au GNN de
                    "re-regarder" l'évidence visuelle plutôt que de seulement
                    lisser des décisions déjà prises.
  - data.y       : le masque ANNOTÉ (vérité terrain), cible de la loss.

Ce template précalcule (mask, features) une fois par image via UNetFeatureExtractor
(U-Net figé), puis les patchifie exactement comme le masque annoté -- donc
correspondance patch-à-patch garantie entre les trois.

Note décodage: chaque clé est décodée EXPLICITEMENT (RGB pour l'image, niveaux
de gris pour le mask) plutôt que via `.decode("rgb8")` global -- sinon le mask
seul-canal serait lui aussi décodé en RGB 3 canaux, silencieusement.
"""

import io

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torch_geometric.data import Data

from patch_graph import patchify, build_grid_graph, unpatchify
from unet_features import UNetFeatureExtractor


def make_graph(mask_pred: np.ndarray, features: np.ndarray, mask_gt: np.ndarray,
               patch_size: int = 32, connectivity: int = 8) -> Data:
    """
    mask_pred: (H, W) ou (H, W, num_classes) -- sortie du U-Net, déjà passée au sigmoid/softmax
    features:  (H, W, C_feat) -- carte de features du U-Net (avant sa conv finale)
    mask_gt:   (H, W) -- masque annoté
    """
    H, W = mask_gt.shape[:2]  # taille AVANT padding, nécessaire pour recadrer plus tard

    pred_patches, (Hp, Wp) = patchify(mask_pred, patch_size)
    feat_patches, _ = patchify(features, patch_size)
    gt_patches, _ = patchify(mask_gt, patch_size)

    edge_index, edge_type, pos = build_grid_graph(Hp, Wp, connectivity=connectivity)

    N = Hp * Wp
    x = pred_patches.reshape(N, -1).float()        # déjà en [0,1] (sortie sigmoid du U-Net)
    context = feat_patches.reshape(N, -1).float()   # features brutes -- pas de /255 (ce ne sont pas des pixels)
    y = (gt_patches.reshape(N, -1).float() / 255.0)

    data = Data(x=x, context=context, y=y, edge_index=edge_index, edge_type=edge_type, pos=pos)
    data.grid_shape = torch.tensor([Hp, Wp])
    data.orig_shape = torch.tensor([H, W])
    return data


def precompute_unet_outputs(image: np.ndarray, extractor: UNetFeatureExtractor):
    """
    Fait tourner le U-Net (figé) sur UNE image entière et renvoie (mask_pred,
    features) au format numpy (H, W, C), prêts pour patchify.

    Attention: le U-Net tourne ici sur l'image ENTIÈRE (pas patch par patch)
    -- s'il tournait patch par patch avec un petit champ récepteur, il
    retrouverait exactement le manque de contexte que le GNN est censé
    corriger. Adaptez selon la façon dont votre U-Net a réellement été
    entraîné/utilisé.
    """
    arr = np.ascontiguousarray(image)
    tensor = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # (1,C,H,W)
    with torch.no_grad():
        mask_logits, features = extractor(tensor)
    mask_pred = torch.sigmoid(mask_logits)[0].permute(1, 2, 0).numpy()  # (H,W,num_classes)
    features_np = features[0].permute(1, 2, 0).numpy()                  # (H,W,C_feat)
    return mask_pred, features_np


def build_pipeline(url: str, extractor: UNetFeatureExtractor, patch_size: int = 32,
                    connectivity: int = 8):
    """
    url: chemin(s) vers les shards, ex: "data/shard-{000000..000099}.tar"

    Ici, image + mask annoté viennent de l'archive ; (mask_pred, features) sont
    calculés à la volée par le U-Net figé. Si vous préférez précalculer et
    stocker (mask_pred, features) dans l'archive une fois pour toutes (plus
    rapide à itérer si le U-Net est lourd), faites tourner
    `precompute_unet_outputs` en amont (script séparé) et adaptez ce pipeline
    pour lire directement les résultats précalculés au lieu d'appeler le U-Net ici.
    """
    def process(sample):
        img_bytes, mask_bytes = sample
        image = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        mask_gt = np.array(Image.open(io.BytesIO(mask_bytes)).convert("L"))
        mask_pred, features = precompute_unet_outputs(image, extractor)
        return make_graph(mask_pred, features, mask_gt, patch_size, connectivity)

    pipeline = (
        wds.WebDataset(url, shardshuffle=False)
        .to_tuple("jpg", "mask.png")  # <- adaptez aux clés réelles de vos shards ; PAS de .decode() global ici
        .map(process)
    )
    return pipeline


def reconstruct_predictions(batch, node_logits: torch.Tensor):
    """Reconstruit l'image complète (masque raffiné) pour chaque graphe d'un batch."""
    graphs = batch.to_data_list()
    images = []
    for i, g in enumerate(graphs):
        node_slice = node_logits[batch.ptr[i]:batch.ptr[i + 1]]
        Hp, Wp = g.grid_shape.tolist()
        H, W = g.orig_shape.tolist()
        images.append(unpatchify(node_slice, Hp, Wp, orig_shape=(H, W)))
    return images


# ---------------------------------------------------------------------------
# Note fine-tuning bout-en-bout: si vous passez freeze=False à
# UNetFeatureExtractor, le gradient de la loss GNN remonte aussi dans le
# U-Net. Il faut alors inclure `extractor.unet.parameters()` dans l'optimizer
# (train.py) et faire tourner `precompute_unet_outputs` DANS la boucle
# d'entraînement plutôt qu'en précalcul -- sinon `mask_pred`/`features` sont
# détachés du graphe de calcul et aucun gradient ne peut y remonter.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import tarfile
    import torch.nn as nn
    from torch_geometric.data import Batch

    class ToyUNet(nn.Module):
        """U-Net jouet, juste pour que le pipeline tourne de bout en bout dans ce test."""
        def __init__(self):
            super().__init__()
            self.enc = nn.Conv2d(3, 8, 3, padding=1)
            self.last_feature_layer = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())
            self.final_conv = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            return self.final_conv(self.last_feature_layer(self.enc(x)))

    extractor = UNetFeatureExtractor(ToyUNet(), feature_layer_name="last_feature_layer", freeze=True)

    def make_fake_shard(path, n=3, size=(70, 50)):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for i in range(n):
                img = np.random.randint(0, 255, size=(*size, 3), dtype=np.uint8)
                mask = (np.random.randint(0, 2, size=size, dtype=np.uint8) * 255)
                for key, arr in [("jpg", img), ("mask.png", mask)]:
                    b = io.BytesIO()
                    Image.fromarray(arr).save(b, format="PNG")
                    b.seek(0)
                    info = tarfile.TarInfo(name=f"{i:06d}.{key}")
                    info.size = len(b.getvalue())
                    tar.addfile(info, b)
        buf.seek(0)
        with open(path, "wb") as f:
            f.write(buf.read())

    make_fake_shard("/tmp/fake_shard2.tar")
    pipeline = build_pipeline("/tmp/fake_shard2.tar", extractor, patch_size=16)

    data_list = list(pipeline)
    for i, data in enumerate(data_list):
        print(f"graphe {i}: x={tuple(data.x.shape)} context={tuple(data.context.shape)} "
              f"y={tuple(data.y.shape)} grid_shape={data.grid_shape.tolist()}")

    batch = Batch.from_data_list(data_list)
    assert batch.x.shape[1] == 16 * 16 * 1          # masque: 1 canal
    assert batch.context.shape[1] == 16 * 16 * 16   # features: 16 canaux (ToyUNet)
    assert batch.y.shape[1] == 16 * 16 * 1          # mask_gt: 1 canal (décodage "L" explicite)
    print("OK - pipeline U-Net -> features -> graphe, avec décodage correct par clé")

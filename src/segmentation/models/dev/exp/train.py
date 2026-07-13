"""
train.py
--------
Boucle d'entraînement branchée sur le U-Net figé (via UNetFeatureExtractor).
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from dataset import build_pipeline
from model import PatchGNN
from unet_features import UNetFeatureExtractor
# from your_unet_module import UNet  # <- votre U-Net entraîné

PATCH_SIZE = 32
MASK_CHANNELS = 1       # 1 = segmentation binaire ; adapter sinon
NUM_CLASSES = 1
HIDDEN_DIM = 128
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Nom de la couche du U-Net juste avant sa conv finale -- cf. unet_features.py
FEATURE_LAYER_NAME = "last_feature_layer"  # <- À ADAPTER à votre U-Net


def collate(list_of_data):
    return Batch.from_data_list(list_of_data)


def main():
    # --- À CONFIGURER ---
    # unet = UNet(...)                       # instanciez votre U-Net
    # unet.load_state_dict(torch.load(...))  # chargez ses poids entraînés
    # extractor = UNetFeatureExtractor(unet, FEATURE_LAYER_NAME, freeze=True)
    raise NotImplementedError(
        "Instanciez votre U-Net + UNetFeatureExtractor ci-dessus avant de lancer "
        "l'entraînement (voir les lignes commentées juste au-dessus)."
    )

    pipeline = build_pipeline("data/shard-{000000..000099}.tar", extractor, patch_size=PATCH_SIZE)
    loader = DataLoader(pipeline, batch_size=BATCH_SIZE, collate_fn=collate)

    # context_channels doit correspondre au nombre de canaux de sortie de
    # FEATURE_LAYER_NAME dans votre U-Net (inspectez `features.shape[1]` en
    # sortie de UNetFeatureExtractor pour le connaître si besoin).
    CONTEXT_CHANNELS = 16  # <- À ADAPTER

    model = PatchGNN(
        mask_channels=MASK_CHANNELS,
        context_channels=CONTEXT_CHANNELS,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_layers=3,
        heads=4,
    ).to(DEVICE)

    # Si extractor.freeze=False (fine-tuning bout-en-bout), ajoutez aussi
    # extractor.unet.parameters() ici -- et faites tourner precompute_unet_outputs
    # DANS la boucle plutôt qu'en précalcul dans build_pipeline (cf. note dataset.py).
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for step, batch in enumerate(loader):
        batch = batch.to(DEVICE)

        logits = model(batch)  # [N_total_patchs_du_batch, num_classes, P, P]

        target = batch.y.view(-1, NUM_CLASSES, PATCH_SIZE, PATCH_SIZE)
        loss = F.binary_cross_entropy_with_logits(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step {step} - loss {loss.item():.4f}")


if __name__ == "__main__":
    main()

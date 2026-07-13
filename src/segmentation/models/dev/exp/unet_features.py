"""
unet_features.py
-----------------
Enveloppe votre U-Net déjà entraîné pour récupérer, en plus du masque, la
carte de features juste AVANT sa dernière conv (1x1 -> num_classes). C'est
beaucoup plus riche que le masque à 1 canal, et ça ne duplique aucun calcul :
on capture simplement une sortie intermédiaire pendant le forward normal du
U-Net, via un hook.

Adaptez `feature_layer_name` au nom exact de la couche juste avant la
dernière conv 1x1 dans VOTRE implémentation. Pour le trouver :

    print(unet)                              # inspecte la structure
    for name, _ in unet.named_modules():      # ou listez tous les noms
        print(name)

Cherchez la dernière couche (souvent un Conv2d/ReLU/BatchNorm dans le dernier
bloc du décodeur) qui précède la conv finale produisant les logits.
"""

import torch
import torch.nn as nn


class UNetFeatureExtractor(nn.Module):
    """
    forward(image) -> (mask_logits, features)
      - mask_logits: [B, num_classes, H, W]        sortie normale du U-Net
      - features:    [B, C_feat, H, W]              activations juste avant la conv finale

    freeze=True (recommandé pour démarrer) : le U-Net ne s'entraîne plus, seul
    le GNN apprend -- pipeline découplé, simple, stable. Passez freeze=False
    plus tard si vous voulez fine-tuner le U-Net conjointement avec le GNN
    (nécessite alors de faire tourner le U-Net PENDANT l'entraînement, pas en
    précalcul -- cf. note en bas de dataset.py).
    """

    def __init__(self, unet: nn.Module, feature_layer_name: str, freeze: bool = True):
        super().__init__()
        self.unet = unet
        self.freeze = freeze
        self._features = None

        named = dict(unet.named_modules())
        if feature_layer_name not in named:
            raise ValueError(
                f"'{feature_layer_name}' introuvable dans le U-Net. Couches disponibles: "
                f"{list(named.keys())}"
            )
        named[feature_layer_name].register_forward_hook(self._hook)

        if freeze:
            for p in self.unet.parameters():
                p.requires_grad_(False)
            self.unet.eval()

    def _hook(self, module, inp, out):
        self._features = out

    def forward(self, image: torch.Tensor):
        if self.freeze:
            with torch.no_grad():
                mask_logits = self.unet(image)
        else:
            mask_logits = self.unet(image)  # gradient conservé -> fine-tuning conjoint possible
        features = self._features  # capturé par le hook pendant l'appel ci-dessus
        return mask_logits, features


if __name__ == "__main__":
    # auto-test: on vérifie le mécanisme du hook sur un faux "U-Net" jouet
    # (juste pour valider la mécanique, pas un vrai U-Net).
    class ToyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Conv2d(3, 8, 3, padding=1)
            self.last_feature_layer = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())
            self.final_conv = nn.Conv2d(16, 1, 1)  # -> logits

        def forward(self, x):
            x = self.enc(x)
            x = self.last_feature_layer(x)
            return self.final_conv(x)

    unet = ToyUNet()
    extractor = UNetFeatureExtractor(unet, feature_layer_name="last_feature_layer", freeze=True)

    img = torch.rand(2, 3, 32, 32)
    mask_logits, features = extractor(img)
    print("mask_logits:", mask_logits.shape)  # [2, 1, 32, 32]
    print("features:", features.shape)         # [2, 16, 32, 32]
    assert mask_logits.shape == (2, 1, 32, 32)
    assert features.shape == (2, 16, 32, 32)
    assert not any(p.requires_grad for p in extractor.unet.parameters()), "le U-Net devrait être figé"
    print("OK - hook fonctionne, U-Net figé comme attendu")

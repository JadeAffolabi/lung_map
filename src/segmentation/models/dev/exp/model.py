"""
model.py
--------
1. PatchEncoder        : petit CNN qui transforme un patch (C,P,P) en vecteur de
                          features. Contrairement à un simple global-average-pool,
                          on garde une petite carte spatiale ("bottleneck") avant
                          de l'aplatir, pour que le décodeur puisse reconstruire un
                          masque pixel-précis (voir discussion dans le message).
2. RelPosAttentionConv  : couche de message-passing type GAT, où la position
                          relative (edge_type, 8 directions) module directement
                          les logits d'attention et le message.
3. PatchDecoder         : inverse de PatchEncoder -- vecteur -> masque (num_classes,P,P).
4. PatchGNN             : assemble le tout et renvoie DIRECTEMENT les logits du
                          masque raffiné par patch, prêts pour la loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class PatchEncoder(nn.Module):
    """CNN léger: patch (C, P, P) -> vecteur (hidden_dim,).

    On downsample x4 (deux MaxPool2d) plutôt que d'aplatir tout l'espace en un
    seul vecteur (AdaptiveAvgPool2d(1)) : ça garde une trace de la structure
    spatiale interne du patch, nécessaire pour que PatchDecoder puisse ensuite
    reconstruire un masque pixel-précis plutôt qu'une simple estimation globale.
    """

    def __init__(self, in_channels: int, hidden_dim: int, patch_size: int,
                 bottleneck_channels: int = 32):
        super().__init__()
        assert patch_size % 4 == 0, "patch_size doit être divisible par 4 dans ce template"
        self.bottleneck_size = patch_size // 4
        self.bottleneck_channels = bottleneck_channels

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, bottleneck_channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        flat_dim = bottleneck_channels * self.bottleneck_size ** 2
        self.proj = nn.Linear(flat_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, P, P]
        h = self.net(x).flatten(1)  # [N, bottleneck_channels * bottleneck_size^2]
        return self.proj(h)          # [N, hidden_dim]


class PatchDecoder(nn.Module):
    """Inverse de PatchEncoder: vecteur (hidden_dim,) -> masque (num_classes, P, P)."""

    def __init__(self, hidden_dim: int, patch_size: int, num_classes: int = 1,
                 bottleneck_channels: int = 32):
        super().__init__()
        self.bottleneck_size = patch_size // 4
        self.bottleneck_channels = bottleneck_channels

        flat_dim = bottleneck_channels * self.bottleneck_size ** 2
        self.unproj = nn.Linear(hidden_dim, flat_dim)
        self.last_conv = nn.ConvTranspose2d(16, num_classes, 4, stride=2, padding=1)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_channels, 16, 4, stride=2, padding=1), nn.ReLU(),
            self.last_conv,
        )
        # /!\ init à zéro du dernier layer: au tout début de l'entraînement, le
        # "delta" prédit est nul -> forward() ci-dessous renvoie exactement le
        # masque initial. Le réseau ne s'en écarte que si ça réduit la loss.
        nn.init.zeros_(self.last_conv.weight)
        nn.init.zeros_(self.last_conv.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        x = self.unproj(h).view(n, self.bottleneck_channels, self.bottleneck_size, self.bottleneck_size)
        return self.net(x)  # [N, num_classes, P, P]  (logits, PAS de sigmoid/softmax ici)


class RelPosAttentionConv(MessagePassing):
    """
    Couche d'attention (type GATv2) où le message et le score d'attention
    dépendent explicitement de la direction relative (edge_type) entre les
    deux patchs, en plus de leurs features.

    Convention PyG: pour edge_index[0]=j (source) -> edge_index[1]=i (destination),
    'message()' reçoit les tenseurs suffixés _j (côté source) et _i (côté destination).
    """

    def __init__(self, in_dim: int, out_dim: int, num_directions: int = 8,
                 heads: int = 4, edge_dim: int = 16):
        super().__init__(aggr="add", node_dim=0)
        self.heads, self.out_dim = heads, out_dim

        self.lin_src = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.lin_dst = nn.Linear(in_dim, heads * out_dim, bias=False)

        # La direction relative est apprise comme un embedding, puis projetée
        # dans le même espace que les messages -> elle peut à la fois biaiser
        # l'attention ET moduler le contenu du message.
        self.edge_emb = nn.Embedding(num_directions, edge_dim)
        self.lin_edge = nn.Linear(edge_dim, heads * out_dim, bias=False)

        self.att = nn.Parameter(torch.empty(1, heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.att)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor):
        src = self.lin_src(x).view(-1, self.heads, self.out_dim)
        dst = self.lin_dst(x).view(-1, self.heads, self.out_dim)
        edge_attr = self.lin_edge(self.edge_emb(edge_type)).view(-1, self.heads, self.out_dim)

        out = self.propagate(edge_index, src=src, dst=dst, edge_attr=edge_attr)  # [N, heads, out_dim]
        return out.mean(dim=1) + self.bias  # moyenne sur les têtes -> [N, out_dim]

    def message(self, src_j, dst_i, edge_attr, index, ptr, size_i):
        msg = src_j + edge_attr                                    # contenu du message conditionné par la direction
        alpha = (self.leaky_relu(dst_i + msg) * self.att).sum(-1)   # logits d'attention [E, heads]
        alpha = softmax(alpha, index, ptr, size_i)                  # softmax par noeud destination
        return msg * alpha.unsqueeze(-1)


class PatchGNN(nn.Module):
    """
    Encodeur + N couches d'attention (résiduel + LayerNorm) + décodeur.

    L'encodeur voit DEUX choses concaténées par patch: `context` (evidence
    visuelle -- image brute OU, mieux, feature map du U-Net qui a produit le
    masque initial) et `mask` (la décision du U-Net). Le "delta" appris peut
    donc soit faire confiance au masque initial (cas faciles), soit
    ré-exploiter le contexte visuel quand le masque et les voisins se
    contredisent. La correction résiduelle (voir forward) reste calculée
    UNIQUEMENT à partir du masque -- le contexte n'intervient que pour
    informer l'attention, jamais comme point de départ de `base_logits`.

    forward(data) renvoie directement les LOGITS du masque raffiné pour chaque
    patch: [N_total_patchs_du_batch, num_classes, P, P].
    """

    def __init__(self, mask_channels: int, context_channels: int, patch_size: int,
                 num_classes: int = 1, hidden_dim: int = 128, num_layers: int = 3,
                 heads: int = 4, num_directions: int = 8, use_abs_pos: bool = True):
        super().__init__()
        self.mask_channels = mask_channels
        self.context_channels = context_channels
        self.patch_size = patch_size
        self.use_abs_pos = use_abs_pos

        self.encoder = PatchEncoder(mask_channels + context_channels, hidden_dim, patch_size)
        self.decoder = PatchDecoder(hidden_dim, patch_size, num_classes)

        if use_abs_pos:
            self.pos_proj = nn.Linear(hidden_dim + 2, hidden_dim)

        self.layers = nn.ModuleList([
            RelPosAttentionConv(hidden_dim, hidden_dim, num_directions, heads)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, data) -> torch.Tensor:
        """data: objet PyG Data/Batch avec x (masque initial), context (image ou features U-Net),
        edge_index, edge_type, (pos)."""
        mask = data.x.view(-1, self.mask_channels, self.patch_size, self.patch_size)
        context = data.context.view(-1, self.context_channels, self.patch_size, self.patch_size)

        h = self.encoder(torch.cat([context, mask], dim=1))  # [N_total, hidden_dim]

        if self.use_abs_pos:
            h = self.pos_proj(torch.cat([h, data.pos], dim=-1))

        for conv, norm in zip(self.layers, self.norms):
            h = norm(h + F.relu(conv(h, data.edge_index, data.edge_type)))  # raffinage par attention

        delta = self.decoder(h)             # [N_total, num_classes, P, P] -- correction (≈0 au début de l'entraînement)
        base_logits = self._to_logit(mask)  # UNIQUEMENT dérivé du masque, jamais du contexte
        return base_logits + delta           # identité au départ, ne s'écarte que si utile

    @staticmethod
    def _to_logit(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """Convertit un masque en probabilité [0,1] vers l'espace logit (inverse du sigmoid),
        pour pouvoir lui ADDITIONNER une correction produite par le décodeur.
        Attention: suppose x déjà en [0,1] par canal (masque binaire/probabilité) -- pour un
        masque multi-classe (indices de classe), one-hot encodez x en num_classes canaux avant
        d'appeler cette fonction, ou adaptez la stratégie de combinaison."""
        x = x.clamp(eps, 1 - eps)
        return torch.log(x / (1 - x))


if __name__ == "__main__":
    # auto-test: forward + backward sur un batch factice de 2 graphes, avec
    # masque + "contexte" (features factices, comme si elles venaient du U-Net),
    # pour vérifier que le gradient remonte partout.
    from torch_geometric.data import Data, Batch
    from patch_graph import build_grid_graph

    def fake_graph(Hp, Wp, mask_C, ctx_C, P):
        edge_index, edge_type, pos = build_grid_graph(Hp, Wp)
        x = torch.rand(Hp * Wp, mask_C * P * P)             # masque initial (bruité, sortie du petit U-Net)
        context = torch.rand(Hp * Wp, ctx_C * P * P)        # features U-Net (ou image), factices ici
        y = torch.randint(0, 2, (Hp * Wp, P * P)).float()   # masque annoté (vérité terrain)
        return Data(x=x, context=context, y=y, edge_index=edge_index, edge_type=edge_type, pos=pos)

    MASK_C, CTX_C, P = 1, 32, 16  # ex: 32 canaux de features U-Net
    g1, g2 = fake_graph(5, 4, MASK_C, CTX_C, P), fake_graph(3, 6, MASK_C, CTX_C, P)
    batch = Batch.from_data_list([g1, g2])

    model = PatchGNN(mask_channels=MASK_C, context_channels=CTX_C, patch_size=P,
                      num_classes=1, hidden_dim=32, num_layers=2, heads=4)
    logits = model(batch)  # [38, 1, 16, 16]
    print("logits:", logits.shape)
    assert logits.shape == (5 * 4 + 3 * 6, 1, P, P)

    # à l'initialisation, delta=0 -> logits doivent correspondre exactement au
    # masque initial converti en espace logit (identité, aucun raffinage "gratuit")
    x_mask = batch.x.view(-1, MASK_C, P, P)
    expected_at_init = PatchGNN._to_logit(x_mask)
    assert torch.allclose(logits, expected_at_init, atol=1e-5), \
        "à l'init, le modèle devrait renvoyer exactement le masque initial (delta=0)"
    print("OK - identité vérifiée à l'initialisation (delta=0)")

    target = batch.y.view(-1, 1, P, P)
    loss = F.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    assert model.encoder.proj.weight.grad is not None, "le gradient n'atteint pas l'encodeur"
    assert model.decoder.unproj.weight.grad is not None, "le gradient n'atteint pas le décodeur"
    assert model.layers[0].att.grad is not None, "le gradient n'atteint pas la couche d'attention"
    print("loss:", loss.item())
    print("OK - forward + backward passent, gradient présent dans encodeur/attention/décodeur")

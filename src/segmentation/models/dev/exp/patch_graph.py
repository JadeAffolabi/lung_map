"""
patch_graph.py
----------------
Deux responsabilités séparées volontairement :
1. patchify()        : découper une image / un mask (H, W, C) en patchs réguliers.
2. build_grid_graph() : construire la topologie du graphe (edge_index) + l'attribut
                        de position relative (edge_type) pour une grille Hp x Wp.

Comme les patchs forment une grille régulière, on n'a jamais besoin de calculer une
adjacence spatiale coûteuse (KNN, Delaunay, etc.) : la topologie est déterministe et
se déduit uniquement de (Hp, Wp).
"""

import numpy as np
import torch

# Les 8 directions possibles, dans un ordre fixe -> index de classe pour edge_type.
# (dr, dc) = déplacement en (ligne, colonne) pour aller du noeud source vers le voisin.
DIRECTIONS = [
    (-1, 0),  # 0: haut
    (-1, 1),  # 1: haut-droite
    (0, 1),   # 2: droite
    (1, 1),   # 3: bas-droite
    (1, 0),   # 4: bas
    (1, -1),  # 5: bas-gauche
    (0, -1),  # 6: gauche
    (-1, -1), # 7: haut-gauche
]
DIR_TO_IDX = {d: i for i, d in enumerate(DIRECTIONS)}
NUM_DIRECTIONS = len(DIRECTIONS)


def patchify(array: np.ndarray, patch_size: int, pad_mode: str = "reflect"):
    """
    Découpe `array` (H, W) ou (H, W, C) en patchs réguliers non-chevauchants.

    Si H ou W n'est pas un multiple de patch_size, l'image est paddée (reflect par
    défaut) plutôt que rognée, pour ne perdre aucune information du bord.

    Retourne:
        patches: torch.Tensor de forme (Hp, Wp, C, patch_size, patch_size)
        (Hp, Wp): dimensions de la grille de patchs
    """
    if array.ndim == 2:
        array = array[..., None]  # (H, W) -> (H, W, 1)

    H, W, C = array.shape
    pad_h = (-H) % patch_size
    pad_w = (-W) % patch_size
    if pad_h or pad_w:
        array = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode=pad_mode)

    H2, W2, _ = array.shape
    Hp, Wp = H2 // patch_size, W2 // patch_size

    # reshape (H2, W2, C) -> (Hp, patch, Wp, patch, C) -> transpose -> (Hp, Wp, C, patch, patch)
    patches = array.reshape(Hp, patch_size, Wp, patch_size, C)
    patches = patches.transpose(0, 2, 4, 1, 3)

    return torch.from_numpy(np.ascontiguousarray(patches)), (Hp, Wp)


def build_grid_graph(Hp: int, Wp: int, connectivity: int = 8):
    """
    Construit le graphe régulier d'une grille Hp x Wp de patchs.

    Indexation des noeuds: noeud (r, c) -> index r * Wp + c  (ordre "row-major",
    doit correspondre à l'ordre utilisé pour aplatir les features des patchs).

    Retourne:
        edge_index: LongTensor [2, E]  (convention PyG: edge_index[0]=source, [1]=destination)
        edge_type:  LongTensor [E]     (classe de direction 0..7, direction source->dest)
        pos:        FloatTensor [N, 2] (position (row, col) normalisée dans [0, 1],
                                         utile comme feature de position *absolue*)
    """
    assert connectivity in (4, 8)
    offsets = DIRECTIONS if connectivity == 8 else [(-1, 0), (0, 1), (1, 0), (0, -1)]

    src_list, dst_list, type_list = [], [], []
    for r in range(Hp):
        for c in range(Wp):
            i = r * Wp + c
            for (dr, dc) in offsets:
                rr, cc = r + dr, c + dc
                if 0 <= rr < Hp and 0 <= cc < Wp:
                    j = rr * Wp + cc
                    src_list.append(i)
                    dst_list.append(j)
                    type_list.append(DIR_TO_IDX[(dr, dc)])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(type_list, dtype=torch.long)

    rows, cols = torch.meshgrid(torch.arange(Hp), torch.arange(Wp), indexing="ij")
    pos = torch.stack(
        [rows.flatten().float() / max(Hp - 1, 1), cols.flatten().float() / max(Wp - 1, 1)],
        dim=1,
    )
    return edge_index, edge_type, pos


def unpatchify(patches: torch.Tensor, Hp: int, Wp: int, orig_shape: tuple = None):
    """
    Inverse de patchify(): recolle des patchs en une image complète.

    patches: soit (N, C, P, P) avec N == Hp*Wp (ordre row-major r*Wp+c, l'ordre
             utilisé par build_grid_graph -- c'est le format que vous obtenez en
             sortie du modèle), soit déjà (Hp, Wp, C, P, P).
    orig_shape: (H, W) *avant* padding, pour rogner le padding ajouté par
                patchify(). Si None, l'image (paddée) complète est retournée.

    Retourne: torch.Tensor (C, H, W).
    """
    if patches.dim() == 4:
        N, C, P, _ = patches.shape
        assert N == Hp * Wp, f"{N} patchs ne correspond pas à la grille {Hp}x{Wp}"
        patches = patches.reshape(Hp, Wp, C, P, P)
    else:
        Hp2, Wp2, C, P, _ = patches.shape
        assert (Hp2, Wp2) == (Hp, Wp), "grille incohérente avec le tenseur de patchs"

    # (Hp, Wp, C, P, P) -> (C, Hp, P, Wp, P) -> (C, Hp*P, Wp*P)
    img = patches.permute(2, 0, 3, 1, 4).reshape(C, Hp * P, Wp * P)

    if orig_shape is not None:
        H, W = orig_shape
        img = img[:, :H, :W]  # on retire le padding ajouté par patchify

    return img


if __name__ == "__main__":
    # petit auto-test
    img = np.random.randint(0, 255, size=(70, 50, 3), dtype=np.uint8)
    patches, (Hp, Wp) = patchify(img, patch_size=16)
    print("patches:", patches.shape, "grille:", (Hp, Wp))

    edge_index, edge_type, pos = build_grid_graph(Hp, Wp, connectivity=8)
    print("edge_index:", edge_index.shape, "edge_type:", edge_type.shape, "pos:", pos.shape)
    assert edge_index.max() < Hp * Wp
    assert edge_type.max() < NUM_DIRECTIONS
    print("OK (patchify + build_grid_graph)")

    # round-trip patchify -> unpatchify, avec padding (70 n'est pas multiple de 16)
    N = Hp * Wp
    flat_patches = patches.reshape(N, patches.shape[2], patches.shape[3], patches.shape[4])
    recon_padded = unpatchify(flat_patches, Hp, Wp)                       # image paddée
    recon_cropped = unpatchify(flat_patches, Hp, Wp, orig_shape=(70, 50))  # recadrée

    original_chw = torch.from_numpy(img).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
    assert recon_cropped.shape == original_chw.shape
    assert torch.equal(recon_cropped, original_chw), "le round-trip ne reconstruit pas l'image d'origine"
    print("OK (round-trip patchify -> unpatchify, avec padding + recadrage)")

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


def create_patches(array: np.ndarray, patch_size: int, pad_mode: str = "reflect"):

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


def merge_patches(patches: torch.Tensor, Hp: int, Wp: int, orig_shape: tuple = None):
    """
    Inverse de create_patches(): recolle des patchs en une image complète.

    patches: soit (N, C, P, P) avec N == Hp*Wp (ordre row-major r*Wp+c, l'ordre
             utilisé par build_grid_graph -- c'est le format que vous obtenez en
             sortie du modèle), soit déjà (Hp, Wp, C, P, P).
    orig_shape: (H, W) *avant* padding, pour rogner le padding ajouté par
             create_patches(). Si None, l'image (paddée) complète est retournée.

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
        img = img[:, :H, :W]  # on retire le padding ajouté par create_patches

    return img


if __name__ == "__main__":
    # petit auto-test
    img = np.random.randint(0, 255, size=(70, 50, 3), dtype=np.uint8)
    patches, (Hp, Wp) = create_patches(img, patch_size=16)
    print("patches:", patches.shape, "grille:", (Hp, Wp))
    print("OK create_patches")

    N = Hp * Wp
    flat_patches = patches.reshape(N, patches.shape[2], patches.shape[3], patches.shape[4])
    reconst_img = merge_patches(flat_patches, Hp, Wp, orig_shape=(70, 50))  # recadrée

    original_img = torch.from_numpy(img).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
    assert reconst_img.shape == original_img.shape
    assert torch.equal(reconst_img, original_img), "La reconstruction ne fonctionne pas."
    print("OK merge_patches")

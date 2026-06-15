from data_module import SegmentationDataModule, SegmentationDataModule2, get_path_shards
import numpy as np
import torch

def get_count(loader):
    count = {}
    for imgs, masks in loader:
        for m in masks:
            t = tuple(np.unique(m).astype(int))
            set_label = tuple(map(int, t))
            if set_label in count:
                count[set_label] += 1
            else:
                count[set_label] = 1
    return count

def analyze_patch_distribution(loader, num_classes=4):
    """
    Pour chaque classe, compte :
    - % de patches qui contiennent AU MOINS un pixel de cette classe
    - distribution du % de pixels par patch (quand la classe est présente)
    """
    patches_with_class = torch.zeros(num_classes)
    total_patches = 0
    pixel_fracs = [[] for _ in range(num_classes)]

    for _, masks in loader:
        B = masks.shape[0]
        total_patches += B
        n_pixels = masks.shape[1] * masks.shape[2]

        for c in range(num_classes):
            present = (masks == c)
            for b in range(B):
                frac = present[b].float().mean().item()
                if frac > 0:
                    patches_with_class[c] += 1
                    pixel_fracs[c].append(frac)

    print(f"Total patches analysés : {total_patches}\n")
    for c in range(num_classes):
        n = int(patches_with_class[c].item())
        pct_patches = 100 * n / total_patches
        if pixel_fracs[c]:
            mean_frac = 100 * sum(pixel_fracs[c]) / len(pixel_fracs[c])
            print(f"Classe {c} : {pct_patches:.1f}% des patches la contiennent "
                  f"| quand présente : {mean_frac:.1f}% des pixels en moyenne")
        else:
            print(f"Classe {c} : absente de tous les patches !")


if __name__ == '__main__':
    datamodule = SegmentationDataModule2(
        train_urls=get_path_shards('train'),
        val_urls=get_path_shards('val'),
        test_urls=get_path_shards('test'),
        rare_train_urls=get_path_shards('train-rare'),
        common_train_urls=get_path_shards('train-common'),
        batch_size=16, 
        num_workers=7
    )

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    print("-"*20,"Train Loader", "-"*20)
    count = get_count(train_loader)
    print(f"COUNT : {count}")
    analyze_patch_distribution(train_loader)

    print("-"*20,"Val Loader", "-"*20)
    count = get_count(val_loader)
    print(f"COUNT : {count}")
    analyze_patch_distribution(val_loader)


    print("-"*20,"Test Loader", "-"*20)
    count = get_count(test_loader)
    print(f"COUNT : {count}")
    analyze_patch_distribution(test_loader)

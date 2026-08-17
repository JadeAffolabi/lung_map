from PIL import Image
import webdataset as wds
import cv2
import numpy as np
import io

from src.segmentation.constants import PATH_SEG_DATA

# --- Configuration ---
input_tar = "input_dataset.tar"   # Path to your source WebDataset
output_tar = "output_dataset.tar" # Path for the new WebDataset
img_ext = "image.png"
mask_ext = "mask.png"

def custom_decoder(key, data):
    if key.endswith("image.png"):
        return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    elif key.endswith("mask.png"):
        return np.array(Image.open(io.BytesIO(data)).convert("P"))
    return None

def apply_transform(sample):
    if img_ext not in sample or mask_ext not in sample:
        return sample

    img = sample[img_ext]
    mask = sample[mask_ext]

    reduc_factor = 4
    old_size = mask.shape
    size_10x = (
        int(old_size[1] / reduc_factor),
        int(old_size[0] / reduc_factor)
    )
    img_10x = cv2.resize(
        img,
        size_10x,
        interpolation=cv2.INTER_CUBIC
    )
    mask_10x = cv2.resize(
        mask,
        size_10x,
        interpolation=cv2.INTER_NEAREST
    )

    sample[img_ext] = img_10x
    sample[mask_ext] = mask_10x

    return sample

if __name__ == '__main__':

    all_path_shard = glob.glob(PATH_SEG_DATA + f"/shards_lchur/*.tar")
    print(f"The following shards will be changed : \n[{all_path_shard}]")

    for path in all_path_shard:
        dataset = wds.WebDataset(path).decode(custom_decoder)

        # encoder=True allows TarWriter to automatically encode the numpy arrays back into bytes (JPG/PNG)
        with wds.TarWriter(path, encoder=True) as sink:
            for sample in dataset:
                processed_sample = apply_transform(sample)
                sink.write(processed_sample)

    print("Dataset processing and saving complete!")
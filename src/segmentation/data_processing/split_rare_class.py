import webdataset as wds
import tarfile, io, os
import numpy as np
from PIL import Image
import glob

from src.segmentation.constants import PATH_SEG_DATA

def custom_decoder(key, data):
    if key.endswith("image.png"):
        return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    elif key.endswith("mask.png"):
        return np.array(Image.open(io.BytesIO(data)).convert("P"))
    return None

def split_shards_by_rare_class(
    input_urls, output_dir, rare_class=2, 
):

    rare_sink   = wds.ShardWriter(f"{output_dir}/dataset-train-rare-%06d.tar",
                                  maxsize=1e9)
    common_sink = wds.ShardWriter(f"{output_dir}/dataset-train-common-%06d.tar",
                                  maxsize=1e9)

    dataset = (
        wds.WebDataset(input_urls)
        .decode(custom_decoder)
        .to_tuple("__key__", "image.png", "mask.png")
    )

    rare_count, common_count = 0, 0
    for key, img, mask in dataset:
        sample = {"__key__": key, "image.png": img, "mask.png": mask}
        if (mask == rare_class).any():
            rare_sink.write(sample)
            rare_count += 1
        else:
            common_sink.write(sample)
            common_count += 1

    rare_sink.close()
    common_sink.close()
    print(f"Rare : {rare_count} | Common : {common_count}")

if __name__ == '__main__':
    dir_shards = str(PATH_SEG_DATA / 'shards')
    train_path = glob.glob(dir_shards + "/dataset-train-*.tar")
    split_shards_by_rare_class(
        train_path,
        dir_shards
    )

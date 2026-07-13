import webdataset as wds
from src.segmentation.constants import PATH_SEG_DATA

from src.segmentation.models.dev.data_module import get_path_shards

if __name__ == '__main__':
    shard_dir = 'shards_bcss10x'
    paths = get_path_shards('train-full', shard_dir)
    n_images = sum(1 for _ in wds.WebDataset(paths))
    print(f"There are {n_images} in : \n {paths}")
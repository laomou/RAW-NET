from .tv_datasets import build_tv_dataset
from .lmdb_datasets import build_lmdb_dataset


def build_dataset(data_set):
    data_type, data_set, image_set = data_set.split('_')
    if data_type == 'tv':
        return build_tv_dataset(data_set, image_set)
    elif data_type == 'lmdb':
        return build_lmdb_dataset(data_set, image_set)
    else:
        raise ValueError(f"Unknown dataset: {data_set}")

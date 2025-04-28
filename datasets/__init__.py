from .tv_datasets import build_tv_dataset


def build_dataset(data_set):
    tv, data_set, image_set = data_set.split('_')
    if tv == 'tv':
        return build_tv_dataset(data_set, image_set)
    else:
        raise ValueError(f"Unknown dataset: {data_set}")

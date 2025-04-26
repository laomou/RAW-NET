from .MINIST import build_minist


def build_dataset(data_set, args):
    data_set, image_set = data_set.split('_')
    if data_set == 'MINIST':
        return build_minist(image_set, args)
    raise ValueError(f"Unknown data_set: {data_set}")

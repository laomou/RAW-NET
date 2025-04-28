from .MNIST import build_mnist


def build_dataset(data_set, args):
    data_set, image_set = data_set.split('_')
    if data_set == 'MNIST':
        return build_mnist(image_set, args)
    raise ValueError(f"Unknown data_set: {data_set}")

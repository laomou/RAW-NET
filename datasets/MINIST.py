import torchvision


def build_minist(image_set, args):
    if image_set == 'train':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        return torchvision.datasets.MNIST(
            root="./data/MINIST",
            train=True,
            download=True,
            transform=transform,
        )
    elif image_set == 'val':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        return torchvision.datasets.MNIST(
            root="./data/MINIST",
            train=False,
            transform=transform,
        )
    raise ValueError(f"Unknown image_set: {image_set}")

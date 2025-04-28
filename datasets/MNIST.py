from torchvision import datasets, transforms


def build_mnist(image_set, args):
    if image_set == 'train':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return datasets.MNIST(
            root="./data/MNIST",
            train=True,
            download=True,
            transform=transform,
        )
    elif image_set == 'val':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return datasets.MNIST(
            root="./data/MNIST",
            train=False,
            transform=transform,
        )
    raise ValueError(f"Unknown image_set: {image_set}")

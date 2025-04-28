from torchvision import datasets, transforms


def build_tv_dataset(data_set, image_set):
    if data_set == "MNIST":
        dataset_class = datasets.MNIST
    else:
        raise ValueError(f"Unknown image_set: {data_set}")

    if image_set == "train":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return dataset_class(
            root="./data/",
            train=True,
            download=True,
            transform=transform,
        )
    elif image_set == "val":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return dataset_class(
            root="./data/",
            train=False,
            download=True,
            transform=transform,
        )
    raise ValueError(f"Unknown image_set: {image_set}")

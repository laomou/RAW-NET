import torchvision
import torchvision.transforms as transforms


def build_dataset(image_set, args):
    if image_set == 'train':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return torchvision.datasets.MNIST(
            root="./data/MINIST",
            train=True,
            download=True,
            transform=transform,
        )
    elif image_set == 'val':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return torchvision.datasets.MNIST(
            root="./data/MINIST",
            train=False,
            transform=transform,
        )

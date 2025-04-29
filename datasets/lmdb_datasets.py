import lmdb
import pickle4 as pickle
from torchvision import transforms
from torch.utils.data import Dataset


class LMDBDataset(Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.transform = transform
        self.target_transform = target_transform

        env = lmdb.open(self.db_path, readonly=True)
        with env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))

    def __getitem__(self, index):
        img, target = None, None
        byteflow = self.txn.get(self.keys[index])
        unpacked = pickle.loads(byteflow)

        img = unpacked[0]
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def build_lmdb_dataset(data_set, image_set):
    if data_set == "MNIST":
        db_path = "./data/mnist_lmdb"
    else:
        raise ValueError(f"Unknown image_set: {data_set}")

    if image_set == "train":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return LMDBDataset(
            db_path=db_path,
            transform=transform,
            target_transform=None
        )
    elif image_set == "val":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return LMDBDataset(
            db_path=db_path,
            transform=transform,
            target_transform=None
        )
    raise ValueError(f"Unknown image_set: {image_set}")

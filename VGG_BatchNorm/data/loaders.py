"""
Data loaders
"""
import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import torchvision.datasets as datasets

# Shared dataset location: <repo>/codes/data (the tar.gz bundled in this folder
# was truncated, so a fresh copy is downloaded there on first use).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, os.pardir, os.pardir, "data"))


class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def get_cifar_loader(root=DEFAULT_ROOT, batch_size=128, train=True, shuffle=True, num_workers=4, n_items=-1):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    data_transforms = transforms.Compose(
        [transforms.ToTensor(),
        normalize])

    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=data_transforms)
    if n_items > 0:
        dataset = Subset(dataset, list(range(min(n_items, len(dataset)))))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return loader

if __name__ == '__main__':
    train_loader = get_cifar_loader()
    for X, y in train_loader:
        print(X[0])
        print(y[0])
        print(X[0].shape)
        img = np.transpose(X[0], [1,2,0])
        plt.imshow(img*0.5 + 0.5)
        plt.savefig('sample.png')
        print(X[0].max())
        print(X[0].min())
        break
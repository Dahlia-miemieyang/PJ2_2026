"""Shared CIFAR-10 data loaders for Project 2.

Provides two transform regimes:
  - 'plain'  : ToTensor + Normalize(0.5,...)   (matches the assignment's default,
               used for the Batch-Norm task so results are comparable to the spec)
  - 'augment': random crop + horizontal flip + per-channel CIFAR normalization
               (used by Task-1 to push test accuracy as high as possible)

The CIFAR-10 archive shipped with the assignment was truncated/corrupt, so we
download a fresh copy into ``codes/data`` (shared by both tasks) on first use.
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import torchvision.datasets as datasets

# Resolve <repo>/codes/data regardless of the caller's working directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(os.path.join(_THIS_DIR, os.pardir, "data"))

# Standard CIFAR-10 channel statistics (computed over the training set).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# The simple normalization used in the assignment's sample loaders.py.
SIMPLE_MEAN = (0.5, 0.5, 0.5)
SIMPLE_STD = (0.5, 0.5, 0.5)

CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def _build_transform(train, regime):
    if regime == "augment":
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    # 'plain' regime — identical for train and test.
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(SIMPLE_MEAN, SIMPLE_STD),
    ])


def get_cifar_loader(root=DATA_ROOT, batch_size=128, train=True, shuffle=None,
                     num_workers=2, n_items=-1, regime="plain"):
    """Return a DataLoader over CIFAR-10.

    Args:
        root: directory holding (or to receive) ``cifar-10-batches-py``.
        regime: 'plain' (assignment default) or 'augment' (Task-1 training).
        n_items: if > 0, use only the first ``n_items`` samples (fast debugging).
        shuffle: defaults to ``train`` when None.
    """
    if shuffle is None:
        shuffle = train
    transform = _build_transform(train, regime)
    dataset = datasets.CIFAR10(root=root, train=train, download=True,
                               transform=transform)
    if n_items > 0:
        dataset = Subset(dataset, list(range(min(n_items, len(dataset)))))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loader


if __name__ == "__main__":
    loader = get_cifar_loader(train=True, regime="augment")
    xs, ys = next(iter(loader))
    print("batch:", xs.shape, ys.shape)
    print("pixel range:", float(xs.min()), float(xs.max()))
    print("classes:", CLASSES)

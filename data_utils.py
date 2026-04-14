"""data_utils.py — Dataset loaders for CIFAR-10 and CIFAR-10-C.
"""

import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


# CIFAR-10-C corruptions

CORRUPTIONS: List[str] = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "shot_noise",
    "zoom_blur",
]


# Data loaders

def get_cifar10_test_loader(
    data_dir: str,
    batch_size: int,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    num_workers: int = 2,
) -> DataLoader:
    """Return a DataLoader for the CIFAR-10 clean test set.

    Args:
        data_dir: Root directory; CIFAR-10 will be downloaded here if absent.
        batch_size: Images per mini-batch.
        mean: Per-channel normalisation mean (length-3 tuple).
        std: Per-channel normalisation std (length-3 tuple).
        num_workers: DataLoader background worker count.

    Returns:
        A :class:`torch.utils.data.DataLoader` over the 10 000-image test set.
    """
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def get_cifar10_train_loader(
    data_dir: str,
    batch_size: int,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    num_workers: int = 2,
    augment: bool = True,
) -> DataLoader:
    """Return a DataLoader for the CIFAR-10 training set.

    Args:
        data_dir: Root directory; CIFAR-10 will be downloaded here if absent.
        batch_size: Images per mini-batch.
        mean: Per-channel normalisation mean.
        std: Per-channel normalisation std.
        num_workers: DataLoader background worker count.
        augment: If ``True``, apply random crop and horizontal flip.

    Returns:
        A :class:`torch.utils.data.DataLoader` over the 50 000-image training
        set.
    """
    if augment:
        tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=tf)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)


def load_cifar10c(
    data_dir: str,
    corruption: str,
    severity: int,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    batch_size: int = 128,
    num_workers: int = 2,
) -> DataLoader:
    """Load one corruption × severity split from CIFAR-10-C as a DataLoader.

    Args:
        data_dir: Path to the directory containing the ``.npy`` files
            (e.g. ``'./data/CIFAR-10-C'``).
        corruption: One of the 15 corruption names in :data:`CORRUPTIONS`.
        severity: Integer in ``[1, 5]`` inclusive.
        mean: Per-channel normalisation mean.
        std: Per-channel normalisation std.
        batch_size: Images per mini-batch.
        num_workers: DataLoader background worker count.

    Returns:
        A :class:`torch.utils.data.DataLoader` over the 10 000 images with
        the requested corruption at the specified severity level.
    """
    if not 1 <= severity <= 5:
        raise ValueError(f"severity must be in [1, 5], got {severity}")

    images_path = os.path.join(data_dir, f"{corruption}.npy")
    labels_path = os.path.join(data_dir, "labels.npy")

    if not os.path.isfile(images_path):
        raise FileNotFoundError(
            f"CIFAR-10-C file not found: {images_path}\n"
            "Download from https://zenodo.org/record/2535967 and extract to "
            f"{data_dir}"
        )
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(
            f"CIFAR-10-C labels not found: {labels_path}"
        )

    # Each .npy: shape (50000, 32, 32, 3) uint8; severities stacked 1→5
    images_all = np.load(images_path)   # (50000, 32, 32, 3)
    labels_all = np.load(labels_path)   # (10000,) — same for all severities

    # Slice the correct severity block (10 000 images each)
    start = (severity - 1) * 10_000
    end   = severity * 10_000
    images = images_all[start:end]      # (10000, 32, 32, 3) uint8

    # Convert uint8 HWC → float CHW in [0,1], then normalise
    images_f = images.astype(np.float32) / 255.0           # [0,1]
    images_t = torch.from_numpy(images_f).permute(0, 3, 1, 2)  # (N,3,32,32)

    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1)
    images_t = (images_t - mean_t) / std_t

    labels_t = torch.from_numpy(labels_all.astype(np.int64))

    ds = TensorDataset(images_t, labels_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)

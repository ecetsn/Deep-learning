"""augmix.py — AugMix data augmentation for CIFAR-10.
"""

import random
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# Primitive augmentation operations

def _enhance(img: Image.Image, factor: float, enhancer_cls) -> Image.Image:
    return enhancer_cls(img).enhance(factor)


def _autocontrast(img: Image.Image, _level: float) -> Image.Image:
    return ImageOps.autocontrast(img)


def _equalize(img: Image.Image, _level: float) -> Image.Image:
    return ImageOps.equalize(img)


def _posterize(img: Image.Image, level: float) -> Image.Image:
    # Map level [0,1] → bits [4,8]
    bits = int(4 + level * 4)
    return ImageOps.posterize(img, bits)


def _rotate(img: Image.Image, level: float) -> Image.Image:
    degrees = level * 30.0
    if random.random() < 0.5:
        degrees = -degrees
    return img.rotate(degrees, fillcolor=(128, 128, 128))


def _solarize(img: Image.Image, level: float) -> Image.Image:
    threshold = int(level * 256)
    return ImageOps.solarize(img, threshold)


def _shear_x(img: Image.Image, level: float) -> Image.Image:
    shear = level * 0.3
    if random.random() < 0.5:
        shear = -shear
    return img.transform(img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0),
                         fillcolor=(128, 128, 128))


def _shear_y(img: Image.Image, level: float) -> Image.Image:
    shear = level * 0.3
    if random.random() < 0.5:
        shear = -shear
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, shear, 1, 0),
                         fillcolor=(128, 128, 128))


def _translate_x(img: Image.Image, level: float) -> Image.Image:
    pixels = int(level * img.size[0] * 0.33)
    if random.random() < 0.5:
        pixels = -pixels
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0),
                         fillcolor=(128, 128, 128))


def _translate_y(img: Image.Image, level: float) -> Image.Image:
    pixels = int(level * img.size[1] * 0.33)
    if random.random() < 0.5:
        pixels = -pixels
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels),
                         fillcolor=(128, 128, 128))


def _color(img: Image.Image, level: float) -> Image.Image:
    return _enhance(img, 1.0 + level * 1.8, ImageEnhance.Color)


def _contrast(img: Image.Image, level: float) -> Image.Image:
    return _enhance(img, 1.0 + level * 1.8, ImageEnhance.Contrast)


def _brightness(img: Image.Image, level: float) -> Image.Image:
    return _enhance(img, 1.0 + level * 1.8, ImageEnhance.Brightness)


def _sharpness(img: Image.Image, level: float) -> Image.Image:
    return _enhance(img, 1.0 + level * 1.8, ImageEnhance.Sharpness)


# Ordered list of (name, function) pairs — the 14 ops from the paper
_AUGMENTATIONS: List[Callable] = [
    _autocontrast,
    _equalize,
    _posterize,
    _rotate,
    _solarize,
    _shear_x,
    _shear_y,
    _translate_x,
    _translate_y,
    _color,
    _contrast,
    _brightness,
    _sharpness,
]


# Core AugMix routine

def augment_and_mix(
    image: Image.Image,
    severity: int = 3,
    width: int = 3,
    depth: int = -1,
) -> Image.Image:
    """Apply the AugMix augmentation to a single PIL image.

    Args:
        image: A PIL image (H × W × 3), expected to be 32 × 32 for CIFAR.
        severity: Augmentation severity level (``1``–``10``); controls the
            magnitude passed to each primitive op.
        width: Number of augmentation chains (``k`` in the paper).
        depth: Chain depth.  If ``-1``, sampled uniformly from ``{1,2,3}``.

    Returns:
        A PIL image after AugMix mixing.
    """
    ws = np.float32(np.random.dirichlet([1.0] * width))  # mixing weights
    m  = np.float32(np.random.beta(1.0, 1.0))             # blend weight

    level = severity / 10.0   # normalise severity to [0,1]
    mix = np.zeros_like(np.array(image, dtype=np.float32))

    for i in range(width):
        d = depth if depth > 0 else np.random.randint(1, 4)
        aug_img = image.copy()
        for _ in range(d):
            op = random.choice(_AUGMENTATIONS)
            aug_img = op(aug_img, level)
        mix += ws[i] * np.array(aug_img, dtype=np.float32)

    # Blend mixed augmentations with original
    orig = np.array(image, dtype=np.float32)
    result = m * orig + (1.0 - m) * mix
    return Image.fromarray(np.uint8(np.clip(result, 0, 255)))


# Dataset wrapper

class AugMixDataset(Dataset):
    """CIFAR-10 dataset wrapper that returns three views per sample.

    Each call to ``__getitem__`` returns ``(clean, augmix1, augmix2, label)``
    where ``clean`` is the standard randomly-cropped and flipped view and
    ``augmix1``/``augmix2`` are two independent AugMix views.  All three
    tensors are normalised.

    Args:
        root: Root directory for CIFAR-10 (downloaded if absent).
        train: If ``True``, load train split; else test split.
        mean: Per-channel normalisation mean.
        std: Per-channel normalisation std.
        severity: AugMix severity (passed to :func:`augment_and_mix`).
        width: AugMix mixture width.
        depth: AugMix chain depth (``-1`` for random).
    """

    def __init__(
        self,
        root: str,
        train: bool,
        mean: Tuple[float, ...],
        std: Tuple[float, ...],
        severity: int = 3,
        width: int = 3,
        depth: int = -1,
    ) -> None:
        self.dataset = datasets.CIFAR10(root, train=train, download=True)
        self.severity = severity
        self.width = width
        self.depth = depth
        self.mean = mean
        self.std = std

        # Always apply base augmentation (crop + flip) before AugMix
        if train:
            self.preprocess = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.preprocess = transforms.Lambda(lambda x: x)  # identity

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return a tuple of (clean, augmix1, augmix2, label).

        Args:
            idx: Sample index.

        Returns:
            A tuple ``(clean, aug1, aug2, label)`` where each image is a
            normalised ``(3, 32, 32)`` float tensor.
        """
        img, label = self.dataset[idx]   # PIL image, int
        img = self.preprocess(img)       # PIL → PIL (crop/flip)

        aug1 = augment_and_mix(img, self.severity, self.width, self.depth)
        aug2 = augment_and_mix(img, self.severity, self.width, self.depth)

        clean = self.to_tensor(img)
        aug1  = self.to_tensor(aug1)
        aug2  = self.to_tensor(aug2)

        return clean, aug1, aug2, label


# JSD Consistency Loss

def jsd_loss(
    logits_clean: torch.Tensor,
    logits_aug1: torch.Tensor,
    logits_aug2: torch.Tensor,
    labels: torch.Tensor,
    criterion: torch.nn.Module,
    lam: float = 12.0,
) -> torch.Tensor:
    """Compute the AugMix JSD consistency loss.

    The total loss is::

        L = CE(f(x_clean), y)
          + (lam / 3) * [KL(p_m ‖ p_clean) + KL(p_m ‖ p_aug1) + KL(p_m ‖ p_aug2)]

    where ``p_m = (p_clean + p_aug1 + p_aug2) / 3`` is the mixture
    distribution.

    Args:
        logits_clean: Model output for the unaugmented view, shape ``(N, C)``.
        logits_aug1: Model output for augmented view 1, shape ``(N, C)``.
        logits_aug2: Model output for augmented view 2, shape ``(N, C)``.
        labels: Ground-truth class indices, shape ``(N,)``.
        criterion: A callable that computes ``CE(logits_clean, labels)``
            (typically ``nn.CrossEntropyLoss()``).
        lam: Scaling factor for the JSD term (default ``12`` as in the paper's
            reference implementation with three views).

    Returns:
        Scalar loss tensor.
    """
    ce_loss = criterion(logits_clean, labels)

    p_clean = F.softmax(logits_clean, dim=1)
    p_aug1  = F.softmax(logits_aug1,  dim=1)
    p_aug2  = F.softmax(logits_aug2,  dim=1)

    # Mixture distribution
    p_mix = (p_clean + p_aug1 + p_aug2) / 3.0

    # KL(p_mix || p_i) = sum p_mix * log(p_mix / p_i)
    # Use log_softmax for numerical stability
    kl = (
        F.kl_div(torch.log(p_clean + 1e-8), p_mix, reduction="batchmean") +
        F.kl_div(torch.log(p_aug1  + 1e-8), p_mix, reduction="batchmean") +
        F.kl_div(torch.log(p_aug2  + 1e-8), p_mix, reduction="batchmean")
    ) / 3.0

    return ce_loss + lam * kl

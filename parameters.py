"""parameters.py — Typed dataclass configurations for CS515 HW3.

Each dataclass corresponds to one experiment stage.  ``main.py`` parses
argparse arguments and constructs the appropriate dataclass, which is then
passed as a single object to functions in ``train.py`` / ``test.py``.
"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch


# Seed helper

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility.

    Args:
        seed: Integer seed value applied to Python ``random``, NumPy, and
            both CPU and CUDA PyTorch generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Dataclasses

@dataclass
class TrainConfig:
    """Hyperparameters for clean (standard) training.

    Args:
        dataset: Dataset name, one of ``'cifar10'``.
        data_dir: Root directory for dataset downloads.
        num_workers: DataLoader worker processes.
        mean: Per-channel normalisation mean.
        std: Per-channel normalisation std.
        num_classes: Number of output classes.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Initial SGD / Adam learning rate.
        weight_decay: L2 regularisation strength.
        label_smoothing: Label smoothing coefficient for cross-entropy.
        scheduler_step_size: StepLR step size (epochs).
        scheduler_gamma: StepLR decay factor.
        seed: Global random seed.
        device: Compute device string (``'cuda'`` or ``'cpu'``).
        checkpoint_dir: Directory for saving model checkpoints.
        save_path: Filename (within ``checkpoint_dir``) for best model.
        log_interval: Print training stats every N batches.
        results_dir: Directory for saving figures and the report log.
    """

    dataset: str = "cifar10"
    data_dir: str = "./data"
    num_workers: int = 2
    mean: Tuple[float, ...] = (0.4914, 0.4822, 0.4465)
    std: Tuple[float, ...] = (0.2023, 0.1994, 0.2010)
    num_classes: int = 10
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    label_smoothing: float = 0.0
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    save_path: str = "teacher_clean.pth"
    log_interval: int = 100
    results_dir: str = "results"


@dataclass
class AugMixConfig(TrainConfig):
    """Hyperparameters for AugMix fine-tuning.

    Extends :class:`TrainConfig` with AugMix-specific fields.

    Args:
        jsd: Whether to use the Jensen-Shannon Divergence consistency loss
            (the paper's recommended ``no_jsd=False`` setting).
        mixture_width: Number of augmentation chains (``k`` in the paper).
        mixture_depth: Depth of each augmentation chain; ``-1`` means
            randomly sampled from ``{1, 2, 3}``.
        aug_severity: Severity level (``1``–``3``) passed to each op.
        save_path: Filename for the AugMix-trained checkpoint.
    """

    jsd: bool = True
    mixture_width: int = 3
    mixture_depth: int = -1         # -1 → random in {1,2,3}
    aug_severity: int = 3
    save_path: str = "teacher_augmix.pth"


@dataclass
class PGDConfig:
    """Hyperparameters for PGD adversarial attack.

    Args:
        norm: Attack norm, one of ``'linf'`` or ``'l2'``.
        eps: Attack budget (``4/255 ≈ 0.0157`` for L∞; ``0.25`` for L2).
        alpha: Per-iteration step size (commonly ``eps / 4``).
        n_iter: Number of PGD iterations.
        random_start: Whether to initialise with a random perturbation
            inside the ε-ball before iterating.
        device: Compute device string.
        batch_size: Batch size used when running the attack on the test set.
        results_dir: Directory for saving figures.
        data_dir: Root directory for CIFAR-10 data.
        mean: Normalisation mean (for un-normalising before attack).
        std: Normalisation std.
        ckpt: Path to the model checkpoint to attack.
    """

    norm: str = "linf"
    eps: float = 4.0 / 255.0
    alpha: float = (4.0 / 255.0) / 4.0
    n_iter: int = 20
    random_start: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 128
    results_dir: str = "results"
    data_dir: str = "./data"
    mean: Tuple[float, ...] = (0.4914, 0.4822, 0.4465)
    std: Tuple[float, ...] = (0.2023, 0.1994, 0.2010)
    ckpt: str = "checkpoints/teacher_clean.pth"
    num_workers: int = 2


@dataclass
class DistillConfig:
    """Hyperparameters for knowledge distillation.

    Args:
        teacher_ckpt: Path to the (pre-trained) teacher checkpoint.
        alpha: Weight of the KL-divergence term (``1 - alpha`` for CE).
        temperature: Softmax temperature for soft targets.
        epochs: Student training epochs.
        batch_size: Mini-batch size.
        learning_rate: Student optimiser initial LR.
        weight_decay: L2 regularisation.
        scheduler_step_size: StepLR step size.
        scheduler_gamma: StepLR decay factor.
        seed: Global random seed.
        device: Compute device.
        checkpoint_dir: Where to save student checkpoint.
        save_path: Filename for best student model.
        data_dir: Root directory for CIFAR-10 data.
        mean: Normalisation mean.
        std: Normalisation std.
        num_classes: Number of output classes.
        log_interval: Print interval (batches).
        results_dir: Directory for figures / report log.
        num_workers: DataLoader workers.
    """

    teacher_ckpt: str = "checkpoints/teacher_augmix.pth"
    alpha: float = 0.9
    temperature: float = 4.0
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    save_path: str = "student_augmix.pth"
    data_dir: str = "./data"
    mean: Tuple[float, ...] = (0.4914, 0.4822, 0.4465)
    std: Tuple[float, ...] = (0.2023, 0.1994, 0.2010)
    num_classes: int = 10
    log_interval: int = 100
    results_dir: str = "results"
    num_workers: int = 2


@dataclass
class EvalConfig:
    """Hyperparameters for evaluation tasks.

    Args:
        ckpt: Path to the model checkpoint to evaluate.
        data_dir: Root directory for CIFAR-10 / CIFAR-10-C data.
        batch_size: Evaluation batch size.
        device: Compute device.
        mean: Normalisation mean.
        std: Normalisation std.
        results_dir: Directory for saving figures and report log.
        num_workers: DataLoader workers.
        teacher_ckpt: Teacher checkpoint used in transfer-attack.
        student_ckpt: Student checkpoint used in transfer-attack.
    """

    ckpt: str = "checkpoints/teacher_clean.pth"
    data_dir: str = "./data"
    batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mean: Tuple[float, ...] = (0.4914, 0.4822, 0.4465)
    std: Tuple[float, ...] = (0.2023, 0.1994, 0.2010)
    results_dir: str = "results"
    num_workers: int = 2
    teacher_ckpt: str = "checkpoints/teacher_augmix.pth"
    student_ckpt: str = "checkpoints/student_augmix.pth"

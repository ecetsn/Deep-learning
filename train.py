"""train.py — Training routines for CS515 HW3.

Provides:
- Clean (standard) training with label smoothing and StepLR.
- AugMix training with JSD consistency loss (3-view batches).
- Hinton knowledge distillation training.
- Shared helpers: epoch loop, validation, history saving.
- ``log_section`` for appending to ``results/report_log.md``.
"""

import copy
import json
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from augmix import AugMixDataset, jsd_loss
from data_utils import get_cifar10_test_loader, get_cifar10_train_loader
from parameters import AugMixConfig, DistillConfig, TrainConfig


# Report-log generator

def log_section(title: str, body: str, results_dir: str = "results") -> None:
    """Append a titled section to the running report log.

    Args:
        title: Section heading written as a Markdown ``##`` header.
        body: Section content (may include tables, Markdown figures, etc.).
        results_dir: Directory that contains ``report_log.md``.
    """
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "report_log.md")
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n## {title}\n\n{body}\n")


# Optimiser / scheduler setup

def _build_optimizer_scheduler(
    params,
    lr: float,
    weight_decay: float,
    step_size: int,
    gamma: float,
):
    """Build an SGD optimiser with momentum and a StepLR scheduler.

    Args:
        params: Model parameters passed to the optimiser.
        lr: Initial learning rate.
        weight_decay: L2 regularisation coefficient.
        step_size: StepLR step (in epochs).
        gamma: StepLR decay factor.

    Returns:
        A tuple ``(optimizer, scheduler)``.
    """
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )
    return optimizer, scheduler


# Training and validation units

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int = 100,
) -> Tuple[float, float]:
    """Run a single standard training epoch.

    Args:
        model: Model to train.
        loader: DataLoader yielding ``(images, labels)`` pairs.
        optimizer: Gradient-descent optimiser.
        criterion: Loss function.
        device: Compute device.
        log_interval: Print progress every ``log_interval`` batches.

    Returns:
        A tuple ``(avg_loss, accuracy)`` over the full epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}]"
                  f"  loss={total_loss/n:.4f}  acc={correct/n:.4f}")

    return total_loss / n, correct / n


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate a model on a validation / test loader.

    Args:
        model: Model to evaluate (set to eval mode internally).
        loader: DataLoader yielding ``(images, labels)`` pairs.
        criterion: Loss function.
        device: Compute device.

    Returns:
        A tuple ``(avg_loss, accuracy)`` over the full loader.
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)

    return total_loss / n, correct / n


def save_history(
    history: Dict,
    path: str,
) -> None:
    """Save a training-history dict as a JSON file.

    Args:
        history: Dictionary with list-valued fields (e.g. loss, accuracy).
        path: Destination file path.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=4)


# Standard training flow

def run_training(
    model: nn.Module,
    config: TrainConfig,
    device: torch.device,
) -> Dict:
    """Train a model with standard cross-entropy loss on CIFAR-10.

    Args:
        model: Untrained (or pre-trained) neural network.
        config: :class:`~parameters.TrainConfig` with all hyperparameters.
        device: Compute device.

    Returns:
        A dict with keys ``'history'``, ``'best_acc'``, and ``'save_path'``.
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.results_dir,    exist_ok=True)

    save_path = os.path.join(config.checkpoint_dir, config.save_path)
    history_path = os.path.join(config.checkpoint_dir,
                                config.save_path.replace(".pth", "_history.json"))

    train_loader = get_cifar10_train_loader(
        config.data_dir, config.batch_size, config.mean, config.std,
        config.num_workers, augment=True
    )
    val_loader = get_cifar10_test_loader(
        config.data_dir, config.batch_size, config.mean, config.std,
        config.num_workers
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer, scheduler = _build_optimizer_scheduler(
        model.parameters(), config.learning_rate, config.weight_decay,
        config.scheduler_step_size, config.scheduler_gamma
    )

    best_acc     = float("-inf")
    best_weights = copy.deepcopy(model.state_dict())
    history      = {"train_loss": [], "train_acc": [], "val_loss": [],
                    "val_acc": [], "lr": []}

    for epoch in range(1, config.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config.log_interval
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        print(f"Epoch {epoch:3d}/{config.epochs} | "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} | lr={current_lr:.6f}")

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        save_history(history, history_path)

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)
            print(f"  ✓ Saved best model (val_acc={best_acc:.4f}) → {save_path}")

    model.load_state_dict(best_weights)
    print(f"\nTraining complete.  Best val accuracy: {best_acc:.4f}")

    log_section(
        "Train-clean Training Summary",
        f"- Epochs: {config.epochs}\n"
        f"- LR: {config.learning_rate}, WD: {config.weight_decay}\n"
        f"- Best val accuracy: **{best_acc * 100:.2f}%**\n"
        f"- Checkpoint: `{save_path}`",
        results_dir=config.results_dir,
    )

    return {"history": history, "best_acc": float(best_acc), "save_path": save_path}


# AugMix training flow

def _train_augmix_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_jsd: bool,
    log_interval: int = 100,
) -> Tuple[float, float]:
    """Run a single AugMix training epoch.

    When ``use_jsd=True`` the loader must yield 4-tuples
    ``(clean, aug1, aug2, labels)`` from :class:`~augmix.AugMixDataset`.
    When ``use_jsd=False`` the loader yields normal ``(imgs, labels)`` pairs
    and the function is equivalent to :func:`train_one_epoch`.

    Args:
        model: Model to train.
        loader: AugMix or standard DataLoader.
        optimizer: Gradient-descent optimiser.
        criterion: Cross-entropy loss (used for CE term inside JSD).
        device: Compute device.
        use_jsd: Whether to compute the full JSD consistency loss.
        log_interval: Print progress every ``log_interval`` batches.

    Returns:
        A tuple ``(avg_loss, accuracy)`` over the full epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for batch_idx, batch in enumerate(loader):
        if use_jsd:
            clean, aug1, aug2, labels = batch
            clean, aug1, aug2 = (clean.to(device), aug1.to(device),
                                  aug2.to(device))
            labels = labels.to(device)

            optimizer.zero_grad()
            logits_clean = model(clean)
            logits_aug1  = model(aug1)
            logits_aug2  = model(aug2)
            loss = jsd_loss(logits_clean, logits_aug1, logits_aug2,
                             labels, criterion)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item() * clean.size(0)
            correct    += logits_clean.argmax(1).eq(labels).sum().item()
            n          += labels.size(0)
        else:
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += labels.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}]"
                  f"  loss={total_loss/n:.4f}  acc={correct/n:.4f}")

    return total_loss / n, correct / n


def run_augmix_training(
    model: nn.Module,
    config: AugMixConfig,
    device: torch.device,
) -> Dict:
    """Fine-tune a model using AugMix with optional JSD consistency loss.

    Args:
        model: Untrained or pre-trained neural network.
        config: :class:`~parameters.AugMixConfig` with AugMix hyperparameters.
        device: Compute device.

    Returns:
        A dict with keys ``'history'``, ``'best_acc'``, and ``'save_path'``.
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.results_dir,    exist_ok=True)

    save_path    = os.path.join(config.checkpoint_dir, config.save_path)
    history_path = os.path.join(config.checkpoint_dir,
                                config.save_path.replace(".pth", "_history.json"))

    if config.jsd:
        train_dataset = AugMixDataset(
            config.data_dir, train=True,
            mean=config.mean, std=config.std,
            severity=config.aug_severity,
            width=config.mixture_width,
            depth=config.mixture_depth,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size,
            shuffle=True, num_workers=config.num_workers, pin_memory=True
        )
    else:
        train_loader = get_cifar10_train_loader(
            config.data_dir, config.batch_size, config.mean, config.std,
            config.num_workers, augment=True
        )

    val_loader = get_cifar10_test_loader(
        config.data_dir, config.batch_size, config.mean, config.std,
        config.num_workers
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer, scheduler = _build_optimizer_scheduler(
        model.parameters(), config.learning_rate, config.weight_decay,
        config.scheduler_step_size, config.scheduler_gamma
    )

    best_acc     = float("-inf")
    best_weights = copy.deepcopy(model.state_dict())
    history      = {"train_loss": [], "train_acc": [], "val_loss": [],
                    "val_acc": [], "lr": []}

    for epoch in range(1, config.epochs + 1):
        tr_loss, tr_acc = _train_augmix_epoch(
            model, train_loader, optimizer, criterion, device,
            use_jsd=config.jsd, log_interval=config.log_interval
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        print(f"Epoch {epoch:3d}/{config.epochs} | "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} | lr={current_lr:.6f}")

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        save_history(history, history_path)

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)
            print(f"  ✓ Saved best model (val_acc={best_acc:.4f}) → {save_path}")

    model.load_state_dict(best_weights)
    print(f"\nAugMix training complete.  Best val accuracy: {best_acc:.4f}")

    jsd_str = "JSD consistency loss (no_jsd=False)" if config.jsd else "CE only (no_jsd=True)"
    log_section(
        "Task 2 — AugMix Training Summary",
        f"- Loss: {jsd_str}\n"
        f"- Mixture width: {config.mixture_width}, depth: {config.mixture_depth},"
        f" severity: {config.aug_severity}\n"
        f"- Epochs: {config.epochs}, LR: {config.learning_rate}\n"
        f"- Best val accuracy: **{best_acc * 100:.2f}%**\n"
        f"- Checkpoint: `{save_path}`",
        results_dir=config.results_dir,
    )

    return {"history": history, "best_acc": float(best_acc), "save_path": save_path}


# Knowledge Distillation (Hinton KD)

class _HintonKDLoss(nn.Module):
    """Hinton knowledge distillation loss.

    L = α · KL(log_softmax(s/T) ‖ softmax(t/T)) · T² + (1−α) · CE(s, y)

    Args:
        temperature: Softmax temperature ``T``.
        alpha: Weight of the KL term (``1 - alpha`` for CE).
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.9) -> None:
        super().__init__()
        self.T     = temperature
        self.alpha = alpha
        self.kl    = nn.KLDivLoss(reduction="batchmean")
        self.ce    = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KD loss.

        Args:
            student_logits: Student model output, shape ``(N, C)``.
            teacher_logits: Teacher model output, shape ``(N, C)``.
            labels: Ground-truth integer labels, shape ``(N,)``.

        Returns:
            Scalar loss tensor.
        """
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(teacher_logits  / self.T, dim=1)
        kl_loss = self.kl(soft_student, soft_teacher) * (self.T ** 2)
        ce_loss = self.ce(student_logits, labels)
        return self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss


def run_distillation(
    student: nn.Module,
    teacher: nn.Module,
    config: DistillConfig,
    device: torch.device,
) -> Dict:
    """Knowledge-distill a student from a (frozen) teacher on CIFAR-10.

    Args:
        student: Untrained student network.
        teacher: Pre-trained, frozen teacher network.
        config: :class:`~parameters.DistillConfig` with all hyperparameters.
        device: Compute device.

    Returns:
        A dict with keys ``'history'``, ``'best_acc'``, and ``'save_path'``.
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.results_dir,    exist_ok=True)

    save_path    = os.path.join(config.checkpoint_dir, config.save_path)
    history_path = os.path.join(config.checkpoint_dir,
                                config.save_path.replace(".pth", "_history.json"))

    train_loader = get_cifar10_train_loader(
        config.data_dir, config.batch_size, config.mean, config.std,
        config.num_workers, augment=True
    )
    val_loader = get_cifar10_test_loader(
        config.data_dir, config.batch_size, config.mean, config.std,
        config.num_workers
    )

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    criterion = _HintonKDLoss(temperature=config.temperature, alpha=config.alpha)
    optimizer, scheduler = _build_optimizer_scheduler(
        student.parameters(), config.learning_rate, config.weight_decay,
        config.scheduler_step_size, config.scheduler_gamma
    )

    best_acc     = float("-inf")
    best_weights = copy.deepcopy(student.state_dict())
    history      = {"train_loss": [], "train_acc": [], "val_loss": [],
                    "val_acc": [], "lr": []}

    for epoch in range(1, config.epochs + 1):
        student.train()
        total_loss, correct, n = 0.0, 0, 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                t_logits = teacher(imgs)
            optimizer.zero_grad()
            s_logits = student(imgs)
            loss = criterion(s_logits, t_logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item() * imgs.size(0)
            correct    += s_logits.argmax(1).eq(labels).sum().item()
            n          += labels.size(0)

            if (batch_idx + 1) % config.log_interval == 0:
                print(f"  [{batch_idx+1}/{len(train_loader)}]"
                      f"  loss={total_loss/n:.4f}  acc={correct/n:.4f}")

        tr_loss, tr_acc = total_loss / n, correct / n
        val_loss, val_acc = validate(
            student, val_loader, nn.CrossEntropyLoss(), device
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        print(f"Epoch {epoch:3d}/{config.epochs} | "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} | lr={current_lr:.6f}")

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        save_history(history, history_path)

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(student.state_dict())
            torch.save(best_weights, save_path)
            print(f"  ✓ Saved best student (val_acc={best_acc:.4f}) → {save_path}")

    student.load_state_dict(best_weights)
    print(f"\nDistillation complete.  Best student val accuracy: {best_acc:.4f}")

    log_section(
        "Task 4 — Distillation Training Summary",
        f"- Teacher checkpoint: `{config.teacher_ckpt}`\n"
        f"- α={config.alpha}, T={config.temperature}\n"
        f"- Epochs: {config.epochs}, LR: {config.learning_rate}\n"
        f"- Best student val accuracy: **{best_acc * 100:.2f}%**\n"
        f"- Student checkpoint: `{save_path}`",
        results_dir=config.results_dir,
    )

    return {"history": history, "best_acc": float(best_acc), "save_path": save_path}

"""test.py — Evaluation routines for CS515 HW3.

Provides:
- ``eval_clean``: Standard clean-accuracy evaluation.
- ``eval_cifar10c``: CIFAR-10-C robustness sweep (15 corruptions × 5 severities).
- ``eval_pgd``: PGD-20 adversarial accuracy for L∞ and L2 norms.
- ``run_gradcam``: Grad-CAM heatmap generation for clean vs. adversarial images.
- ``run_tsne``: t-SNE embedding of penultimate-layer features.
- ``eval_transfer_attack``: Transferability experiment (teacher → student).
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from attacks import evaluate_adversarial, pgd_l2, pgd_linf
from data_utils import CORRUPTIONS, get_cifar10_test_loader, load_cifar10c
from gradcam import GradCAM, visualize_gradcam
from train import log_section


# CIFAR-10 class names for labelling
_CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# Clean evaluation

def eval_clean(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute top-1 accuracy on clean data.

    Args:
        model: Neural network in eval mode.
        loader: DataLoader yielding ``(images, labels)`` pairs.
        device: Compute device.

    Returns:
        Top-1 accuracy as a float in ``[0.0, 1.0]``.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)
    acc = correct / total if total > 0 else 0.0
    print(f"Clean accuracy: {acc * 100:.2f}%")
    return acc


# CIFAR-10-C evaluation

def eval_cifar10c(
    model: nn.Module,
    data_dir: str,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 2,
    results_dir: str = "results",
    ckpt_tag: str = "model",
) -> Dict[str, float]:
    """Evaluate a model on all CIFAR-10-C corruptions and severities.

    Args:
        model: Neural network in eval mode.
        data_dir: Path to the directory containing CIFAR-10-C ``.npy`` files.
        mean: Normalisation mean.
        std: Normalisation std.
        device: Compute device.
        batch_size: Evaluation batch size.
        num_workers: DataLoader worker count.
        results_dir: Directory for saving the bar-chart PNG and appending to
            the report log.
        ckpt_tag: Short identifier appended to saved figure filenames.

    Returns:
        A dict mapping ``corruption_name`` → mean accuracy (averaged over
        5 severities).  Also contains a ``'mCA'`` key for the global mean.
    """
    os.makedirs(results_dir, exist_ok=True)
    model.eval()

    results: Dict[str, List[float]] = {c: [] for c in CORRUPTIONS}

    for corruption in tqdm(CORRUPTIONS, desc="CIFAR-10-C corruptions"):
        for severity in range(1, 6):
            loader = load_cifar10c(
                data_dir=data_dir,
                corruption=corruption,
                severity=severity,
                mean=mean,
                std=std,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    preds = model(imgs).argmax(dim=1)
                    correct += preds.eq(labels).sum().item()
                    total   += labels.size(0)
            acc = correct / total if total > 0 else 0.0
            results[corruption].append(acc)

    # Per-corruption mean (over severities)
    per_corruption_mean = {c: float(np.mean(v)) for c, v in results.items()}
    mca = float(np.mean(list(per_corruption_mean.values())))
    per_corruption_mean["mCA"] = mca

    # --- Print table ---
    print(f"\n{'Corruption':<25} {'Acc (avg over sev)':>20}")
    print("-" * 47)
    for c in CORRUPTIONS:
        print(f"  {c:<23} {per_corruption_mean[c] * 100:>18.2f}%")
    print("-" * 47)
    print(f"  {'mCA':<23} {mca * 100:>18.2f}%\n")

    # --- Bar chart ---
    fig, ax = plt.subplots(figsize=(14, 5))
    accs_pct = [per_corruption_mean[c] * 100 for c in CORRUPTIONS]
    bars = ax.bar(CORRUPTIONS, accs_pct, color="steelblue", edgecolor="white")
    ax.axhline(mca * 100, color="crimson", linestyle="--",
               label=f"mCA = {mca*100:.1f}%")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Corruption Type")
    ax.set_title(f"CIFAR-10-C Per-Corruption Accuracy — {ckpt_tag}")
    ax.set_xticks(range(len(CORRUPTIONS)))
    ax.set_xticklabels(CORRUPTIONS, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    bar_path = os.path.join(results_dir, f"cifar10c_bar_{ckpt_tag}.png")
    plt.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"[CIFAR-10-C] Bar chart saved → {bar_path}")

    # --- Detailed per-severity table for log ---
    table_lines = ["| Corruption | sev1 | sev2 | sev3 | sev4 | sev5 | mean |",
                   "|---|---|---|---|---|---|---|"]
    for c in CORRUPTIONS:
        row = results[c]
        pct = [f"{v*100:.1f}%" for v in row]
        mean_str = f"{per_corruption_mean[c]*100:.1f}%"
        table_lines.append(f"| {c} | {' | '.join(pct)} | {mean_str} |")

    log_body = (
        f"I evaluated `{ckpt_tag}` on all 15 CIFAR-10-C corruptions × 5 severities.\n\n"
        + "\n".join(table_lines)
        + f"\n\n**mCA = {mca*100:.2f}%**\n\n"
        + f"![CIFAR-10-C bar chart](./{os.path.basename(bar_path)})"
    )
    log_section(f"CIFAR-10-C Results — {ckpt_tag}", log_body,
                results_dir=results_dir)

    return per_corruption_mean


def plot_cifar10c_comparison(
    results_clean: Dict[str, float],
    results_augmix: Dict[str, float],
    results_dir: str = "results",
) -> None:
    """Save a grouped bar chart comparing clean-model vs AugMix-model per corruption.

    Args:
        results_clean: Per-corruption accuracy dict from the clean teacher.
        results_augmix: Per-corruption accuracy dict from the AugMix teacher.
        results_dir: Directory for saving the PNG.
    """
    os.makedirs(results_dir, exist_ok=True)
    x = np.arange(len(CORRUPTIONS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width / 2,
           [results_clean[c] * 100  for c in CORRUPTIONS],
           width, label="Clean teacher",  color="steelblue", edgecolor="white")
    ax.bar(x + width / 2,
           [results_augmix[c] * 100 for c in CORRUPTIONS],
           width, label="AugMix teacher", color="darkorange", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(CORRUPTIONS, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Clean vs AugMix — Per-Corruption CIFAR-10-C Accuracy")
    ax.legend()
    plt.tight_layout()
    save_path = os.path.join(results_dir, "cifar10c_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[CIFAR-10-C] Comparison chart saved → {save_path}")

    log_section(
        "Task 2 — CIFAR-10-C Comparison Plot",
        f"![Clean vs AugMix per-corruption comparison](./{os.path.basename(save_path)})",
        results_dir=results_dir,
    )


# PGD adversarial evaluation

def eval_pgd(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    norm: str,
    eps: float,
    alpha: float,
    n_iter: int,
    random_start: bool = True,
    results_dir: str = "results",
    ckpt_tag: str = "model",
) -> float:
    """Evaluate adversarial accuracy under PGD-``n_iter`` for a given norm.

    Args:
        model: Neural network (in eval mode).
        loader: DataLoader over the clean test set.
        device: Compute device.
        norm: One of ``'linf'`` or ``'l2'``.
        eps: Attack budget.
        alpha: Per-step size.
        n_iter: Number of PGD iterations.
        random_start: Random initialisation within ε-ball.
        results_dir: Directory for appending to the report log.
        ckpt_tag: Short identifier for log entries.

    Returns:
        Adversarial accuracy as a float in ``[0.0, 1.0]``.

    Raises:
        ValueError: If ``norm`` is not ``'linf'`` or ``'l2'``.
    """
    if norm not in ("linf", "l2"):
        raise ValueError(f"norm must be 'linf' or 'l2', got '{norm}'")

    attack_fn = pgd_linf if norm == "linf" else pgd_l2
    norm_label = "L∞" if norm == "linf" else "L2"

    print(f"[PGD-{n_iter}] norm={norm_label}  ε={eps:.4f}  α={alpha:.4f}")
    acc, total = evaluate_adversarial(
        model, loader, attack_fn, device,
        eps=eps, alpha=alpha, n_iter=n_iter, random_start=random_start
    )

    print(f"[PGD-{n_iter}] {norm_label} adversarial accuracy ({ckpt_tag}): "
          f"{acc * 100:.2f}%  (n={total})")

    log_section(
        f"Task 3 — PGD-{n_iter} {norm_label} ({ckpt_tag})",
        f"- ε={eps:.6f}, α={alpha:.6f}, iterations={n_iter}, random_start={random_start}\n"
        f"- **Adversarial accuracy: {acc*100:.2f}%** (over {total} samples)",
        results_dir=results_dir,
    )
    return acc


# Grad-CAM

def run_gradcam(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_layer: nn.Module,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    norm: str = "linf",
    eps: float = 4.0 / 255.0,
    alpha: float = 1.0 / 255.0,
    n_iter: int = 20,
    n_samples: int = 2,
    results_dir: str = "results",
    ckpt_tag: str = "model",
) -> None:
    """Generate Grad-CAM heatmaps for clean vs. adversarial images.

    Finds up to ``n_samples`` correctly-classified clean images that are
    misclassified under PGD-20 L∞ and saves a side-by-side 2×2 PNG.

    Args:
        model: Neural network in eval mode.
        loader: DataLoader over the clean test set.
        device: Compute device.
        target_layer: The ``nn.Module`` used for Grad-CAM (last conv layer).
        mean: Normalisation mean (for de-normalising to uint8 for display).
        std: Normalisation std.
        norm: Attack norm (``'linf'`` or ``'l2'``).
        eps: Attack budget.
        alpha: Per-step size.
        n_iter: PGD iterations.
        n_samples: Number of qualifying images to process.
        results_dir: Directory for saving PNGs and appending to the log.
        ckpt_tag: Short identifier appended to filenames.
    """
    os.makedirs(results_dir, exist_ok=True)
    attack_fn = pgd_linf if norm == "linf" else pgd_l2
    cam = GradCAM(model, target_layer)

    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32, device=device).view(1, 3, 1, 1)

    found = 0
    log_entries: List[str] = []

    for imgs, labels in loader:
        if found >= n_samples:
            break
        imgs, labels = imgs.to(device), labels.to(device)

        # Find indices correctly classified clean
        model.eval()
        with torch.no_grad():
            clean_preds = model(imgs).argmax(dim=1)

        # Run PGD
        adv_imgs = attack_fn(model, imgs, labels, eps=eps, alpha=alpha, n_iter=n_iter)
        with torch.no_grad():
            adv_preds = model(adv_imgs).argmax(dim=1)

        mask = clean_preds.eq(labels) & adv_preds.ne(labels)
        indices = mask.nonzero(as_tuple=False).squeeze(1)

        for idx in indices:
            if found >= n_samples:
                break

            i = idx.item()
            x_clean = imgs[i:i+1]
            x_adv   = adv_imgs[i:i+1]
            true_cls = labels[i].item()
            clean_cls = clean_preds[i].item()
            adv_cls   = adv_preds[i].item()

            # Generate Grad-CAM heatmaps
            cam_c = cam.generate(x_clean.requires_grad_(True), class_idx=clean_cls)
            cam_a = cam.generate(x_adv.requires_grad_(True),   class_idx=adv_cls)

            # De-normalise for display
            def _denorm(t: torch.Tensor) -> np.ndarray:
                t = (t * std_t + mean_t).clamp(0, 1)
                return (t.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

            save_name = f"gradcam_{ckpt_tag}_{found}.png"
            save_path = os.path.join(results_dir, save_name)
            visualize_gradcam(
                _denorm(x_clean), _denorm(x_adv),
                cam_c, cam_a,
                fname=save_path,
                true_label=_CIFAR10_CLASSES[true_cls],
                clean_pred=_CIFAR10_CLASSES[clean_cls],
                adv_pred=_CIFAR10_CLASSES[adv_cls],
            )

            entry = (
                f"- Sample {found}: true=`{_CIFAR10_CLASSES[true_cls]}`, "
                f"clean_pred=`{_CIFAR10_CLASSES[clean_cls]}`, "
                f"adv_pred=`{_CIFAR10_CLASSES[adv_cls]}`\n"
                f"  ![Grad-CAM {found}](./{save_name})"
            )
            log_entries.append(entry)
            found += 1

    cam.remove_hooks()

    log_section(
        f"Task 3 — Grad-CAM ({ckpt_tag})",
        f"I generated Grad-CAM heatmaps for {found} images that were correctly "
        f"classified clean but misclassified under PGD-{n_iter} ({norm.upper()}, ε={eps:.4f}).\n\n"
        + "\n".join(log_entries),
        results_dir=results_dir,
    )


# t-SNE

def run_tsne(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    hook_layer: nn.Module,
    norm: str = "linf",
    eps: float = 4.0 / 255.0,
    alpha: float = 1.0 / 255.0,
    n_iter: int = 20,
    n_samples: int = 1000,
    results_dir: str = "results",
    ckpt_tag: str = "model",
) -> None:
    """Extract penultimate-layer features and plot a t-SNE embedding.

    Extracts features for up to ``n_samples`` clean test images and their
    PGD-20 adversarial counterparts, runs sklearn t-SNE, and saves a PNG.

    Args:
        model: Neural network in eval mode.
        loader: DataLoader over the clean test set.
        device: Compute device.
        hook_layer: The ``nn.Module`` whose output is used as the feature
            (typically ``model.avgpool`` or the penultimate linear layer).
        norm: Attack norm (``'linf'`` or ``'l2'``).
        eps: Attack budget.
        alpha: Per-step size.
        n_iter: PGD iterations.
        n_samples: Maximum number of images to embed.
        results_dir: Directory for saving the PNG and appending to the log.
        ckpt_tag: Short identifier appended to filenames.
    """
    os.makedirs(results_dir, exist_ok=True)
    attack_fn = pgd_linf if norm == "linf" else pgd_l2
    model.eval()

    # Register hook on the specified layer
    features_list: List[torch.Tensor] = []

    def _hook(module, input, output):
        features_list.append(output.detach().cpu().view(output.size(0), -1))

    handle = hook_layer.register_forward_hook(_hook)

    clean_feats_all, adv_feats_all, labels_all = [], [], []
    collected = 0

    for imgs, labels in loader:
        if collected >= n_samples:
            break
        n_take = min(n_samples - collected, imgs.size(0))
        imgs, labels = imgs[:n_take].to(device), labels[:n_take].to(device)

        # Clean pass
        features_list.clear()
        with torch.no_grad():
            model(imgs)
        clean_feats_all.append(features_list[0])

        # Adversarial pass
        adv = attack_fn(model, imgs, labels, eps=eps, alpha=alpha, n_iter=n_iter)
        features_list.clear()
        with torch.no_grad():
            model(adv)
        adv_feats_all.append(features_list[0])

        labels_all.append(labels.cpu())
        collected += imgs.size(0)

    handle.remove()

    clean_feats = torch.cat(clean_feats_all, dim=0).numpy()
    adv_feats   = torch.cat(adv_feats_all,   dim=0).numpy()
    all_feats   = np.concatenate([clean_feats, adv_feats], axis=0)

    print(f"[t-SNE] Fitting on {all_feats.shape[0]} samples  ({clean_feats.shape[0]} "
          f"clean + {adv_feats.shape[0]} adv)…")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    emb  = tsne.fit_transform(all_feats)

    n = clean_feats.shape[0]
    emb_clean = emb[:n]
    emb_adv   = emb[n:]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(emb_clean[:, 0], emb_clean[:, 1], s=6, alpha=0.6,
               color="steelblue",  label="Clean")
    ax.scatter(emb_adv[:, 0],   emb_adv[:, 1],   s=6, alpha=0.6,
               color="crimson",    label=f"PGD-{n_iter} ({norm.upper()})")
    ax.set_title(f"t-SNE of penultimate features — {ckpt_tag}")
    ax.legend(markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    save_path = os.path.join(results_dir, f"tsne_{ckpt_tag}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[t-SNE] Saved → {save_path}")

    log_section(
        f"Task 3 — t-SNE ({ckpt_tag})",
        f"I embedded penultimate-layer features of {n} clean and {n} adversarial "
        f"samples using sklearn t-SNE (perplexity=30, n_iter=1000).\n\n"
        f"![t-SNE embedding](./{os.path.basename(save_path)})",
        results_dir=results_dir,
    )


# Adversarial transferability

def eval_transfer_attack(
    teacher: nn.Module,
    student: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eps: float = 4.0 / 255.0,
    alpha: float = 1.0 / 255.0,
    n_iter: int = 20,
    results_dir: str = "results",
) -> Dict[str, float]:
    """Evaluate adversarial transferability from teacher to student (and white-box).

    Generates PGD-20 L∞ adversarial examples using the **teacher** as the
    surrogate model and evaluates the **student** accuracy on those examples
    (transfer attack).  Also evaluates the student under a white-box attack
    (adversarial examples generated directly on the student).

    Args:
        teacher: AugMix-trained teacher network (used as surrogate).
        student: Student network (from Task 4 distillation).
        loader: DataLoader over the clean CIFAR-10 test set.
        device: Compute device.
        eps: L∞ budget.
        alpha: Per-step size.
        n_iter: PGD iterations.
        results_dir: Directory for appending to the report log.

    Returns:
        A dict with keys ``'student_clean'``, ``'student_on_teacher_adv'``
        (transfer), and ``'student_whitebox'``.
    """
    os.makedirs(results_dir, exist_ok=True)
    teacher.eval()
    student.eval()

    # 1. Clean accuracy of student
    correct_clean, total = 0, 0
    correct_transfer, correct_wb = 0, 0

    for imgs, labels in tqdm(loader, desc="Transfer attack"):
        imgs, labels = imgs.to(device), labels.to(device)

        # Clean
        with torch.no_grad():
            correct_clean += student(imgs).argmax(1).eq(labels).sum().item()

        # Teacher-crafted adversarials → student (transfer)
        imgs_adv_teacher = pgd_linf(
            teacher, imgs, labels, eps=eps, alpha=alpha, n_iter=n_iter
        )
        with torch.no_grad():
            correct_transfer += (
                student(imgs_adv_teacher).argmax(1).eq(labels).sum().item()
            )

        # Student white-box adversarials → student
        imgs_adv_student = pgd_linf(
            student, imgs, labels, eps=eps, alpha=alpha, n_iter=n_iter
        )
        with torch.no_grad():
            correct_wb += (
                student(imgs_adv_student).argmax(1).eq(labels).sum().item()
            )

        total += labels.size(0)

    acc_clean    = correct_clean    / total
    acc_transfer = correct_transfer / total
    acc_wb       = correct_wb       / total
    transfer_gap = acc_transfer - acc_wb   # positive → transfer is weaker

    print(f"[Transfer] Student clean accuracy:         {acc_clean    * 100:.2f}%")
    print(f"[Transfer] Student on teacher-adv (xfer):  {acc_transfer * 100:.2f}%")
    print(f"[Transfer] Student white-box accuracy:      {acc_wb       * 100:.2f}%")
    print(f"[Transfer] Transfer gap (xfer - wb):        {transfer_gap * 100:+.2f}%")

    log_section(
        "Task 5 — Adversarial Transferability",
        f"I generated PGD-{n_iter} L∞ adversarial examples using the AugMix teacher "
        f"as the surrogate (ε={eps:.4f}) and evaluated the distilled student on them.\n\n"
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| Student clean accuracy | {acc_clean*100:.2f}% |\n"
        f"| Student accuracy on teacher-crafted adversarials (transfer) | {acc_transfer*100:.2f}% |\n"
        f"| Student accuracy on self-crafted adversarials (white-box) | {acc_wb*100:.2f}% |\n"
        f"| Transfer gap (transfer − white-box) | {transfer_gap*100:+.2f}% |",
        results_dir=results_dir,
    )

    return {
        "student_clean":            acc_clean,
        "student_on_teacher_adv":   acc_transfer,
        "student_whitebox":         acc_wb,
        "transfer_gap":             transfer_gap,
    }
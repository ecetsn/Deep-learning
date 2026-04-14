"""attacks.py — PGD adversarial attacks implemented from scratch.

Provides Projected Gradient Descent (PGD) for both L∞ and L2 threat models.
"""

from typing import Tuple

import torch
import torch.nn as nn


# Utilities

def _clamp_image(x: torch.Tensor) -> torch.Tensor:
    """Clamp a normalised tensor to the valid CIFAR-10 pixel range.

    Args:
        x: Normalised image tensor of any shape.

    Returns:
        Clamped tensor.
    """
    return torch.clamp(x, -3.0, 3.0)


# L-infinity PGD

def pgd_linf(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    n_iter: int,
    random_start: bool = True,
) -> torch.Tensor:
    """PGD attack under the L∞ threat model.

    Maximises the cross-entropy loss subject to the constraint
    ``‖δ‖_∞ ≤ eps``.

    Args:
        model: Target neural network (in eval mode).
        x: Clean input images, shape ``(N, C, H, W)``, already normalised.
        y: Ground-truth labels, shape ``(N,)``.
        eps: L∞ perturbation budget.
        alpha: Per-step perturbation size.
        n_iter: Number of PGD iterations.
        random_start: If ``True``, initialise the perturbation with a random
            vector drawn uniformly from the ε-ball.

    Returns:
        Adversarial examples tensor of the same shape as ``x``.
    """
    if eps <= 0 or alpha <= 0:
        raise ValueError("eps and alpha must be positive.")

    model.eval()
    criterion = nn.CrossEntropyLoss()
    x = x.clone().detach()

    if random_start:
        delta = torch.empty_like(x).uniform_(-eps, eps)
    else:
        delta = torch.zeros_like(x)

    delta = delta.requires_grad_(True)

    for _ in range(n_iter):
        logits = model(x + delta)
        loss = criterion(logits, y)
        loss.backward()

        with torch.no_grad():
            # Ascent step in sign direction
            delta_data = delta + alpha * delta.grad.sign()
            # Project onto L∞ ε-ball first, THEN clamp to valid image range
            delta_data = torch.clamp(delta_data, -eps, eps)
            perturbed   = torch.clamp(x + delta_data, -3.0, 3.0)
            delta_data  = torch.clamp(perturbed - x, -eps, eps)

        delta = delta_data.detach().requires_grad_(True)

    return (x + delta).detach()


# L2 PGD

def pgd_l2(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    n_iter: int,
    random_start: bool = True,
) -> torch.Tensor:
    """PGD attack under the L2 threat model.

    Maximises the cross-entropy loss subject to the constraint
    ``‖δ‖_2 ≤ eps``.

    Args:
        model: Target neural network (in eval mode).
        x: Clean input images, shape ``(N, C, H, W)``, already normalised.
        y: Ground-truth labels, shape ``(N,)``.
        eps: L2 perturbation budget.
        alpha: Per-step perturbation size.
        n_iter: Number of PGD iterations.
        random_start: If ``True``, initialise the perturbation with a random
            vector drawn uniformly from the L2 ε-ball.

    Returns:
        Adversarial examples tensor of the same shape as ``x``.
    """
    if eps <= 0 or alpha <= 0:
        raise ValueError("eps and alpha must be positive.")

    model.eval()
    criterion = nn.CrossEntropyLoss()
    x = x.clone().detach()
    batch_size = x.shape[0]

    if random_start:
        noise = torch.randn_like(x)
        # Normalise to unit sphere, then scale to eps-ball
        norms = noise.view(batch_size, -1).norm(dim=1, keepdim=True)
        norms = norms.view(batch_size, 1, 1, 1)
        scale = torch.rand(batch_size, 1, 1, 1, device=x.device) * eps
        delta = noise / (norms + 1e-8) * scale
    else:
        delta = torch.zeros_like(x)

    delta = delta.requires_grad_(True)

    for _ in range(n_iter):
        logits = model(x + delta)
        loss = criterion(logits, y)
        loss.backward()

        with torch.no_grad():
            grad = delta.grad

            # Normalise gradient to unit L2 norm per sample
            g_norms = grad.view(batch_size, -1).norm(dim=1, keepdim=True)
            g_norms = g_norms.view(batch_size, 1, 1, 1).clamp(min=1e-8)

            # Ascent step along normalised gradient
            delta_data = delta + alpha * grad / g_norms

            # Clamp perturbed image to valid range FIRST
            perturbed  = torch.clamp(x + delta_data, -3.0, 3.0)
            delta_data = perturbed - x

            # Project onto L2 ε-ball
            d_norms = delta_data.view(batch_size, -1).norm(dim=1, keepdim=True)
            d_norms = d_norms.view(batch_size, 1, 1, 1).clamp(min=1e-8)
            scale = torch.clamp(eps / d_norms, max=1.0)
            delta_data = delta_data * scale

        delta = delta_data.detach().requires_grad_(True)

    return (x + delta).detach()


# Batch evaluation helper

def evaluate_adversarial(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    attack_fn,
    device: torch.device,
    **attack_kwargs,
) -> Tuple[float, int]:
    """Evaluate a model's adversarial accuracy using a given attack function.

    Args:
        model: Target neural network (moved to ``device``).
        loader: DataLoader over the evaluation set.
        attack_fn: One of :func:`pgd_linf` or :func:`pgd_l2`.
        device: Compute device.
        **attack_kwargs: Additional keyword arguments forwarded to
            ``attack_fn`` (e.g. ``eps``, ``alpha``, ``n_iter``).

    Returns:
        A tuple ``(accuracy, total_samples)`` where ``accuracy`` is in
        ``[0.0, 1.0]``.
    """
    model.eval()
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack_fn(model, x, y, **attack_kwargs)
        with torch.no_grad():
            preds = model(x_adv).argmax(dim=1)
        correct += preds.eq(y).sum().item()
        total   += y.size(0)

    acc = correct / total if total > 0 else 0.0
    return acc, total

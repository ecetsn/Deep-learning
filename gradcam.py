"""gradcam.py — Grad-CAM implementation using forward/backward hooks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional


class GradCAM:
    """Grad-CAM visualisation for a specific convolutional layer.

    Uses forward and backward hooks to capture activations and gradients
    during the model's forward and backward passes.

    Args:
        model: The neural network to visualise.
        target_layer: The ``nn.Module`` whose output activations will be used
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # Hooks

    def _save_activation(self, module, input, output) -> None:
        """Forward hook: store the layer output (activations)."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        """Backward hook: store :math:`\\partial L / \\partial A`."""
        self._gradients = grad_output[0].detach()

    # Public API

    def generate(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Compute the Grad-CAM heatmap for a single input image.

        Args:
            x: Normalised input tensor, shape ``(1, C, H, W)``.
            class_idx: Target class index.  If ``None``, uses the predicted
                class (``argmax`` of the output logits).

        Returns:
            A 2-D ``np.ndarray`` of shape ``(H, W)`` with values in
            ``[0.0, 1.0]``, bilinearly upsampled to match the input spatial
            size.
        """
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(x)     # (1, num_classes)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[0, class_idx]
        score.backward()

        # Global average pool gradients: α_k = (1/(HW)) Σ_{i,j} ∂L/∂A_k^{ij}
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, K, 1, 1)

        # Weighted combination of activations
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam = F.relu(cam)       # discard negative evidence

        # Upsample to input resolution
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear",
                            align_corners=False)

        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam

    def remove_hooks(self) -> None:
        """Remove the registered hooks to free resources."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# Visualisation helper

def visualize_gradcam(
    clean_img: np.ndarray,
    adv_img: np.ndarray,
    cam_clean: np.ndarray,
    cam_adv: np.ndarray,
    fname: str,
    true_label: str = "",
    clean_pred: str = "",
    adv_pred: str = "",
) -> None:
    """Save a 2×2 Grad-CAM comparison figure.

    The top row shows the clean image and its Grad-CAM overlay; the bottom
    row shows the adversarial image and its Grad-CAM overlay.

    Args:
        clean_img: Clean image as a ``(H, W, 3)`` uint8 array.
        adv_img: Adversarial image as a ``(H, W, 3)`` uint8 array.
        cam_clean: Grad-CAM heatmap for the clean image, shape ``(H, W)``.
        cam_adv: Grad-CAM heatmap for the adversarial image, shape ``(H, W)``.
        fname: Absolute or relative path at which to save the PNG figure.
        true_label: Human-readable true class name (for title).
        clean_pred: Human-readable clean prediction name.
        adv_pred: Human-readable adversarial prediction name.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    axes[0, 0].imshow(clean_img)
    axes[0, 0].set_title(f"Clean  (pred: {clean_pred})")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(clean_img)
    axes[0, 1].imshow(cam_clean, alpha=0.5, cmap="jet")
    axes[0, 1].set_title(f"Grad-CAM clean\n(true: {true_label})")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(adv_img)
    axes[1, 0].set_title(f"Adversarial  (pred: {adv_pred})")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(adv_img)
    axes[1, 1].imshow(cam_adv, alpha=0.5, cmap="jet")
    axes[1, 1].set_title(f"Grad-CAM adversarial")
    axes[1, 1].axis("off")

    plt.suptitle("Grad-CAM: Clean vs Adversarial", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Grad-CAM] Saved → {fname}")

"""
gradcam_engine.py
─────────────────
Novelty 1: Grad-CAM Guided SAM Prompting
  - Computes real Grad-CAM from EfficientNet's last conv layer
  - Extracts top-K hotspot points from the heatmap
  - Returns those points as SAM prompts (instead of a blind bounding box)

This is the core publishable contribution:
  CNN explains itself → SAM segments based on that explanation
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F


class GradCAMEngine:
    """
    Hooks into EfficientNet-B0's last convolutional block to
    compute Grad-CAM maps without any external library dependency.
    """

    def __init__(self, model: torch.nn.Module):
        self.model      = model
        self.gradients  = None
        self.activations = None
        self._hooks     = []
        self._register_hooks()

    def _register_hooks(self):
        # EfficientNet-B0: last conv block is model.features[-1]
        target_layer = self.model.features[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(target_layer.register_forward_hook(forward_hook))
        self._hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def compute(self, img_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Returns a Grad-CAM heatmap normalized to [0, 1].
        Shape: (H, W) matching img_tensor spatial dims after upscaling.
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass — keep graph for backward
        output = self.model(img_tensor)

        # Backward on the target class score
        score = output[0, class_idx]
        score.backward()

        # Global average pool the gradients over spatial dims
        # gradients: (1, C, h, w)  →  weights: (C,)
        weights = self.gradients.mean(dim=(2, 3)).squeeze(0)

        # activations: (1, C, h, w)  →  (C, h, w)
        acts = self.activations.squeeze(0)

        # Weighted sum of activations
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        # ReLU — keep only positive contributions
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image size (224 → original H, W)
        cam = cv2.resize(cam, (img_tensor.shape[3], img_tensor.shape[2]))

        return cam.astype(np.float32)

    def cleanup(self):
        for h in self._hooks:
            h.remove()


def gradcam_to_sam_prompts(
    gradcam_map: np.ndarray,
    original_h: int,
    original_w: int,
    top_k: int = 5,
    threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a Grad-CAM heatmap into SAM point prompts.

    Strategy:
      1. Resize cam to original image resolution
      2. Threshold at `threshold` to find high-attention regions
      3. Pick top_k highest-value points as foreground prompts
      4. Pick corners (low-attention) as background prompts

    Returns:
        point_coords: np.ndarray shape (N, 2) in (x, y) format
        point_labels: np.ndarray shape (N,)  — 1=foreground, 0=background
    """

    # Resize to original image resolution
    cam_full = cv2.resize(gradcam_map, (original_w, original_h))

    # ── Foreground points: top-K hotspots ─────────────────────────────────
    fg_coords = []
    cam_copy  = cam_full.copy()

    for _ in range(top_k):
        idx     = np.unravel_index(np.argmax(cam_copy), cam_copy.shape)
        val     = cam_copy[idx]
        if val < threshold:
            break
        yx = idx
        fg_coords.append([yx[1], yx[0]])   # (x, y)
        # Suppress local neighborhood so next argmax is a different region
        r = max(original_h, original_w) // 20
        y0 = max(0, yx[0] - r); y1 = min(original_h, yx[0] + r)
        x0 = max(0, yx[1] - r); x1 = min(original_w, yx[1] + r)
        cam_copy[y0:y1, x0:x1] = 0

    if not fg_coords:
        # Fallback: use image center
        fg_coords = [[original_w // 2, original_h // 2]]

    # ── Background points: image corners (always low-attention in MRI) ────
    margin  = min(original_h, original_w) // 10
    bg_coords = [
        [margin, margin],
        [original_w - margin, margin],
        [margin, original_h - margin],
        [original_w - margin, original_h - margin],
    ]

    all_coords = fg_coords + bg_coords
    all_labels = [1] * len(fg_coords) + [0] * len(bg_coords)

    return (
        np.array(all_coords, dtype=np.float32),
        np.array(all_labels, dtype=np.int32)
    )


def draw_gradcam_overlay(img_bgr: np.ndarray, gradcam_map: np.ndarray) -> np.ndarray:
    """
    Blends a Grad-CAM heatmap over the original BGR image.
    Returns a BGR image for encoding.
    """
    h, w    = img_bgr.shape[:2]
    cam_r   = cv2.resize(gradcam_map, (w, h))
    heatmap = cv2.applyColorMap(
        (cam_r * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    return cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
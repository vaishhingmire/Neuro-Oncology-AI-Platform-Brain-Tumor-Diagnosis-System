"""
segmentation_refiner.py
───────────────────────
Research-grade tumor mask refinement.

Key fix: Raised GradCAM suppression threshold from 0.15 → 0.35
         so only high-activation regions are kept as tumor mask.
         Added max_tumor_ratio cap (30%) to prevent whole-brain masking.
"""

import numpy as np
import cv2


def get_gradcam_bbox(gradcam: np.ndarray, threshold: float = 0.4):
    """
    Extracts a tight bounding box from the GradCAM heatmap.
    Used to constrain SAM segmentation to the high-activation region.

    Returns (x1, y1, x2, y2) or None if no strong activation found.
    """
    gradcam_norm = gradcam.copy().astype(np.float32)
    if gradcam_norm.max() > 1.0:
        gradcam_norm /= 255.0

    if gradcam_norm.max() < 0.01:
        return None

    hot_mask = (gradcam_norm >= threshold).astype(np.uint8)

    # Dilate slightly to include tumor edges
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    hot_mask = cv2.dilate(hot_mask, kernel, iterations=2)

    coords = cv2.findNonZero(hot_mask)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    img_h, img_w = gradcam.shape[:2]

    # Add 10% padding
    pad_x = int(w * 0.10)
    pad_y = int(h * 0.10)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    return [float(x1), float(y1), float(x2), float(y2)]


def refine_segmentation(mask: np.ndarray, gradcam: np.ndarray):
    """
    Research-grade tumor mask refinement.

    Steps:
    1. Resolution alignment (SAM mask → GradCAM size)
    2. GradCAM-guided suppression (threshold=0.35, only if real GradCAM)
    3. Morphological cleanup
    4. Largest tumor component selection
    5. Max ratio cap (30%) — prevents whole-brain masking
    6. Tumor validity filtering (min 0.5%)

    Returns
    -------
    refined_mask  : np.ndarray  (H, W) uint8
    is_valid_tumor: bool
    tumor_ratio   : float
    """

    # ── Step 1: Align resolution ──────────────────────────────────────────
    refined_mask = mask.copy()

    if refined_mask.shape != gradcam.shape:
        refined_mask = cv2.resize(
            refined_mask.astype(np.uint8),
            (gradcam.shape[1], gradcam.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    refined_mask = (refined_mask > 0).astype(np.uint8)

    # ── Step 2: GradCAM suppression ───────────────────────────────────────
    # Only apply if we have real GradCAM (not zeros).
    # FIX: Raised threshold 0.15 → 0.35 so only tumor-focused regions kept.
    # Old threshold (0.15) was too permissive — kept most of the brain.
    gradcam_arr = np.array(gradcam, dtype=np.float32)

    if gradcam_arr.max() > 1.0:
        gradcam_arr = gradcam_arr / 255.0

    has_real_gradcam = gradcam_arr.max() > 0.01

    if has_real_gradcam:
        if gradcam_arr.shape != refined_mask.shape:
            gradcam_arr = cv2.resize(
                gradcam_arr,
                (refined_mask.shape[1], refined_mask.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        # Keep only pixels where GradCAM activation ≥ 0.35
        # This removes low-activation background while preserving tumor core
        SUPPRESSION_THRESHOLD = 0.35
        refined_mask = np.where(
            gradcam_arr < SUPPRESSION_THRESHOLD, 0, refined_mask
        ).astype(np.uint8)
    else:
        print("[Refiner] No real GradCAM — skipping suppression, keeping SAM mask")

    # ── Step 3: Morphological cleanup ────────────────────────────────────
    kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN,  kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)

    # ── Step 4: Keep largest connected component ──────────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        refined_mask, connectivity=8
    )

    if num_labels > 1:
        largest_idx  = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cleaned_mask = np.zeros_like(refined_mask)
        cleaned_mask[labels == largest_idx] = 1
        refined_mask = cleaned_mask

    # ── Step 5: Max ratio cap — prevents whole-brain masking ─────────────
    # If SAM masks more than 30% of image it's likely over-segmenting.
    # Shrink to the GradCAM hotspot bounding box in that case.
    MAX_TUMOR_RATIO = 0.30
    raw_ratio = float(np.sum(refined_mask)) / (refined_mask.size + 1e-8)

    if raw_ratio > MAX_TUMOR_RATIO and has_real_gradcam:
        print(f"[Refiner] Mask too large ({raw_ratio:.2%}) — "
              f"cropping to GradCAM hotspot bbox")
        cam_bbox = get_gradcam_bbox(gradcam_arr, threshold=0.5)
        if cam_bbox is not None:
            x1, y1, x2, y2 = map(int, cam_bbox)
            h, w = refined_mask.shape
            x1 = max(0, min(x1, w)); x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h)); y2 = max(0, min(y2, h))
            cropped = np.zeros_like(refined_mask)
            cropped[y1:y2, x1:x2] = refined_mask[y1:y2, x1:x2]
            refined_mask = cropped
            print(f"[Refiner] Cropped to bbox [{x1},{y1},{x2},{y2}]")

    # ── Step 6: Validity check ────────────────────────────────────────────
    tumor_pixels = int(np.sum(refined_mask))
    total_pixels = refined_mask.size
    tumor_ratio  = tumor_pixels / (total_pixels + 1e-8)

    MIN_TUMOR_RATIO = 0.005
    is_valid_tumor  = tumor_ratio > MIN_TUMOR_RATIO

    if not is_valid_tumor:
        refined_mask[:] = 0

    print(f"[Refiner] tumor_pixels={tumor_pixels}, "
          f"tumor_ratio={tumor_ratio:.4f}, "
          f"valid={is_valid_tumor}, "
          f"gradcam_used={has_real_gradcam}")

    return refined_mask.astype(np.uint8), bool(is_valid_tumor), float(tumor_ratio)
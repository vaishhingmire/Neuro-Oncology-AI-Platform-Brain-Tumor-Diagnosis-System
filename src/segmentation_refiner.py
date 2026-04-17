import numpy as np
import cv2


def refine_tumor_mask(mask: np.ndarray, gradcam: np.ndarray, thresh_weight: float = 0.3) -> tuple:
    """
    Improves SAM mask using:
    - Grad-CAM weighted filtering
    - Morphological refinement
    - Connected-component filtering
    - Largest lesion preservation
    """

    # 1. Grad-CAM weighting: Suppress mask where gradcam is very low
    weighted_mask = (mask > 0).astype(np.float32)

    # Ensure weighted_mask matches gradcam resolution before applying
    if weighted_mask.shape != gradcam.shape:
        weighted_mask = cv2.resize(
            weighted_mask,
            (gradcam.shape[1], gradcam.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    weighted_mask = np.where(gradcam < 0.1, 0, weighted_mask)

    # 2. Morphological refinement (Closing to fill holes, then opening to remove noise)
    binary_mask = (weighted_mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # 3. Connected-component filtering (Largest lesion preservation)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)

    refined_mask = np.zeros_like(opened)
    dice_stability = 0.0

    if num_labels > 1:
        # Find the largest component (excluding background at index 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        refined_mask[labels == largest_label] = 1

        # Calculate Dice stability metric against original mask
        orig_binary = (mask > 0).astype(np.uint8)

        # Resize orig_binary to match refined_mask if needed
        if orig_binary.shape != refined_mask.shape:
            orig_binary = cv2.resize(
                orig_binary,
                (refined_mask.shape[1], refined_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        intersection = np.logical_and(orig_binary, refined_mask).sum()
        union = orig_binary.sum() + refined_mask.sum()
        dice_stability = (2.0 * intersection) / (union + 1e-6)

    return refined_mask.astype(np.uint8), float(dice_stability)


def refine_segmentation(mask: np.ndarray, gradcam: np.ndarray) -> tuple:
    """
    Wrapper called by pipeline.py.
    Returns: (refined_mask, is_valid_tumor, tumor_ratio)
    """

    # Normalize gradcam to [0, 1] if needed
    gradcam = np.array(gradcam, dtype=np.float32)
    if gradcam.max() > 1.0:
        gradcam = gradcam / 255.0

    # Ensure gradcam is 2D
    if gradcam.ndim == 3:
        gradcam = gradcam.mean(axis=2)

    refined_mask, dice_stability = refine_tumor_mask(mask, gradcam)

    total_pixels = refined_mask.size
    tumor_pixels = refined_mask.sum()
    tumor_ratio = float(tumor_pixels) / float(total_pixels + 1e-6)

    is_valid_tumor = (tumor_ratio > 0.01) and (dice_stability > 0.1)

    return refined_mask.astype(np.uint8), bool(is_valid_tumor), float(tumor_ratio)
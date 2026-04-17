import numpy as np

def refine_bbox_from_mask(mask: np.ndarray, original_bbox: list, margin: int = 5) -> list:
    """
    Bounding Box Auto-Correction: Shrink YOLO bbox using segmentation mask.
    Returns: [x_min, y_min, x_max, y_max]
    """
    bin_mask = (mask > 0).astype(np.uint8)
    y_indices, x_indices = np.where(bin_mask > 0)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return original_bbox

    x_min_m = np.min(x_indices)
    x_max_m = np.max(x_indices)
    y_min_m = np.min(y_indices)
    y_max_m = np.max(y_indices)
    
    # Add margin
    x_min_m = max(0, x_min_m - margin)
    y_min_m = max(0, y_min_m - margin)
    x_max_m = min(mask.shape[1], x_max_m + margin)
    y_max_m = min(mask.shape[0], y_max_m + margin)
    
    return [int(x_min_m), int(y_min_m), int(x_max_m), int(y_max_m)]

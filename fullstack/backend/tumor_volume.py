"""
tumor_volume.py
───────────────
Novelty 3: Tumor Volume & Severity Estimation from SAM Mask

Converts the 2D SAM segmentation mask into clinically meaningful metrics:
  - Tumor area in mm² (using standard MRI pixel spacing)
  - Estimated volume in mm³ (area × slice thickness)
  - WHO grade suggestion based on size thresholds
  - Geometric features: shape regularity, boundary roughness

Standard MRI assumptions (can be overridden if DICOM metadata available):
  - Pixel spacing: 1.0 mm × 1.0 mm (typical 1.5T brain MRI)
  - Slice thickness: 5.0 mm (standard axial T1)
"""

import numpy as np
import cv2


# ── Standard MRI physical parameters ──────────────────────────────────────────
DEFAULT_PIXEL_SPACING_MM = 1.0    # mm per pixel (isotropic)
DEFAULT_SLICE_THICKNESS_MM = 5.0  # mm (standard axial brain MRI slice)


def estimate_tumor_metrics(
    mask: np.ndarray,
    pixel_spacing_mm: float = DEFAULT_PIXEL_SPACING_MM,
    slice_thickness_mm: float = DEFAULT_SLICE_THICKNESS_MM,
    tumor_class: str = "Unknown"
) -> dict:
    """
    Computes clinical metrics from a 2D binary segmentation mask.

    Args:
        mask:              2D uint8 numpy array (0 or 255)
        pixel_spacing_mm:  Physical size of one pixel in mm
        slice_thickness_mm: MRI slice thickness in mm
        tumor_class:       CNN predicted class for grading context

    Returns dict with all volume/shape metrics.
    """

    mask_bin = (mask > 0).astype(np.uint8)
    total_pixels = mask_bin.size
    tumor_pixels = int(mask_bin.sum())

    if tumor_pixels == 0:
        return _empty_metrics()

    # ── Area & Volume ──────────────────────────────────────────────────────
    pixel_area_mm2   = pixel_spacing_mm ** 2
    tumor_area_mm2   = tumor_pixels * pixel_area_mm2
    tumor_volume_mm3 = tumor_area_mm2 * slice_thickness_mm
    tumor_volume_cm3 = tumor_volume_mm3 / 1000.0

    # ── Bounding box & dimensions ──────────────────────────────────────────
    coords = np.argwhere(mask_bin)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    bbox_h_mm = (y_max - y_min + 1) * pixel_spacing_mm
    bbox_w_mm = (x_max - x_min + 1) * pixel_spacing_mm
    max_diameter_mm = max(bbox_h_mm, bbox_w_mm)

    # ── Shape regularity (circularity) ────────────────────────────────────
    # Perfect circle = 1.0, irregular/spiky = lower
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    boundary_roughness = 1.0

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        perim   = cv2.arcLength(largest, True)
        if perim > 0:
            circularity = (4 * np.pi * area) / (perim ** 2)
            # Roughness: ratio of convex hull perimeter to actual perimeter
            hull  = cv2.convexHull(largest)
            hull_perim = cv2.arcLength(hull, True)
            boundary_roughness = float(perim / hull_perim) if hull_perim > 0 else 1.0

    # ── WHO grade suggestion ───────────────────────────────────────────────
    who_grade, grade_rationale = _suggest_who_grade(
        max_diameter_mm, circularity, boundary_roughness, tumor_class
    )

    # ── Severity based on volume ───────────────────────────────────────────
    if tumor_volume_cm3 < 5:
        size_severity = "Small (<5 cm³)"
    elif tumor_volume_cm3 < 20:
        size_severity = "Moderate (5–20 cm³)"
    elif tumor_volume_cm3 < 60:
        size_severity = "Large (20–60 cm³)"
    else:
        size_severity = "Very Large (>60 cm³)"

    return {
        # Core metrics
        "tumor_pixels":       tumor_pixels,
        "tumor_ratio":        float(tumor_pixels / total_pixels),
        "tumor_area_mm2":     round(tumor_area_mm2, 2),
        "tumor_volume_mm3":   round(tumor_volume_mm3, 2),
        "tumor_volume_cm3":   round(tumor_volume_cm3, 3),

        # Dimensions
        "max_diameter_mm":    round(max_diameter_mm, 2),
        "bbox_height_mm":     round(bbox_h_mm, 2),
        "bbox_width_mm":      round(bbox_w_mm, 2),

        # Shape features
        "circularity":        round(float(circularity), 3),
        "boundary_roughness": round(float(boundary_roughness), 3),

        # Clinical interpretation
        "size_severity":      size_severity,
        "who_grade_suggestion": who_grade,
        "grade_rationale":    grade_rationale,

        # MRI parameters used
        "pixel_spacing_mm":   pixel_spacing_mm,
        "slice_thickness_mm": slice_thickness_mm,
    }


def _suggest_who_grade(
    diameter_mm: float,
    circularity: float,
    roughness: float,
    tumor_class: str
) -> tuple[str, str]:
    """
    Suggests a WHO grade based on morphological features.
    NOTE: This is a heuristic suggestion — not a clinical diagnosis.
    Final grading always requires histopathology.
    """

    rationale_parts = []

    # Size scoring
    if diameter_mm < 20:
        size_score = 1
        rationale_parts.append(f"small diameter ({diameter_mm:.1f}mm)")
    elif diameter_mm < 40:
        size_score = 2
        rationale_parts.append(f"moderate diameter ({diameter_mm:.1f}mm)")
    else:
        size_score = 3
        rationale_parts.append(f"large diameter ({diameter_mm:.1f}mm)")

    # Shape irregularity scoring (irregular = higher grade)
    if circularity > 0.75:
        shape_score = 1
        rationale_parts.append("regular/circular boundary")
    elif circularity > 0.50:
        shape_score = 2
        rationale_parts.append("moderately irregular boundary")
    else:
        shape_score = 3
        rationale_parts.append("highly irregular boundary")

    # Boundary roughness
    if roughness > 1.3:
        shape_score += 1
        rationale_parts.append("rough/spiculated edges")

    total = size_score + shape_score

    # Class-specific adjustments
    if tumor_class == "Meningioma":
        # Meningiomas are usually WHO I–II
        total = min(total, 4)
    elif tumor_class == "Glioma":
        # Gliomas can be higher grade
        pass

    if total <= 3:
        grade = "WHO Grade I (suggested)"
    elif total <= 5:
        grade = "WHO Grade II (suggested)"
    elif total <= 7:
        grade = "WHO Grade III (suggested)"
    else:
        grade = "WHO Grade IV (suggested)"

    rationale = "; ".join(rationale_parts)
    return grade, rationale


def _empty_metrics() -> dict:
    return {
        "tumor_pixels":         0,
        "tumor_ratio":          0.0,
        "tumor_area_mm2":       0.0,
        "tumor_volume_mm3":     0.0,
        "tumor_volume_cm3":     0.0,
        "max_diameter_mm":      0.0,
        "bbox_height_mm":       0.0,
        "bbox_width_mm":        0.0,
        "circularity":          0.0,
        "boundary_roughness":   1.0,
        "size_severity":        "None detected",
        "who_grade_suggestion": "N/A",
        "grade_rationale":      "No tumor mask",
        "pixel_spacing_mm":     DEFAULT_PIXEL_SPACING_MM,
        "slice_thickness_mm":   DEFAULT_SLICE_THICKNESS_MM,
    }
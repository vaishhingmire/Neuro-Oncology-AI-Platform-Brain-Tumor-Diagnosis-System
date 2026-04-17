import numpy as np
from scipy import ndimage
import cv2

def extract_features(mask: np.ndarray, brain_mask: np.ndarray = None) -> dict:
    if mask.sum() == 0:
        return {
            "tumor_found": False,
            "area_percent": 0.0,
            "location": "N/A",
            "centroid": None,
            "morphology": {
                "area": 0.0,
                "eccentricity": 0.0,
                "compactness": 0.0,
                "boundary_irregularity": 0.0,
                "skull_proximity": 0.0
            }
        }

    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {
            "tumor_found": False,
            "area_percent": 0.0,
            "location": "N/A",
            "centroid": None,
            "morphology": {
                "area": 0.0,
                "eccentricity": 0.0,
                "compactness": 0.0,
                "boundary_irregularity": 0.0,
                "skull_proximity": 0.0
            }
        }
        
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    
    # Eccentricity
    eccentricity = 0.0
    if len(largest) >= 5:
        (x, y), (MA, ma), angle = cv2.fitEllipse(largest)
        if ma > 0:
            a = ma / 2
            b = MA / 2
            eccentricity = np.sqrt(1 - (b**2)/(a**2)) if a >= b else np.sqrt(1 - (a**2)/(b**2))
            
    # Compactness (Perimeter^2 / Area) -> smaller is more compact (circle is 4*pi)
    compactness = (perimeter ** 2) / (area + 1e-6)
    
    # Boundary Irregularity: compare actual perimeter to bounding circle perimeter
    equivalent_diameter = np.sqrt(4 * area / np.pi)
    perfect_perimeter = np.pi * equivalent_diameter
    boundary_irregularity = perimeter / (perfect_perimeter + 1e-6)
    
    labeled, num = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, num+1))
    largest_idx = np.argmax(sizes) + 1
    tumor = labeled == largest_idx

    centroid = ndimage.center_of_mass(tumor)
    area_percent = round(tumor.sum() / mask.size * 100, 2)

    h, w = mask.shape
    vertical = "upper" if centroid[0] < h/2 else "lower"
    horizontal = "left" if centroid[1] < w/2 else "right"
    location = f"{vertical} {horizontal}"
    
    # Skull Proximity (distance to the non-zero boundaries of brain mask or image edge)
    skull_proximity = 0.0
    if brain_mask is not None:
        # Distance transform from the background of the brain mask
        # 0 in brain_mask means background (skull)
        dist_transform = cv2.distanceTransform((brain_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        cy, cx = int(centroid[0]), int(centroid[1])
        if 0 <= cy < h and 0 <= cx < w:
            # Shortest distance from tumor centroid to the edge of the brain
            skull_proximity = float(dist_transform[cy, cx])
    else:
        # Fallback approximation: distance to nearest image edge
        cy, cx = int(centroid[0]), int(centroid[1])
        skull_proximity = float(min(cy, h - cy, cx, w - cx))

    return {
        "tumor_found": True,
        "area_percent": area_percent,
        "location": location,
        "centroid": centroid,
        "morphology": {
            "area": float(area),
            "eccentricity": float(eccentricity),
            "compactness": float(compactness),
            "boundary_irregularity": float(boundary_irregularity),
            "skull_proximity": float(skull_proximity)
        }
    }
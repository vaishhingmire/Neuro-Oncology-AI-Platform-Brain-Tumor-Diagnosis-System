import numpy as np
import cv2
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor


class SAMSegmentor:
    """
    Hybrid Medical SAM Segmentor

    Improvements:
    ✔ Grad-CAM semantic guidance
    ✔ YOLO bounding-box constraint
    ✔ Positive + Negative prompts
    ✔ Noise suppression
    """

    def __init__(self, checkpoint_path: str, model_type: str = "vit_b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)

        self.predictor = SamPredictor(sam)
        print(f"SAM loaded on {self.device} ✓")

    # ---------------------------------------------------------
    # MAIN SEGMENT FUNCTION
    # ---------------------------------------------------------
    def segment(self,
                img: Image.Image,
                cam: np.ndarray,
                box=None) -> dict:
        """
        Args:
            img : PIL MRI image
            cam : Grad-CAM heatmap (0–1 normalized)
            box : YOLO bounding box [x1,y1,x2,y2] (optional)

        Returns:
            segmentation dictionary
        """

        img_np = np.array(img.convert("RGB"))
        ih, iw = img_np.shape[:2]

        # -------------------------------------------------
        # STEP 1 — Prepare Grad-CAM
        # -------------------------------------------------
        cam_resized = cv2.resize(cam, (iw, ih))

        # pick hottest points
        flat_idx = np.argsort(cam_resized.flatten())[::-1]
        top_points = []
        min_dist = 20

        for idx in flat_idx:
            py = idx // iw
            px = idx % iw

            too_close = False
            for ex, ey in top_points:
                if abs(px-ex) < min_dist and abs(py-ey) < min_dist:
                    too_close = True
                    break

            if not too_close:
                top_points.append((px, py))

            if len(top_points) == 3:
                break

        # -------------------------------------------------
        # STEP 2 — Set SAM image
        # -------------------------------------------------
        self.predictor.set_image(img_np)

        # -------------------------------------------------
        # STEP 3 — Positive + Negative prompts
        # -------------------------------------------------
        
        points_list = []
        labels_list = []
        
        # 1. YOLO bbox center (if box provided)
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            points_list.append([cx, cy])
            labels_list.append(1)
            
        # 2. GradCAM hotspots
        for px, py in top_points:
            points_list.append([px, py])
            labels_list.append(1)
            
        # 3. Negative background points outside bbox
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            # Find low-activation points outside the bbox
            bg_mask = cam_resized < np.percentile(cam_resized, 20)
            bg_mask[y1:y2, x1:x2] = False
            bg_y, bg_x = np.where(bg_mask)
            
            if len(bg_y) > 0:
                num_bg = min(4, len(bg_y))
                idx = np.random.choice(len(bg_y), num_bg, replace=False)
                for i in idx:
                    points_list.append([bg_x[i], bg_y[i]])
                    labels_list.append(0)
        else:
            # Fallback if no box
            points_list.extend([
                [10, 10], [iw - 10, 10], [10, ih - 10], [iw - 10, ih - 10]
            ])
            labels_list.extend([0, 0, 0, 0])

        input_points = np.array(points_list) if len(points_list) > 0 else None
        input_labels = np.array(labels_list) if len(labels_list) > 0 else None

        # YOLO bounding box constraint
        input_box = np.array(box) if box is not None else None

        # -------------------------------------------------
        # STEP 4 — SAM Prediction
        # -------------------------------------------------
        masks, scores, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_box,
            multimask_output=True,
        )

        # -------------------------------------------------
        # STEP 5 — Select Best Mask
        # -------------------------------------------------
        cam_binary = (cam_resized >
                      np.percentile(cam_resized, 70)).astype(float)

        best_mask = None
        best_score = -1

        for mask, score in zip(masks, scores):
            mask_float = mask.astype(float)

            intersection = (mask_float * cam_binary).sum()
            union = (mask_float + cam_binary).clip(0, 1).sum()
            iou = intersection / (union + 1e-8)

            combined = float(score) * 0.4 + float(iou) * 0.6

            if combined > best_score:
                best_score = combined
                best_mask = mask

        if best_mask is None:
            best_mask = masks[np.argmax(scores)]

        # -------------------------------------------------
        # STEP 6 — Post-processing
        # -------------------------------------------------
        mask_uint8 = best_mask.astype(np.uint8) * 255

        kernel = np.ones((5, 5), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

        # -------------------------------------------------
        # STEP 7 — Extract tumor info
        # -------------------------------------------------
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bbox = None
        location = "unknown"
        area_pct = 0.0

        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)

            pad = 8
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(iw, x + w + pad)
            y2 = min(ih, y + h + pad)

            bbox = [x1, y1, x2, y2]

            area_pct = round(
                (mask_uint8 > 0).sum() / (ih * iw) * 100, 2)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            location = (
                f"{'upper' if cy < ih//2 else 'lower'} "
                f"{'left' if cx < iw//2 else 'right'}"
            )

        # resize display mask
        mask_256 = cv2.resize(
            mask_uint8, (256, 256),
            interpolation=cv2.INTER_NEAREST)

        return {
            "mask": mask_256,
            "mask_full": mask_uint8,
            "bbox": bbox,
            "score": round(best_score, 4),
            "area_pct": area_pct,
            "location": location,
            "points": top_points,
        }

    # ---------------------------------------------------------
    # OVERLAY FUNCTION
    # ---------------------------------------------------------
    def overlay_mask(self,
                     img: Image.Image,
                     mask: np.ndarray,
                     color=(255, 50, 50),
                     alpha=0.4):

        img_np = np.array(img.convert("RGB")).copy()
        ih, iw = img_np.shape[:2]

        mask_resized = cv2.resize(
            mask, (iw, ih),
            interpolation=cv2.INTER_NEAREST)

        colored = np.zeros_like(img_np)
        colored[mask_resized > 127] = color

        overlay = cv2.addWeighted(
            img_np, 1.0, colored, alpha, 0)

        contours, _ = cv2.findContours(
            (mask_resized > 127).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(overlay, contours, -1, color, 2)

        return overlay
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image


class YOLODetector:
    CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]
    COLORS = {
        "glioma":     (255, 50,  50),
        "meningioma": (255, 165,  0),
        "pituitary":  (255, 215,  0),
        "notumor":    (50,  205, 50),
    }

    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        self.model = YOLO(model_path)
        self.conf = conf_threshold

    def detect(self, img: Image.Image) -> dict:
        img_np = np.array(img)
        results = self.model(img_np, conf=self.conf, verbose=False)[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf  = round(box.conf[0].item(), 4)
                cls   = int(box.cls[0].item())
                name  = self.CLASS_NAMES[cls]
                w = x2 - x1
                h = y2 - y1
                area_pct = round(
                    (w * h) / (img_np.shape[0] * img_np.shape[1]) * 100, 2)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                ih, iw = img_np.shape[:2]
                location = (
                    f"{'upper' if cy < ih//2 else 'lower'} "
                    f"{'left' if cx < iw//2 else 'right'}"
                )
                detections.append({
                    "class":      name,
                    "confidence": conf,
                    "bbox":       [x1, y1, x2, y2],
                    "area_pct":   area_pct,
                    "location":   location,
                    "size_px":    w * h,
                })

        detections.sort(key=lambda x: x["confidence"], reverse=True)
        primary = detections[0] if detections else None
        tumor_found = primary is not None and primary["class"] != "notumor"

        return {
            "tumor_found": tumor_found,
            "detections":  detections,
            "primary":     primary,
            "total_count": len(detections),
        }

    def detect_from_mask(self, img: Image.Image, mask: np.ndarray,
                         class_name: str, conf: float) -> dict:
        img_np = np.array(img)
        ih, iw = img_np.shape[:2]

        mask_resized = cv2.resize(
            mask.astype(np.uint8), (iw, ih),
            interpolation=cv2.INTER_NEAREST)

        detections = []

        if mask_resized.sum() > 0:
            contours, _ = cv2.findContours(
                mask_resized, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)

                pad = 8
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(iw, x + w + pad)
                y2 = min(ih, y + h + pad)

                area_pct = round((w * h) / (ih * iw) * 100, 2)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                location = (
                    f"{'upper' if cy < ih//2 else 'lower'} "
                    f"{'left' if cx < iw//2 else 'right'}"
                )
                detections.append({
                    "class":      class_name,
                    "confidence": conf,
                    "bbox":       [x1, y1, x2, y2],
                    "area_pct":   area_pct,
                    "location":   location,
                    "size_px":    w * h,
                })

        primary = detections[0] if detections else None
        return {
            "tumor_found": len(detections) > 0,
            "detections":  detections,
            "primary":     primary,
            "total_count": len(detections),
        }

    def detect_from_gradcam(self, img: Image.Image, cam: np.ndarray,
                             class_name: str, conf: float) -> dict:
        img_np = np.array(img)
        ih, iw = img_np.shape[:2]

        # Resize cam to image size
        cam_resized = cv2.resize(cam, (iw, ih))

        # Find the single hottest point
        _, _, _, max_loc = cv2.minMaxLoc(cam_resized)
        cx, cy = max_loc

        # Draw tight box around hottest point (25% of image)
        half_w = int(iw * 0.15)
        half_h = int(ih * 0.15)

        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        x2 = min(iw, cx + half_w)
        y2 = min(ih, cy + half_h)

        area_pct = round(((x2-x1) * (y2-y1)) / (ih * iw) * 100, 2)
        location = (
            f"{'upper' if cy < ih//2 else 'lower'} "
            f"{'left' if cx < iw//2 else 'right'}"
        )

        detections = [{
            "class":      class_name,
            "confidence": conf,
            "bbox":       [x1, y1, x2, y2],
            "area_pct":   area_pct,
            "location":   location,
            "size_px":    (x2-x1) * (y2-y1),
        }]

        return {
            "tumor_found": True,
            "detections":  detections,
            "primary":     detections[0],
            "total_count": 1,
        }

    def draw_boxes(self, img: Image.Image,
                   detection_result: dict) -> np.ndarray:
        img_np = np.array(img.convert("RGB"))

        for det in detection_result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            color = self.COLORS.get(det["class"], (255, 255, 255))
            label = f"{det['class']} {det['confidence']*100:.1f}%"

            # Bounding box
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 3)

            # Label background
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(
                img_np,
                (x1, y1 - th - 12),
                (x1 + tw + 8, y1),
                color, -1)
            cv2.putText(
                img_np, label, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)

            # Center crosshair
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.drawMarker(
                img_np, (cx, cy), color,
                cv2.MARKER_CROSS, 20, 2)

        return img_np
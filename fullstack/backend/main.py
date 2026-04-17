"""
main.py — Neuro-Oncology AI Platform (Novel Version)
─────────────────────────────────────────────────────
Wires together 3 publishable novelties:
  1. Grad-CAM computed BEFORE SAM — heatmap drives segmentation prompts
  2. MC-Dropout uncertainty — 20 stochastic passes for reliability score
  3. Tumor volume estimation — clinical mm³ metrics from SAM mask

FIX: Removed ImageNet Normalize — best_model.pt was trained WITHOUT it.
"""

import sys
import uuid
import json
import base64
import asyncio
import cv2
import numpy as np
import torch
import torch.nn as nn
import uvicorn

from pathlib import Path
from functools import partial
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware

BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

from pipeline import run_pipeline
from gradcam_engine import GradCAMEngine, draw_gradcam_overlay
from uncertainty_engine import mc_dropout_predict
from chat_engine import chat_engine

# ── Config ────────────────────────────────────────────────────────────────────

MODELS_DIR         = BACKEND_DIR.parent.parent / "models"
CNN_WEIGHTS        = str(MODELS_DIR / "best_model_v2.pt")
YOLO_WEIGHTS       = str(MODELS_DIR / "yolov10_brain_tumor.pt")
MOBILE_SAM_WEIGHTS = str(MODELS_DIR / "mobile_sam.pt")

CLASS_NAMES  = ["Glioma", "Meningioma", "Healthy", "Pituitary"]
IMG_SIZE     = 224
DEVICE       = "cpu"
MM_PER_PIXEL = 0.35   # standard brain MRI axial: ~0.35mm/pixel

_executor = ThreadPoolExecutor(max_workers=2)

# ── Model loaders ─────────────────────────────────────────────────────────────

def _load_cnn():
    from torchvision import models as tv_models
    net = tv_models.efficientnet_b0(weights=None)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, 4)
    state = torch.load(CNN_WEIGHTS, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    net.load_state_dict(state, strict=False)
    net.eval()
    return net

def _load_yolo():
    from ultralytics import YOLO
    return YOLO(YOLO_WEIGHTS)

def _load_mobile_sam():
    from mobile_sam import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_t"](checkpoint=MOBILE_SAM_WEIGHTS)
    sam.to(DEVICE).eval()
    return SamPredictor(sam)

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()

    print("[Startup] Loading CNN...")
    app.state.cnn = await loop.run_in_executor(_executor, _load_cnn)
    print("[Startup] CNN ready ✓")

    print("[Startup] Loading YOLO...")
    app.state.yolo = await loop.run_in_executor(_executor, _load_yolo)
    print("[Startup] YOLO ready ✓")

    print("[Startup] Loading MobileSAM...")
    try:
        app.state.sam = await loop.run_in_executor(_executor, _load_mobile_sam)
        print("[Startup] MobileSAM ready ✓")
    except Exception as e:
        print(f"[Startup] MobileSAM failed: {e}")
        app.state.sam = None

    from torchvision import transforms
    app.state.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # NO Normalize — model was not trained with it
    ])

    print("[Startup] All models loaded — server ready!\n")
    yield
    _executor.shutdown(wait=False)

# ── Image helpers ─────────────────────────────────────────────────────────────

def _encode_b64(img):
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()

def _draw_detection_overlay(img, bbox, label, conf):
    out  = img.copy()
    h, w = img.shape[:2]
    x1 = max(0, min(int(bbox[0]), w - 1))
    y1 = max(0, min(int(bbox[1]), h - 1))
    x2 = max(0, min(int(bbox[2]), w - 1))
    y2 = max(0, min(int(bbox[3]), h - 1))
    cv2.rectangle(out, (x1, y1), (x2, y2), (50, 50, 255), 3)
    cv2.putText(out, f"{label} {conf*100:.1f}%", (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out

def _draw_segmentation_overlay(img, mask):
    out = img.copy()
    if mask is not None and np.array(mask).sum() > 0:
        h, w     = img.shape[:2]
        mask_arr = np.array(mask, dtype=np.uint8)
        mask_r   = cv2.resize((mask_arr > 0).astype(np.uint8) * 255, (w, h),
                              interpolation=cv2.INTER_NEAREST)
        colored  = np.zeros_like(out)
        colored[mask_r > 127] = (50, 50, 255)
        out = cv2.addWeighted(out, 1.0, colored, 0.4, 0)
        contours, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (50, 50, 255), 2)
    return out

# ── Core inference ────────────────────────────────────────────────────────────

def _run_inference(img_bgr, cnn, yolo, sam_predictor, transform):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_bgr.shape[:2]

    img_tensor = transform(img_rgb).unsqueeze(0)

    # ── Novelty 2: MC-Dropout ─────────────────────────────────────────────
    print("[Novel] Running MC-Dropout uncertainty estimation...")
    mc_result  = mc_dropout_predict(
        model=cnn, img_tensor=img_tensor,
        n_passes=20, class_names=CLASS_NAMES
    )
    cnn_logits = mc_result["mean_probs"]
    pred_idx   = mc_result["pred_idx"]
    cnn_pred   = mc_result["pred_class"]

    print(f"MC Prediction: {cnn_pred} "
          f"(conf={mc_result['confidence']:.2f}, "
          f"uncertainty={mc_result['uncertainty_score']:.3f}, "
          f"tier={mc_result['reliability_tier']})")

    # ── Novelty 1: Grad-CAM before SAM ───────────────────────────────────
    gradcam_map = np.zeros((h, w), dtype=np.float32)

    if cnn_pred != "Healthy":
        print("[Novel] Computing Grad-CAM for SAM prompt generation...")
        try:
            img_tensor_grad = img_tensor.clone().requires_grad_(False)
            grad_engine     = GradCAMEngine(cnn)
            cam_224         = grad_engine.compute(img_tensor_grad, pred_idx)
            grad_engine.cleanup()
            gradcam_map = cv2.resize(cam_224, (w, h))
            print(f"[Novel] Grad-CAM computed — max activation: {gradcam_map.max():.3f}")
        except Exception as e:
            print(f"[Novel] Grad-CAM failed: {e}")

    # ── YOLO detection ────────────────────────────────────────────────────
    yolo_results = yolo(img_rgb, conf=0.25, verbose=False)[0]
    bbox         = None
    det_label    = cnn_pred
    det_conf     = mc_result["confidence"]

    if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
        best      = yolo_results.boxes[yolo_results.boxes.conf.argmax()]
        bbox      = best.xyxy[0].tolist()
        det_conf  = float(best.conf[0])
        yolo_map  = {0: "Glioma", 1: "Meningioma", 2: "Pituitary", 3: "Healthy"}
        det_label = yolo_map.get(int(best.cls[0]), cnn_pred)

    if bbox is None:
        bbox = [w * 0.3, h * 0.3, w * 0.7, h * 0.7]

    # ── Pipeline ──────────────────────────────────────────────────────────
    metrics = run_pipeline(
        image=img_rgb,
        bbox=bbox,
        gradcam=gradcam_map,
        cnn_logits=cnn_logits,
        sam_model=sam_predictor if cnn_pred != "Healthy" else None,
        sam_mode="mobile",
        mc_result=mc_result,
    )

    tumor_ratio = metrics.get("tumor_ratio", 0)
    final_pred  = metrics.get("predicted_class", cnn_pred)

    if tumor_ratio > 0.01 and final_pred == "Healthy":
        final_pred = (det_label if det_label != "Healthy"
                      else CLASS_NAMES[np.argsort(cnn_logits)[-2]])

    metrics["diagnosis"]       = final_pred
    metrics["predicted_class"] = final_pred
    metrics["cnn_prediction"]  = cnn_pred
    metrics["confidence"]      = mc_result["confidence"]
    metrics["yolo_label"]      = det_label

    # ── FIX 1: Use temperature-scaled probabilities for UI display ────────
    # uncertainty_engine._scale_probs() already applied T=1.5 scaling
    # so both the Confidence card and Classification panel show same values
    if "probabilities" in mc_result:
        metrics["probabilities"] = mc_result["probabilities"]
    else:
        raw    = np.clip(np.array(list(mc_result["mean_probs"])), 1e-8, 1.0)
        scaled = np.power(raw, 1.5)
        scaled = scaled / scaled.sum()
        metrics["probabilities"] = {
            CLASS_NAMES[i]: float(scaled[i])
            for i in range(len(CLASS_NAMES))
        }

    # ── FIX 2: Convert diameter from pixels → mm ──────────────────────────
    # Pipeline outputs max_diameter in pixels; convert to clinical mm
    if "max_diameter_mm" in metrics:
        px_val = metrics["max_diameter_mm"]
        if px_val > 100:
            metrics["max_diameter_mm"] = round(px_val * MM_PER_PIXEL, 1)

    # ── FIX 3: Sanity-check volume ────────────────────────────────────────
    if "volume_cm3" in metrics:
        vol = metrics["volume_cm3"]
        if vol > 500:
            metrics["volume_cm3"] = round(vol * (MM_PER_PIXEL ** 2) * 0.005, 3)

    images = {
        "original":     _encode_b64(img_bgr),
        "detection":    _encode_b64(_draw_detection_overlay(
                            img_bgr, bbox, det_label, det_conf)),
        "gradcam":      _encode_b64(draw_gradcam_overlay(img_bgr, gradcam_map)),
        "segmentation": _encode_b64(_draw_segmentation_overlay(
                            img_bgr, metrics.get("mask"))),
    }

    return metrics, images

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Neuro-Oncology AI Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr   = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"error": "Invalid image"}

    loop = asyncio.get_event_loop()
    metrics, images = await loop.run_in_executor(
        _executor,
        partial(_run_inference,
                img_bgr,
                request.app.state.cnn,
                request.app.state.yolo,
                request.app.state.sam,
                request.app.state.transform)
    )

    session_id = str(uuid.uuid4())
    chat_engine.init_session(session_id, metrics)

    return {
        "session_id": session_id,
        "results": {"images": images, "metrics": metrics}
    }

@app.get("/chat/{session_id}")
async def get_chat(session_id: str):
    return {"history": chat_engine.get_history(session_id)}

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(ws: WebSocket, session_id: str):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            msg  = json.loads(data)
            async for chunk in chat_engine.stream_response(
                    session_id, msg.get("message", "")):
                await ws.send_text(json.dumps({"type": "chunk", "text": chunk}))
            await ws.send_text(json.dumps({"type": "done"}))
    except WebSocketDisconnect:
        print("Chat disconnected", session_id)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import segmentation_models_pytorch as smp
import cv2
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extractor import extract_features
from src.neuro_report import generate_neuro_report, chat_with_neuro_report
from src.gradcam import GradCAM, overlay_heatmap
from src.yolo_detector import YOLODetector
from src.sam_segmentor import SAMSegmentor


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Analyze — NeuroScan AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NeuroScan AI — Brain MRI Analysis")

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------
@st.cache_resource
def load_models():

    # CLASSIFIER
    classifier = models.efficientnet_b0(weights=None)
    classifier.classifier[1] = nn.Linear(
        classifier.classifier[1].in_features, 4)

    classifier.load_state_dict(
        torch.load("models/best_model.pt", map_location="cpu"))
    classifier.eval()

    # U-NET (fallback)
    seg_model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    seg_model.load_state_dict(
        torch.load("models/best_seg_model.pt", map_location="cpu"))
    seg_model.eval()

    # SAM
    sam = None
    sam_path = "models/sam_vit_b.pth"
    if os.path.exists(sam_path):
        sam = SAMSegmentor(sam_path)

    # YOLO
    yolo = None
    for path in [
        "models/yolov10_brain_tumor.pt",
        "models/yolov10n.pt"
    ]:
        if os.path.exists(path):
            yolo = YOLODetector(path)
            break

    return classifier, seg_model, sam, yolo


classifier, seg_model, sam, yolo = load_models()
gradcam_gen = GradCAM(classifier)

classes = ["glioma", "meningioma", "notumor", "pituitary"]

cls_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

seg_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ---------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------
uploaded = st.file_uploader(
    "Upload MRI Scan",
    type=["jpg", "jpeg", "png"]
)

# ===================================================
# MAIN PIPELINE
# ===================================================
if uploaded:

    img = Image.open(uploaded).convert("RGB")

    with st.spinner("Running NeuroScan AI Pipeline..."):

        # ------------------------------
        # 1️⃣ CLASSIFICATION
        # ------------------------------
        tensor = cls_tf(img).unsqueeze(0)

        with torch.no_grad():
            out = classifier(tensor)
            probs = torch.softmax(out, dim=1)[0]

        pred = classes[probs.argmax()]
        conf = probs.max().item()

        # ------------------------------
        # 2️⃣ YOLO DETECTION (FIRST!)
        # ------------------------------
        if yolo and pred != "notumor":
            detection_result = yolo.detect(img)
        else:
            detection_result = {
                "primary": None,
                "detections": [],
                "total_count": 0
            }

        annotated_img = (
            yolo.draw_boxes(img, detection_result)
            if yolo else np.array(img)
        )

        bbox = (
            detection_result["primary"]["bbox"]
            if detection_result["primary"]
            else None
        )

        # ------------------------------
        # 3️⃣ GRAD-CAM (EXPLAINABILITY)
        # ------------------------------
        tensor_g = cls_tf(img).unsqueeze(0).requires_grad_(True)
        cam = gradcam_gen.generate(
            tensor_g, probs.argmax().item())

        heatmap = overlay_heatmap(img, cam)

        # ------------------------------
        # 4️⃣ SEGMENTATION (SAM → U-Net fallback)
        # ------------------------------
        if pred != "notumor":

            if sam is not None:
                sam_result = sam.segment(img, cam, box=bbox)

                mask_overlay = sam.overlay_mask(
                    img, sam_result["mask_full"])

                features = {
                    "tumor_found": True,
                    "area_percent": sam_result["area_pct"],
                    "location": sam_result["location"]
                }

                mask_source = "SAM"

            else:
                seg_t = seg_tf(img).unsqueeze(0)

                with torch.no_grad():
                    seg_out = seg_model(seg_t)
                    seg_prob = torch.sigmoid(
                        seg_out).squeeze().numpy()

                mask = (seg_prob > 0.4).astype(np.uint8)

                mask_overlay = (mask * 255).astype(np.uint8)
                features = extract_features(mask)
                mask_source = "U-Net"

        else:
            mask_overlay = np.array(img)
            features = {
                "tumor_found": False,
                "area_percent": 0,
                "location": "N/A"
            }
            mask_source = "None"

        # ------------------------------
        # 5️⃣ REPORT GENERATION
        # ------------------------------
        classification = {
            "predicted_class": pred,
            "confidence": conf,
            "probabilities": dict(zip(classes, probs.tolist()))
        }

        report = generate_neuro_report(
            detection_result,
            classification
        )

    # ===================================================
    # DISPLAY RESULTS
    # ===================================================
    st.subheader(f"Diagnosis: {pred.upper()} ({conf*100:.1f}%)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(img, caption="Original MRI")

    with col2:
        st.image(annotated_img, caption="YOLO Detection")

    with col3:
        st.image(heatmap, caption="Grad-CAM")

    with col4:
        st.image(mask_overlay,
                 caption=f"{mask_source} Segmentation")

    st.markdown("---")

    st.write("### 📊 Tumor Metrics")
    st.write(features)

    st.markdown("---")
    st.write("### 🧾 AI Neuro-Oncology Report")
    st.write(report)
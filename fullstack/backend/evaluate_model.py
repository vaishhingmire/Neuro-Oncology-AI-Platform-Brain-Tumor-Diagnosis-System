"""
evaluate_model.py
─────────────────
Generates all figures needed for the research paper:
  1. Confusion matrix (heatmap)
  2. Per-class accuracy table
  3. Precision / Recall / F1 table
  4. Grad-CAM visualization samples

Run from project root:
  python evaluate_model.py
"""

import os
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (confusion_matrix, classification_report,
                              precision_score, recall_score, f1_score)
from PIL import Image
from pathlib import Path
import cv2

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH  = Path(r"C:\Users\vaishnavi hingmire\OneDrive\Desktop\mri_project\models\best_model_v3.pt")
TEST_DIR    = Path(r"C:\Users\vaishnavi hingmire\Downloads\archive (1)\Testing")
OUTPUT_DIR  = Path(r"C:\Users\vaishnavi hingmire\OneDrive\Desktop\mri_project\paper_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Match ImageFolder alphabetical order
CLASS_NAMES = ["Glioma", "Meningioma", "Healthy", "Pituitary"]
IMG_SIZE    = 224

print("="*60)
print("Neuro-Oncology AI — Paper Evaluation Script")
print("="*60)

# ── Load model ────────────────────────────────────────────────────────────────

model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 4)
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()
print(f"Model loaded: {MODEL_PATH.name}")

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])

# ── Run evaluation ────────────────────────────────────────────────────────────

dataset = ImageFolder(root=str(TEST_DIR), transform=transform)
loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"Test samples: {len(dataset)}")
print(f"Class order: {dataset.classes}")

all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:
        outputs = model(images)
        preds   = outputs.argmax(1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# Map dataset class indices to display names
idx_to_display = {}
for cls, idx in dataset.class_to_idx.items():
    display = "Healthy" if cls == "no_tumor" else cls.capitalize()
    idx_to_display[idx] = display

display_labels = [idx_to_display[i] for i in range(4)]
overall_acc    = (all_preds == all_labels).mean() * 100
print(f"\nOverall Accuracy: {overall_acc:.2f}%")

# ── Figure 1: Confusion Matrix ────────────────────────────────────────────────

cm = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
plt.colorbar(im, ax=ax, label='Percentage (%)')

ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(display_labels, fontsize=12)
ax.set_yticklabels(display_labels, fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
ax.set_title('Confusion Matrix — EfficientNet-B0\n(Brain Tumor MRI Classification)',
             fontsize=14, fontweight='bold', pad=15)

for i in range(4):
    for j in range(4):
        color = 'white' if cm_norm[i, j] > 50 else 'black'
        ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.1f}%)',
                ha='center', va='center', fontsize=11,
                color=color, fontweight='bold')

plt.tight_layout()
cm_path = OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Confusion matrix saved: {cm_path}")

# ── Figure 2: Per-class Metrics Bar Chart ────────────────────────────────────

precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
recall    = recall_score(all_labels, all_preds, average=None, zero_division=0)
f1        = f1_score(all_labels, all_preds, average=None, zero_division=0)
per_class_acc = cm.diagonal() / cm.sum(axis=1)

x     = np.arange(4)
width = 0.2
fig, ax = plt.subplots(figsize=(11, 6))

bars1 = ax.bar(x - 1.5*width, per_class_acc*100, width, label='Accuracy',  color='#2196F3', alpha=0.85)
bars2 = ax.bar(x - 0.5*width, precision*100,     width, label='Precision', color='#4CAF50', alpha=0.85)
bars3 = ax.bar(x + 0.5*width, recall*100,        width, label='Recall',    color='#FF9800', alpha=0.85)
bars4 = ax.bar(x + 1.5*width, f1*100,            width, label='F1-Score',  color='#9C27B0', alpha=0.85)

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Tumor Class', fontsize=13, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
ax.set_title('Per-Class Performance Metrics\n(Accuracy, Precision, Recall, F1-Score)',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(display_labels, fontsize=12)
ax.set_ylim(0, 115)
ax.legend(fontsize=11, loc='lower right')
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=overall_acc, color='red', linestyle='--', alpha=0.5, label=f'Overall Acc: {overall_acc:.1f}%')

plt.tight_layout()
metrics_path = OUTPUT_DIR / "per_class_metrics.png"
plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Per-class metrics saved: {metrics_path}")

# ── Figure 3: System Architecture Diagram ────────────────────────────────────

fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_facecolor('#0D1117')
fig.patch.set_facecolor('#0D1117')

def box(ax, x, y, w, h, color, title, subtitle="", fontsize=9):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.1", linewidth=2,
        edgecolor=color, facecolor=color+'22')
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.15 if subtitle else 0),
            title, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=color)
    if subtitle:
        ax.text(x + w/2, y + h/2 - 0.25, subtitle,
                ha='center', va='center', fontsize=7, color='#AAAAAA')

def arrow(ax, x1, y1, x2, y2, color='#555555'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

# Input
box(ax, 0.3, 2.8, 1.8, 1.4, '#64B5F6', 'MRI Input', 'Brain Scan\n(any format)', 10)

# CNN
box(ax, 2.5, 2.8, 2.2, 1.4, '#81C784', 'EfficientNet-B0', 'CNN Classifier\n4 classes', 10)
arrow(ax, 2.1, 3.5, 2.5, 3.5)

# MC Dropout
box(ax, 2.5, 0.4, 2.2, 1.8, '#FFB74D', 'MC-Dropout\n(Novelty 2)', '20 stochastic\nforward passes\n→ uncertainty', 9)
ax.annotate('', xy=(3.6, 2.8), xytext=(3.6, 2.2),
            arrowprops=dict(arrowstyle='->', color='#FFB74D', lw=1.5))

# GradCAM
box(ax, 5.1, 2.8, 2.2, 1.4, '#FF8A65', 'Grad-CAM\n(Novelty 1)', 'Heatmap\ngeneration', 9)
arrow(ax, 4.7, 3.5, 5.1, 3.5)

# YOLO
box(ax, 5.1, 0.4, 2.2, 1.8, '#CE93D8', 'YOLOv10\nDetector', 'Bounding box\nextraction', 9)
arrow(ax, 2.1, 3.0, 5.1, 1.3, '#555555')

# SAM Prompting
box(ax, 7.7, 2.8, 2.4, 1.4, '#FF8A65', 'SAM Prompting\n(Novelty 1)', 'GradCAM hotspots\n→ point prompts', 9)
arrow(ax, 7.3, 3.5, 7.7, 3.5)
arrow(ax, 6.2, 2.2, 8.0, 2.8, '#CE93D8')

# MobileSAM
box(ax, 7.7, 0.4, 2.4, 1.8, '#4DB6AC', 'MobileSAM', 'Tumor\nsegmentation\nmask', 9)
arrow(ax, 8.9, 2.8, 8.9, 2.2)

# Volume
box(ax, 10.5, 2.8, 2.2, 1.4, '#4DB6AC', 'Volume Est.\n(Novelty 3)', 'mm³ + WHO\ngrade', 9)
arrow(ax, 10.1, 3.5, 10.5, 3.5)
arrow(ax, 8.9, 1.3, 10.5, 2.9, '#4DB6AC')

# Fusion
box(ax, 13.1, 2.5, 2.5, 2.0, '#EF5350', 'Smart Fusion', 'CNN + SAM\ncross-validation\n→ final diagnosis', 9)
arrow(ax, 12.7, 3.5, 13.1, 3.5)
arrow(ax, 4.6, 1.3, 13.3, 2.5, '#FFB74D')

# Title
ax.text(8, 6.5, 'Neuro-Oncology AI Pipeline Architecture',
        ha='center', va='center', fontsize=15,
        fontweight='bold', color='white')
ax.text(8, 6.0,
        'Novel: Grad-CAM → SAM Prompts  |  MC-Dropout Uncertainty  |  Tumor Volume Estimation',
        ha='center', va='center', fontsize=10, color='#AAAAAA')

arch_path = OUTPUT_DIR / "architecture_diagram.png"
plt.savefig(arch_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
plt.close()
print(f"✓ Architecture diagram saved: {arch_path}")

# ── Print metrics table ───────────────────────────────────────────────────────

print("\n" + "="*60)
print("PAPER METRICS TABLE")
print("="*60)
print(f"{'Class':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-"*55)
for i, cls in enumerate(display_labels):
    print(f"{cls:<15} {per_class_acc[i]*100:>9.1f}% "
          f"{precision[i]*100:>9.1f}% "
          f"{recall[i]*100:>9.1f}% "
          f"{f1[i]*100:>9.1f}%")
print("-"*55)
macro_p = precision.mean()*100
macro_r = recall.mean()*100
macro_f = f1.mean()*100
print(f"{'Macro Avg':<15} {overall_acc:>9.1f}% {macro_p:>9.1f}% {macro_r:>9.1f}% {macro_f:>9.1f}%")
print("="*60)

print(f"\n✓ All figures saved to: {OUTPUT_DIR}")
print("\nFiles generated:")
print("  1. confusion_matrix.png    ← use in Section 4.3")
print("  2. per_class_metrics.png   ← use in Section 4.3")
print("  3. architecture_diagram.png ← use in Section 3.1")
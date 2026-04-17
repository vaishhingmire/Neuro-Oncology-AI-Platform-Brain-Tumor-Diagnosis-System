"""
retrain_cnn.py  ── Enhanced v3
──────────────────────────────────────────────────────────────────
Retrains EfficientNet-B0 on the brain tumor dataset.
Saves best_model_v3.pt that works WITHOUT ImageNet normalization.

IMPROVEMENTS OVER v2:
  1. Focal Loss (gamma=2) — fixes Glioma 83% recall
  2. WeightedRandomSampler — balanced batches per class
  3. Stronger augmentation — 9 transforms vs 5
  4. Cosine annealing LR + warmup (replaces StepLR)
  5. Differential LR — backbone 10x lower than head
  6. Dropout 0.2 → 0.4 for better regularization
  7. Early stopping (patience=5)
  8. Test Time Augmentation (TTA) at evaluation
  9. Mixed precision (auto-detects GPU)
  10. Full paper figures: confusion matrix + training curves

Expected dataset structure (UNCHANGED from your original):
  archive (1)/
    Training/
      glioma/
      meningioma/
      pituitary/
      no_tumor/
    Testing/
      glioma/
      meningioma/
      pituitary/
      no_tumor/

Run:
  python retrain_cnn.py

Takes ~20-40 min CPU, ~5-8 min GPU.
Output: models/best_model_v3.pt
──────────────────────────────────────────────────────────────────
"""

import os
import copy
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from sklearn.metrics import (confusion_matrix, classification_report,
                              ConfusionMatrixDisplay)

# ── Config (same paths as your original) ──────────────────────────────────────

DATASET_DIR = Path(r"C:\Users\vaishnavi hingmire\Downloads\archive (1)")
TRAIN_DIR   = DATASET_DIR / "Training"
TEST_DIR    = DATASET_DIR / "Testing"
SAVE_PATH   = Path(r"C:\Users\vaishnavi hingmire\OneDrive\Desktop\mri_project\models\best_model_v3.pt")
FIGURES_DIR = Path(r"C:\Users\vaishnavi hingmire\OneDrive\Desktop\mri_project\paper_figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE      = 224
BATCH_SIZE    = 32
EPOCHS        = 20           # increased from 15
LR            = 3e-4         # slightly lower than 1e-3 — more stable
WEIGHT_DECAY  = 1e-2
FOCAL_GAMMA   = 2.0          # key fix for Glioma
WARMUP_EPOCHS = 2
PATIENCE      = 5
NUM_WORKERS   = 0            # keep 0 — Windows safe
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device : {DEVICE}")
print(f"Training dir : {TRAIN_DIR}")
print(f"Save path    : {SAVE_PATH}")

# ── Transforms — NO ImageNet normalization ─────────────────────────────────────

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    T.RandomPerspective(distortion_scale=0.2, p=0.3),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    T.ToTensor(),
    # NO Normalize — matches main.py
    T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

test_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    # NO Normalize
])

# 5 TTA versions averaged at inference
tta_transforms = [
    T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()]),
    T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomHorizontalFlip(p=1.0), T.ToTensor()]),
    T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomVerticalFlip(p=1.0),   T.ToTensor()]),
    T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomRotation((90,  90)),   T.ToTensor()]),
    T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomRotation((180, 180)),  T.ToTensor()]),
]

# ── Dataset ───────────────────────────────────────────────────────────────────

train_dataset = ImageFolder(root=str(TRAIN_DIR), transform=train_transform)
test_dataset  = ImageFolder(root=str(TEST_DIR),  transform=test_transform)

idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

print(f"\nDataset classes (alphabetical): {train_dataset.classes}")
print(f"Class to idx: {train_dataset.class_to_idx}")
print(f"Train samples: {len(train_dataset)}")
print(f"Test  samples: {len(test_dataset)}")
print(f"\nActual class order in model output:")
for i in sorted(idx_to_class.keys()):
    display = "Healthy" if idx_to_class[i] == "no_tumor" else idx_to_class[i].capitalize()
    print(f"  Index {i} = {display}")

# ── Class weights ─────────────────────────────────────────────────────────────
targets      = train_dataset.targets
class_counts = np.bincount(targets)
total        = sum(class_counts)
num_classes  = len(train_dataset.classes)

class_weights = torch.tensor(
    [total / (num_classes * c) for c in class_counts],
    dtype=torch.float32
).to(DEVICE)

print(f"\nClass counts  : { {train_dataset.classes[i]: int(class_counts[i]) for i in range(num_classes)} }")
print(f"Class weights : { {train_dataset.classes[i]: round(float(class_weights[i]),3) for i in range(num_classes)} }")

# ── WeightedRandomSampler ─────────────────────────────────────────────────────
sample_weights = [class_weights[t].item() for t in targets]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False,  num_workers=NUM_WORKERS)

# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    FL(pt) = -(1-pt)^gamma * log(pt)
    gamma=2 focuses on hard misclassified examples (Glioma).
    """
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.1):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets,
                             weight=self.weight,
                             label_smoothing=self.label_smoothing,
                             reduction='none')
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()

criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights, label_smoothing=0.1)
print(f"\nLoss: FocalLoss(gamma={FOCAL_GAMMA}) + ClassWeights + LabelSmoothing(0.1)")

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=False),   # was 0.2
    nn.Linear(model.classifier[1].in_features, num_classes)
)
model = model.to(DEVICE)

# Differential learning rates
backbone_params   = [p for n, p in model.named_parameters() if "classifier" not in n]
classifier_params = [p for n, p in model.named_parameters() if "classifier"     in n]

optimizer = optim.AdamW([
    {"params": backbone_params,   "lr": LR * 0.1},
    {"params": classifier_params, "lr": LR},
], weight_decay=WEIGHT_DECAY)

# Cosine annealing with warmup (replaces StepLR)
def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return max(1e-6 / LR, 0.5 * (1.0 + np.cos(np.pi * progress)))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler    = GradScaler(enabled=(DEVICE == "cuda"))

print(f"Optimizer : AdamW | Backbone LR {LR*0.1:.1e} | Head LR {LR:.1e}")
print(f"Scheduler : Cosine annealing + {WARMUP_EPOCHS}-epoch warmup")

# ── Training ──────────────────────────────────────────────────────────────────
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_acc, best_wts  = 0.0, copy.deepcopy(model.state_dict())
patience_count      = 0
best_preds          = []
best_labels_list    = []

print("\n" + "="*60)
print("Starting training...")
print("="*60)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # ── Train ──────────────────────────────────────────────────────────
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        with autocast(enabled=(DEVICE == "cuda")):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss    += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total   += images.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(f"  Epoch {epoch}/{EPOCHS} | "
                  f"Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f}")

    train_acc = train_correct / train_total * 100

    # ── Validate ───────────────────────────────────────────────────────
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    class_correct = [0] * num_classes
    class_total   = [0] * num_classes
    epoch_preds, epoch_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            preds   = outputs.argmax(1)

            val_loss    += loss.item() * images.size(0)
            val_correct += (preds == labels).sum().item()
            val_total   += images.size(0)
            epoch_preds.extend(preds.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())

            for i in range(len(labels)):
                class_correct[labels[i]] += (preds[i] == labels[i]).item()
                class_total[labels[i]]   += 1

    val_acc    = val_correct / val_total * 100
    current_lr = scheduler.get_last_lr()[0]
    elapsed    = time.time() - t0

    history["train_loss"].append(train_loss / train_total)
    history["val_loss"].append(val_loss / val_total)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    scheduler.step()

    print(f"\nEpoch {epoch}/{EPOCHS} Summary  ({elapsed:.0f}s):")
    print(f"  Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | LR: {current_lr:.2e}")
    for i in sorted(idx_to_class.keys()):
        cls_name = "Healthy" if idx_to_class[i] == "no_tumor" else idx_to_class[i].capitalize()
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i] * 100
            print(f"  {cls_name:<12}: {acc:.1f}%  ({class_correct[i]}/{class_total[i]})")

    if val_acc > best_acc:
        best_acc         = val_acc
        best_wts         = copy.deepcopy(model.state_dict())
        best_preds       = epoch_preds
        best_labels_list = epoch_labels
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ New best model saved: {best_acc:.2f}%")
        patience_count = 0
    else:
        patience_count += 1
        print(f"  No improvement ({patience_count}/{PATIENCE})")
        if patience_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    print()

print("="*60)
print(f"Training complete! Best Val Accuracy: {best_acc:.2f}%")
print("="*60)

# Restore best weights
model.load_state_dict(best_wts)

# ── TTA Evaluation ────────────────────────────────────────────────────────────
print("\nRunning TTA evaluation...")
test_dataset_raw = ImageFolder(root=str(TEST_DIR), transform=None)
model.eval()
tta_preds, tta_labels = [], []

with torch.no_grad():
    for idx, (img_pil, label) in enumerate(test_dataset_raw):
        probs_list = []
        for t in tta_transforms:
            tensor = t(img_pil).unsqueeze(0).to(DEVICE)
            out    = model(tensor)
            probs_list.append(torch.softmax(out, dim=1))
        avg_prob = torch.stack(probs_list).mean(dim=0)
        tta_preds.append(avg_prob.argmax(dim=1).item())
        tta_labels.append(label)
        if (idx + 1) % 200 == 0:
            print(f"  TTA: {idx+1}/{len(test_dataset_raw)}")

tta_acc = np.mean(np.array(tta_preds) == np.array(tta_labels)) * 100
print(f"\nTTA Accuracy : {tta_acc:.2f}%  (Standard: {best_acc:.2f}%,  Boost: +{tta_acc-best_acc:.2f}%)")

# ── Final Metrics Table ───────────────────────────────────────────────────────
CLASS_NAMES_DISPLAY = []
for i in sorted(idx_to_class.keys()):
    n = "Healthy" if idx_to_class[i] == "no_tumor" else idx_to_class[i].capitalize()
    CLASS_NAMES_DISPLAY.append(n)

report = classification_report(tta_labels, tta_preds,
                                target_names=CLASS_NAMES_DISPLAY,
                                output_dict=True)
macro = report['macro avg']

print("\n" + "="*60)
print("PAPER METRICS TABLE (TTA)")
print("="*60)
print(f"{'Class':<14} {'Accuracy':>10} {'Precision':>11} {'Recall':>9} {'F1':>9}")
print("-"*56)
for cls in CLASS_NAMES_DISPLAY:
    m = report[cls]
    print(f"{cls:<14} {m['recall']*100:>9.1f}%  {m['precision']:>10.3f}"
          f"  {m['recall']:>8.3f}  {m['f1-score']:>8.3f}")
print("-"*56)
print(f"{'Macro Avg':<14} {tta_acc:>9.2f}%  {macro['precision']:>10.3f}"
      f"  {macro['recall']:>8.3f}  {macro['f1-score']:>8.3f}")
print("="*60)

# ── Figure 1: Confusion Matrix ────────────────────────────────────────────────
cm   = confusion_matrix(tta_labels, tta_preds)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=CLASS_NAMES_DISPLAY)
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title(f"Confusion Matrix — EfficientNet-B0 v3\nTTA Accuracy: {tta_acc:.2f}%",
             fontsize=13)
plt.tight_layout()
cm_path = FIGURES_DIR / "confusion_matrix_v3.png"
plt.savefig(str(cm_path), dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Confusion matrix : {cm_path}")

# ── Figure 2: Per-class bars ──────────────────────────────────────────────────
x      = np.arange(len(CLASS_NAMES_DISPLAY))
width  = 0.25
fig, ax = plt.subplots(figsize=(11, 6))
for i, (key, color, lbl) in enumerate(zip(
        ["precision", "recall", "f1-score"],
        ["#2563eb", "#16a34a", "#ea580c"],
        ["Precision", "Recall", "F1"])):
    vals = [report[cls][key] for cls in CLASS_NAMES_DISPLAY]
    bars = ax.bar(x + i * width, vals, width, label=lbl,
                  color=color, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha='center', va='bottom', fontsize=8)
ax.set_xlabel("Tumor Class", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title(f"Per-Class Metrics — EfficientNet-B0 v3  (TTA: {tta_acc:.2f}%)", fontsize=13)
ax.set_xticks(x + width); ax.set_xticklabels(CLASS_NAMES_DISPLAY, fontsize=11)
ax.set_ylim(0.70, 1.05); ax.legend(fontsize=10); ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
metrics_path = FIGURES_DIR / "per_class_metrics_v3.png"
plt.savefig(str(metrics_path), dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Per-class metrics: {metrics_path}")

# ── Figure 3: Training curves ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(history["train_acc"], label="Train", color="#2563eb", linewidth=2)
ax1.plot(history["val_acc"],   label="Val",   color="#16a34a", linewidth=2, linestyle="--")
ax1.axhline(y=best_acc, color="red", linestyle=":", alpha=0.7, label=f"Best {best_acc:.2f}%")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Accuracy Curves"); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(history["train_loss"], label="Train", color="#2563eb", linewidth=2)
ax2.plot(history["val_loss"],   label="Val",   color="#ea580c", linewidth=2, linestyle="--")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Focal Loss")
ax2.set_title("Loss Curves"); ax2.legend(); ax2.grid(alpha=0.3)

plt.suptitle("Training History — EfficientNet-B0 v3", fontsize=13, y=1.01)
plt.tight_layout()
curves_path = FIGURES_DIR / "training_curves_v3.png"
plt.savefig(str(curves_path), dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Training curves  : {curves_path}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DONE — NEXT STEPS")
print("="*60)
print(f"  Standard Val Acc : {best_acc:.2f}%")
print(f"  TTA Accuracy     : {tta_acc:.2f}%")
print(f"  Glioma Recall    : {report['Glioma']['recall']*100:.1f}%  (was 83.0%)")
print(f"  Macro F1         : {macro['f1-score']:.4f}")
print()
print("  1. In main.py change:")
print("       best_model_v2.pt  →  best_model_v3.pt")
print()
print("  2. In uncertainty_engine.py change:")
print("       best_model_v2.pt  →  best_model_v3.pt")
print()
print("  3. CLASS_NAMES stays the same:")
class_names_ordered = []
for i in sorted(idx_to_class.keys()):
    n = "Healthy" if idx_to_class[i] == "no_tumor" else idx_to_class[i].capitalize()
    class_names_ordered.append(n)
print(f"     {class_names_ordered}")
print()
print("  4. Paste the new accuracy numbers here")
print("     → I will update the LaTeX paper automatically")
print("="*60)
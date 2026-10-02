# Model Version Control & Release Management Protocol

This document outlines the version control, checkpoint tracking, and GitHub release procedures for the **YOLO_ChestXray** project.

---

## 🎯 Objectives

1. **Reproducibility**: Ensure every training stage, hyperparameter set, and checkpoint can be traced back to exact code and dataset states.
2. **Fault Tolerance**: Protect against training session interruptions (e.g., Google Colab GPU timeout/disconnections) by saving incremental checkpoints.
3. **Artifact Integrity**: Preserve all evaluation metrics, confusion matrices, and precision-recall curves alongside trained weights for every milestone.

---

## 🏷️ Checkpoint Naming & Versioning Scheme

### Checkpoint Files

Checkpoints are saved at the end of every 10-epoch training stage under `checkpoints/` and synchronized with Google Drive:

```text
checkpoints/
├── checkpoint_10.pt      # Stage 1 (Epoch 10)
├── checkpoint_20.pt      # Stage 2 (Epoch 20)
├── checkpoint_30.pt      # Stage 3 (Epoch 30)
├── checkpoint_40.pt      # Stage 4 (Epoch 40)
├── checkpoint_50.pt      # Stage 5 (Epoch 50)
├── best.pt               # Highest validation mAP@0.5 achieved
└── last.pt               # Most recent step (used for uninterrupted resume)
```

> ⚠️ **Rule**: Never keep only `best.pt`. Every intermediate milestone checkpoint (`checkpoint_xx.pt`) must be archived to enable rollbacks and comparative ablations.

---

## 🚀 Stage-wise Training Schedule

| Stage | Epoch Range | Base Model / Resume Target | Output Checkpoint | Release Tag |
|:---|:---|:---|:---|:---|
| **Stage 1** | Epoch 1 – 10 | `yolov7.pt` (Pretrained) | `checkpoint_10.pt` | `Release_10` |
| **Stage 2** | Epoch 11 – 20 | `checkpoint_10.pt` | `checkpoint_20.pt` | `Release_20` |
| **Stage 3** | Epoch 21 – 30 | `checkpoint_20.pt` | `checkpoint_30.pt` | `Release_30` |
| **Stage 4** | Epoch 31 – 40 | `checkpoint_30.pt` | `checkpoint_40.pt` | `Release_40` |
| **Stage 5** | Epoch 41 – 50 | `checkpoint_40.pt` | `checkpoint_50.pt` | `Release_50` |

---

## 📊 Evaluation & Artifact Packaging

At the conclusion of each stage, validation is executed and the following deliverables are collected:

- `best.pt`
- `checkpoint_xx.pt`
- `results.csv`
- `results.png`
- `confusion_matrix.png`
- `PR_curve.png`
- `F1_curve.png`

---

## 🛑 Early Stopping & Convergence Criteria

Training proceeds to subsequent stages **only if** validation metrics (primarily `mAP@0.5` and `mAP@0.5:0.95`) demonstrate measurable gain.

### Example Convergence Tracking:
- **Epoch 10**: `mAP@0.5 = 0.32` ➔ ✅ Continue
- **Epoch 20**: `mAP@0.5 = 0.48` ➔ ✅ Continue
- **Epoch 30**: `mAP@0.5 = 0.59` ➔ ✅ Continue
- **Epoch 40**: `mAP@0.5 = 0.61` ➔ ✅ Continue
- **Epoch 50**: `mAP@0.5 = 0.61` ➔ 🛑 Performance plateaued; terminate training and select best checkpoint.

---

## 🐙 Git & GitHub Workflow

### 1. Daily Development & Code Commits
```bash
git add .
git commit -m "feat(training): update training pipeline and configs for stage X"
git push origin main
```

### 2. Creating GitHub Milestone Releases
When a 10-epoch stage completes:
```bash
# Tag the release commit
git tag -a Release_10 -m "Release after 10 epochs of YOLOv7 training on ChestXray8"
git push origin Release_10
```

Upload the packaged weights and evaluation plots (`checkpoint_10.pt`, `results.png`, `confusion_matrix.png`, etc.) directly to the GitHub Release assets.

---

## ☁️ Google Drive Backup Directory Layout

```text
My Drive/
└── YOLO_ChestXray/
    ├── checkpoints/
    │   ├── checkpoint_10.pt
    │   ├── checkpoint_20.pt
    │   ├── best.pt
    │   └── last.pt
    ├── results/
    │   ├── results.csv
    │   └── figures/
    └── releases/
        └── Release_10.zip
```

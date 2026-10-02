# YOLO_ChestXray: Chest X-ray Disease Detection with YOLOv7

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv7](https://img.shields.io/badge/Model-YOLOv7-orange.svg)](https://github.com/WongKinYiu/yolov7)
[![Dataset](https://img.shields.io/badge/Dataset-ChestXray8-green.svg)](https://www.kaggle.com/datasets/spritan1/yolo-annotated-chestxray-8-object-detection/data)
[![GitHub Release](https://img.shields.io/github/v/release/wupinlai/YOLO_ChestXray?include_prereleases&style=flat)](https://github.com/wupinlai/YOLO_ChestXray/releases)

---

## 📌 Project Overview

**YOLO_ChestXray** applies the state-of-the-art **YOLOv7** object detection architecture to detect and localize multiple thoracic pathologies on chest radiographs (CXR). Utilizing the annotated ChestXray8 dataset with bounding-box annotations across 14 distinct disease classes, this repository provides an end-to-end pipeline covering:

- 📊 **Dataset Preparation & Verification**
- 🚀 **Stage-wise Resumable Training** (optimized for Google Colab GPU sessions)
- 📈 **Comprehensive Validation & Metrics Tracking** (mAP@0.5, mAP@0.5:0.95, Confusion Matrix, PR/F1 Curves)
- 🔍 **Inference & Pathology Localization**
- 🏷️ **Checkpoint & Model Version Control** via GitHub Releases & Cloud Storage

---

## 🩺 Dataset Information

- **Dataset**: [ChestXray8 YOLO Annotated Object Detection](https://www.kaggle.com/datasets/spritan1/yolo-annotated-chestxray-8-object-detection/data)
- **Format**: YOLO format (Normalized bounding box: `<class_id> <x_center> <y_center> <width> <height>`)
- **Total Classes**: 14 Chest Pathologies

### Disease Classes (14)

| ID | Disease Name | ID | Disease Name |
|:---|:---|:---|:---|
| 0 | Atelectasis | 7 | Pneumothorax |
| 1 | Cardiomegaly | 8 | Consolidation |
| 2 | Effusion | 9 | Edema |
| 3 | Infiltration | 10 | Emphysema |
| 4 | Mass | 11 | Fibrosis |
| 5 | Nodule | 12 | Pleural Thickening |
| 6 | Pneumonia | 13 | Hernia |

---

## 📂 Repository Structure

```text
YOLO_ChestXray/
├── configs/
│   └── chestxray.yaml        # Dataset configuration & class definitions
├── datasets/
│   └── chestxray8/           # Dataset root (Train / Val image & label splits)
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       └── val/
│           ├── images/
│           └── labels/
├── notebooks/                # Jupyter / Google Colab training notebooks
├── scripts/                  # Training, evaluation, and inference scripts
│   ├── train.py
│   ├── validate.py
│   └── inference.py
├── checkpoints/              # Stored model weights (checkpoint_xx.pt, best.pt)
├── reports/                  # Validation reports, metric plots, PR curves
├── releases/                 # Packaged release archives & artifacts
├── plan.md                   # Full development roadmap & execution plan
├── version control.md        # Model & release version control protocol
└── README.md                 # Project documentation
```

---

## ⚙️ Configuration

Dataset configuration is specified in `configs/chestxray.yaml`:

```yaml
train: datasets/chestxray8/train/images
val: datasets/chestxray8/val/images

nc: 14

names:
  - Atelectasis
  - Cardiomegaly
  - Effusion
  - Infiltration
  - Mass
  - Nodule
  - Pneumonia
  - Pneumothorax
  - Consolidation
  - Edema
  - Emphysema
  - Fibrosis
  - Pleural_Thickening
  - Hernia
```

---

## 🚀 Training Strategy & Execution

To safeguard against Colab timeouts and session disconnections, training is partitioned into **10-epoch incremental stages** with full checkpointing and Google Drive synchronization.

### 5-Stage Schedule

| Stage | Epochs | Checkpoint Saved | Resumed From |
|:---|:---|:---|:---|
| **Stage 1** | 1 ~ 10 | `checkpoint_10.pt` | Pretrained YOLOv7 weights |
| **Stage 2** | 11 ~ 20 | `checkpoint_20.pt` | `checkpoint_10.pt` |
| **Stage 3** | 21 ~ 30 | `checkpoint_30.pt` | `checkpoint_20.pt` |
| **Stage 4** | 31 ~ 40 | `checkpoint_40.pt` | `checkpoint_30.pt` |
| **Stage 5** | 41 ~ 50 | `checkpoint_50.pt` | `checkpoint_40.pt` |

### Resuming Training
```bash
python train.py --resume runs/train/exp/weights/last.pt
```

### Validation & Metrics Collection
Run evaluation after each stage:
```bash
python test.py --data configs/chestxray.yaml --weights checkpoints/checkpoint_10.pt
```

Outputs generated:
- `results.csv` & `results.png`
- `confusion_matrix.png`
- `PR_curve.png` & `F1_curve.png`

---

## 🔍 Inference

Run detection on new chest radiograph images:

```bash
python detect.py --weights checkpoints/best.pt --source test_images/ --conf 0.25
```

Predictions will output bounding boxes, disease labels, and detection confidence scores under `runs/detect/`.

---

## 📦 Version Control & Release Management

Detailed release protocols and checkpoint management instructions can be found in [`version control.md`](./version%20control.md).

For every 10-epoch milestone, a GitHub Release is created bundling:
- Model weight: `checkpoint_xx.pt` and `best.pt`
- Metric logs: `results.csv`
- Evaluation figures: `results.png`, `confusion_matrix.png`, `PR_curve.png`, `F1_curve.png`

---

## 👤 Author & Acknowledgments

- **Author**: Wupin Lai
- **Project**: YOLO_ChestXray (Research Methods / AI Project)

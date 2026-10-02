# YOLO_ChestXray

使用 YOLOv7 針對 ChestXray8 YOLO Annotation Dataset 進行胸腔 X 光疾病偵測訓練。

## Project Goal

建立可偵測胸腔病灶的 YOLOv7 模型，並完成：

- Training
- Validation
- Inference
- Model Version Control
- GitHub Release Management

---

# Dataset

Dataset Source:

https://www.kaggle.com/datasets/spritan1/yolo-annotated-chestxray-8-object-detection/data

Dataset Features

- ChestXray8 Dataset
- YOLO Annotation Format
- Bounding Box Labels
- Train / Validation Split
- 14 Disease Classes

Example Diseases

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
- Pleural Thickening
- Hernia

---

# Development Environment

## Local Development

Antigravity IDE

### Source Repository

https://github.com/wupinlai/YOLO_ChestXray

---

## Training Environment

Google Colab

GPU Recommended

- Tesla T4
- L4
- A100

---

# Repository Structure

```text
YOLO_ChestXray
│
├── configs
│   └── chestxray.yaml
│
├── datasets
│   └── chestxray8
│
├── notebooks
│
├── scripts
│   ├── train.py
│   ├── validate.py
│   ├── inference.py
│
├── checkpoints
│
├── reports
│
├── releases
│
└── README.md
```

---

# Dataset Structure

```text
datasets/chestxray8
│
├── train
│   ├── images
│   └── labels
│
└── val
    ├── images
    └── labels
```

---

# Dataset Configuration

File:

```text
configs/chestxray.yaml
```

Example:

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

# Training Strategy

## Important

Do NOT train all epochs at once.

Google Colab may disconnect unexpectedly.

Training must support resume checkpoints.

---

## Recommended Training Schedule

### Stage 1

```text
Epoch 1~10
```

Save:

```text
checkpoint_10.pt
```

---

### Stage 2

```text
Epoch 11~20
```

Resume From:

```text
checkpoint_10.pt
```

Save:

```text
checkpoint_20.pt
```

---

### Stage 3

```text
Epoch 21~30
```

Save:

```text
checkpoint_30.pt
```

---

### Stage 4

```text
Epoch 31~40
```

Save:

```text
checkpoint_40.pt
```

---

### Stage 5

```text
Epoch 41~50
```

Save:

```text
checkpoint_50.pt
```

---

# Resume Training

Example:

```bash
python train.py \
  --resume runs/train/exp/weights/last.pt
```

Resume training after Colab disconnects.

---

# Checkpoint Management

Store every checkpoint.

```text
checkpoint_10.pt
checkpoint_20.pt
checkpoint_30.pt
checkpoint_40.pt
checkpoint_50.pt
```

Do not keep only:

```text
best.pt
```

---

# Validation

After every training stage:

```bash
python test.py \
  --data configs/chestxray.yaml \
  --weights checkpoint_xx.pt
```

Collect:

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95

Output:

```text
results.csv
results.png
confusion_matrix.png
PR_curve.png
F1_curve.png
```

---

# Inference

Example:

```bash
python detect.py \
  --weights best.pt \
  --source test_images/
```

Output:

```text
runs/detect/
```

Generated Results:

- Bounding Boxes
- Disease Name
- Confidence Score

---

# Google Drive Backup

All training outputs should be backed up to Google Drive.

```text
YOLO_ChestXray
│
├── checkpoints
├── results
└── releases
```

Files:

```text
best.pt
last.pt
results.csv
results.png
```

---

# Git Workflow

## Commit

```bash
git add .
git commit -m "update training"
git push origin main
```

---

# Release Strategy

Create releases every 10 epochs.

Example:

```text
Release_10
Release_20
Release_30
Release_40
Release_50
```

---

# Release Assets

Each release should include:

```text
best.pt
checkpoint_xx.pt
results.csv
results.png
confusion_matrix.png
PR_curve.png
F1_curve.png
```

---

# Early Stopping Rule

Continue training only when validation metrics improve.

Example:

```text
Epoch 10
mAP = 0.32

Epoch 20
mAP = 0.48

Epoch 30
mAP = 0.59

Epoch 40
mAP = 0.61

Epoch 50
mAP = 0.61
```

Stop training if model performance plateaus.

---

# Project Workflow

```text
Antigravity IDE
        ↓
GitHub
        ↓
Google Colab
        ↓
Dataset Preparation
        ↓
Train 10 Epoch
        ↓
Checkpoint
        ↓
Validation
        ↓
Release
        ↓
Resume Training
        ↓
Checkpoint
        ↓
Validation
        ↓
Release
        ↓
Best Model
        ↓
Inference
        ↓
Final Report
```

---

# Deliverables

- Source Code
- YOLOv7 Config
- Dataset Config
- Training Logs
- Validation Results
- Inference Results
- Checkpoints
- best.pt
- GitHub Releases
- Final Report

---

# Author

Wupin Lai

YOLO_ChestXray Project

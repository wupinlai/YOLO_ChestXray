# Plan1.md - AI Training, Validation, Inference and Error Analysis Framework

## Environment Check

1. Verify Runtime Environment
2. Verify GPU Availability
3. Verify Kaggle API Token
4. Download Dataset
5. Verify Dataset Integrity
6. Start Training

---

# Kaggle Access

Credential Location:

/content/drive/MyDrive/Secrets/kaggle_token.txt

Load Token:

```python
from google.colab import drive
import os

drive.mount('/content/drive')

with open('/content/drive/MyDrive/Secrets/kaggle_token.txt','r') as f:
    token = f.read().strip()

os.environ['KAGGLE_API_TOKEN'] = token
```

Verify:

```bash
kaggle datasets list
```

---

# Training Workflow

Training is divided into 5 stages.

Stage 1 : Epoch 1-10
Stage 2 : Epoch 11-20
Stage 3 : Epoch 21-30
Stage 4 : Epoch 31-40
Stage 5 : Epoch 41-50

After every stage:

1. Save checkpoint
2. Run validation
3. Save training metrics
4. Save validation metrics
5. Update metrics_history.csv

---

# Required Outputs Per Stage

```text
checkpoint_xx.pt
train_results.csv
train_results.png
val_results.csv
confusion_matrix.png
PR_curve.png
F1_curve.png
```

---

# Metrics History

```csv
Epoch,Precision,Recall,mAP50,mAP50_95,TrainLoss,ValLoss
```

Location:

```text
reports/metrics_history.csv
```

---

# Final Comparison Report

Generate:

```text
map50_comparison.png
map50_95_comparison.png
precision_recall_comparison.png
loss_comparison.png
training_summary.csv
```

---

# Best Model Selection

Criteria:

1. Highest mAP50_95
2. Highest mAP50
3. Lowest Validation Loss

Output:

```text
best_model.pt
```

---

# Final Inference Verification

Before training:

Randomly select one image from validation dataset.

Save:

```text
reports/final_inference/sample_image.jpg
```

After Epoch 50:

Run inference using:

```text
best_model.pt
```

Generate:

```text
reports/final_inference/inference_result.jpg
```

Image must contain:

- Bounding Box
- Class Label
- Confidence Score
- IoU Value

---

# Confidence Analysis

Generate:

```text
confidence_distribution.png
```

Include:

- Average Confidence
- Maximum Confidence
- Minimum Confidence

---

# Class Statistics

Generate:

```text
class_distribution.png
class_confidence.png
```

Generate CSV:

```text
detection_summary.csv
```

Columns:

```csv
Class,DetectionCount,AvgConfidence,MaxConfidence,MinConfidence
```

---

# IoU Analysis

Generate:

```text
iou_distribution.png
class_iou_comparison.png
iou_statistics.csv
```

Columns:

```csv
Image,Class,IoU,Confidence
```

Include:

- Average IoU
- Median IoU
- Maximum IoU
- Minimum IoU

---

# Error Analysis

Generate:

```text
false_positive.csv
false_negative.csv
misclassification.csv
class_performance.csv
error_statistics.csv
error_distribution.png
```

Error Types:

- False Positive
- False Negative
- Misclassification
- Low IoU

---

# Automated Error Annotation

Generate individual annotated images.

## False Positive

Folder:

```text
reports/error_analysis/false_positive/
```

Label:

```text
[FP]
Pred:<Class>
Conf:<Value>
```

## False Negative

Folder:

```text
reports/error_analysis/false_negative/
```

Label:

```text
[FN]
GT:<Class>
```

## Misclassification

Folder:

```text
reports/error_analysis/misclassification/
```

Label:

```text
[MC]
GT:<Class>
Pred:<Class>
Conf:<Value>
```

## Low IoU

Folder:

```text
reports/error_analysis/low_iou/
```

Label:

```text
[Low IoU]
IoU:<Value>
Conf:<Value>
```

---

# Error Annotation Standard

## Color Definition

```text
Ground Truth      : Green     (#00FF00)
Prediction        : Blue      (#0080FF)
False Positive    : Red       (#FF0000)
False Negative    : Orange    (#FFA500)
Misclassification : Purple    (#B400FF)
Low IoU           : Yellow    (#FFFF00)
Correct Detection : Cyan      (#00FFFF)
```

---

# Visualization Standard

## Font

Priority:

```text
Noto Sans
Arial
DejaVu Sans
```

## Font Size

```text
Bounding Box Label : 18 px
Confidence / IoU   : 16 px
Error Type         : 20 px Bold
Severity           : 22 px Bold
Legend             : 16 px
```

---

# Label Layout Rules

Bounding Box Label:

```text
<Class>
Conf=0.923
IoU=0.817
```

Placement Order:

1. Top Left
2. Top Right
3. Inside Top

Text must never exceed image boundary.

---

# Legend Rules

Position:

```text
Top Right Corner
```

Background:

```text
White
Opacity 80%
```

Content:

```text
Green   = Ground Truth
Blue    = Prediction
Red     = False Positive
Orange  = False Negative
Purple  = Misclassification
Yellow  = Low IoU
Cyan    = Correct Detection
```

---

# Severity Rules

CRITICAL:

```text
IoU < 0.30
or False Negative
```

WARNING:

```text
0.30 <= IoU < 0.50
```

NORMAL:

```text
IoU >= 0.50
```

---

# Information Panel

Displayed at image bottom.

Fields:

```text
Image Name
Ground Truth
Prediction
Confidence
IoU
Error Type
Severity
```

---

# Error Gallery

Generate:

```text
error_gallery.jpg
worst_iou_cases.jpg
top5_detections.jpg
```

Layout:

```text
2 x 2 Grid
```

Each image must display:

```text
Class
Confidence
IoU
Error Type
Severity
```

---

# Report Requirements

Generate:

```text
final_report.pdf
```

Sections:

Page 1 - Project Summary
Page 2 - Training Metrics
Page 3 - Original vs Inference Result
Page 4 - Detection Detail
Page 5 - Confidence Analysis
Page 6 - Class Statistics
Page 7 - IoU Analysis
Page 8 - Error Statistics
Page 9 - Error Gallery
Page 10+ - FP/FN/MC/Low IoU Cases

Original Image and Inference Result must be displayed side-by-side.

---

# Image Quality Requirements

```text
Minimum Resolution : 1920 x 1080
PDF Image Quality  : 300 DPI
```

Accessibility:

Color cannot be the only indicator.
Always display:

```text
FP
FN
MC
LowIoU
Correct
```

---

# Acceptance Criteria

```text
✓ Epoch 50 Completed
✓ Validation After Every Stage
✓ Training Metrics Saved
✓ Validation Metrics Saved
✓ Metrics History Generated
✓ Comparison Charts Generated
✓ Best Model Selected
✓ Random Dataset Image Selected
✓ Actual Inference Executed
✓ Annotated Inference Image Exported
✓ Confidence Statistics Generated
✓ Class Statistics Generated
✓ IoU Analysis Generated
✓ False Positive Analysis Generated
✓ False Negative Analysis Generated
✓ Misclassification Analysis Generated
✓ Auto Error Annotation Generated
✓ Error Gallery Generated
✓ Final Report PDF Generated
✓ Inference Embedded In Final Report
```

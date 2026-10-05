# Plan1.md - YOLO ChestXray Training & Validation Master Plan

Version: 2.0

Framework: YOLOv7
Platform: Google Colab GPU

This plan includes:
- YOLOv7 legacy compatibility environment builder
- GPU/CUDA validation
- Micromamba Python 3.10 environment creation
- YOLOv7 torch.load compatibility patch
- Resume training workflow
- Validation after every stage
- Metrics history tracking
- Confidence / IoU / Error analysis
- Runtime recovery
- Final PDF report generation

## Key Requirement

YOLOv7 must run in a compatible environment.
If Python > 3.10 or Torch >= 2.6, automatically build an isolated Python 3.10 + Torch 2.0.1 environment before training.

## Training Stages

Stage 1: Epoch 1-10
Stage 2: Epoch 11-20
Stage 3: Epoch 21-30
Stage 4: Epoch 31-40
Stage 5: Epoch 41-50

Validation is required after every stage.

## Compatibility Environment

Python 3.10
Torch 2.0.1
TorchVision 0.15.2
NumPy 1.23.5

## Resume Policy

Always use:

--resume

Never continue training using only:

--weights checkpoint_xx.pt

## Final Deliverables

- metrics_history.csv
- checkpoint_10.pt ~ checkpoint_50.pt
- best_model.pt
- confidence_distribution.png
- iou_distribution.png
- error_gallery.jpg
- final_report.pdf
- project_manifest.json

## Acceptance Criteria

GPU Available
CUDA Enabled
YOLOv7 Patch Applied
Resume Training Verified
Validation After Every Stage
Final Report Generated

# YOLO_ChestXray - 初始基準實驗組 (Experiment 1: Baseline 640x640)

本實驗組為胸部 X 光病灶偵測的**初始基準實驗組 (Exp 1 Baseline)**。
採用 YOLOv7 官方標準架構與 COCO 預設超參數，輸入解析度為 `640 x 640`，並在嚴格的 5 階段 `--resume` 機制下進行 **150 輪 (Epochs)** 訓練，與高解析度驗證組 (Exp 2 1024x1024) 及醫學專項最佳化組 (Exp 3 Medical 1024x1024) 建立同輪次、標準化的科學對照基準線。

---

## 📌 實驗設計與超參數比較 (Exp1 vs Exp2 vs Exp3)

| 比較項目 | 初始基準組 (Exp 1 Baseline) | 驗證對照組 (Exp 2 Validation) | 醫學專項最佳化組 (Exp 3 Medical) | 效益與設計動機 |
| :--- | :--- | :--- | :--- | :--- |
| **輸入解析度 (Resolution)** | `640 x 640` | `1024 x 1024` | `1024 x 1024` | 標準基準尺寸，檢驗解析度提升之邊際效益 |
| **總訓練輪次 (Epochs)** | **150 輪 (30 x 5 階段)** | **150 輪 (30 x 5 階段)** | **150 輪 (30 x 5 階段)** | 三組嚴格統一總輪次，消除週期長短之干擾 |
| **階段訓練機制** | **標準 `--resume` 斷點接續** | **標準 `--resume` 斷點接續** | **標準 `--resume` 斷點接續** | 每個 Stage 精準跑 30 輪，學習率平滑衰減 |
| **批次大小 (Batch Size)** | `16` (訓練) / `32` (驗證) | `8` (訓練) / `16` (驗證) | `8` (訓練) / `16` (驗證) | 640 解析度具備更高運算效率與 GPU 記憶體裕度 |
| **CLAHE 對比增強** | ❌ 未啟用 | ❌ 未啟用 | ✅ 啟用 (`clipLimit=2.0`) | 評估傳統灰階輸入 vs 區域直方圖增強之差異 |
| **超參數設定檔** | `data/hyp.scratch.p5.yaml` | `data/hyp.scratch.p5.yaml` | `configs/hyp.medical.yaml` | COCO 標準基準超參數 (`lr0:0.01`, `lrf:0.1`) |
| **Google Drive 存放路徑** | `MyDrive/YOLO_ChestXray` | `MyDrive/YOLO_ChestXray_Exp2_1024` | `MyDrive/YOLO_ChestXray_Exp3_Medical` | 三組目錄完全獨立，權重、日誌與報告互不干擾 |

---

## 🚀 階段訓練進度表 (Stage Schedule)

| 階段 (Stage) | 起始輪次 | 目標輪次 | 本階段執行輪數 | 超參數設定檔 | 接續權重來源 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | Epoch 0 | Epoch 30 | 30 Epochs | `data/hyp.scratch.p5.yaml` | `yolov7.pt` (預訓練) |
| **Stage 2** | Epoch 30 | Epoch 60 | 30 Epochs | `data/hyp.scratch.p5.yaml` | `checkpoints/checkpoint_30.pt` (`--resume`) |
| **Stage 3** | Epoch 60 | Epoch 90 | 30 Epochs | `data/hyp.scratch.p5.yaml` | `checkpoints/checkpoint_60.pt` (`--resume`) |
| **Stage 4** | Epoch 90 | Epoch 120 | 30 Epochs | `data/hyp.scratch.p5.yaml` | `checkpoints/checkpoint_90.pt` (`--resume`) |
| **Stage 5** | Epoch 120 | Epoch 150 | 30 Epochs | `data/hyp.scratch.p5.yaml` | `checkpoints/checkpoint_120.pt` (`--resume`) |

---

## 💻 快速開始 (Colab 執行)

可以直接點擊下方按鈕在 Google Colab 中開啟執行：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wupinlai/YOLO_ChestXray/blob/main/notebooks/YOLO_ChestXray_Colab_Training.ipynb)

---

## 🛠️ 本地 / CLI 執行指令

```bash
# Stage 1: Epochs 1 ~ 30
python scripts/train.py --weights yolov7.pt --stage 1 --epochs 30 --img-size 640 640 --batch-size 16 --name stage_1

# Stage 2: Epochs 31 ~ 60
python scripts/train.py --resume checkpoints/checkpoint_30.pt --stage 2 --epochs 60 --img-size 640 640 --batch-size 16 --name stage_2

# Stage 3: Epochs 61 ~ 90
python scripts/train.py --resume checkpoints/checkpoint_60.pt --stage 3 --epochs 90 --img-size 640 640 --batch-size 16 --name stage_3

# Stage 4: Epochs 91 ~ 120
python scripts/train.py --resume checkpoints/checkpoint_90.pt --stage 4 --epochs 120 --img-size 640 640 --batch-size 16 --name stage_4

# Stage 5: Epochs 121 ~ 150
python scripts/train.py --resume checkpoints/checkpoint_120.pt --stage 5 --epochs 150 --img-size 640 640 --batch-size 16 --name stage_5

# 最佳模型全面驗證
python scripts/validate.py --conf-thres 0.05 --weights checkpoints/best_model.pt --data configs/chestxray.yaml --img-size 640 --batch-size 32 --name val_results

# 全流程後處理診斷分析與 16 頁 Master PDF 報告
python scripts/run_plan1_pipeline.py --conf-thres 0.05
```

---

## 📦 產出資產與報告規範 (Standard Deliverables)

1. **模型權重 (Checkpoints)**：
   - `checkpoints/best_model.pt`（驗證集 mAP 最佳權重）
   - `checkpoints/checkpoint_30.pt` ~ `checkpoint_150.pt`（各 Stage 接續權重）
2. **完整 16 頁診斷 PDF 報告**：
   - `reports/final_report/final_report.pdf`
3. **指標與視覺化圖表**：
   - `reports/metrics_history.csv`（150 輪實測完整指標）
   - `reports/learning_rate_curve.png` / `reports/gpu_usage.png`
   - `reports/top5_detections.jpg` / `reports/worst_iou_cases.jpg`
   - `reports/fp_gallery.jpg` / `reports/fn_gallery.jpg` / `reports/misclassification_gallery.jpg` / `reports/low_iou_gallery.jpg`
   - `reports/confidence_threshold_analysis.png`
4. **發布壓縮包 (GitHub Release)**：
   - `Release_Exp1_Baseline_640x640_150epochs.zip`

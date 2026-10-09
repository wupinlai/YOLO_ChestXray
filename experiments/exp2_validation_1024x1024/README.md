# YOLO_ChestXray - 驗證組 (Experiment 2: High-Resolution 1024x1024)

本實驗組為針對胸部 X 光微小病灶（如微小結節 Nodule、肺泡浸潤 Infiltration、細微纖維化 Fibrosis 等）之**高解析度驗證對照組**。
透過提升影像解析度至 `1024x1024`，並在嚴格的 `--resume` 機制下進行 **150 輪 (Epochs)** 訓練，與初始實驗組 (Exp1 640x640) 形成同輪次、不同解析度的科學對照。

---

## 📌 實驗設計與超參數比較 (Exp1 vs Exp2)

| 比較項目 | 初始實驗組 (Exp1 Baseline) | 驗證對照組 (Exp2 Validation) | 效益與設計動機 |
| :--- | :--- | :--- | :--- |
| **輸入解析度 (Resolution)** | `640 x 640` | **`1024 x 1024`** | 大幅保留微小病變結構細節，減少下採樣造成的像素遺失 |
| **總訓練輪次 (Epochs)** | 150 輪 | **150 輪 (30 x 5 階段)** | 嚴格控制訓練週期相同，確保對比之公平性 |
| **階段訓練機制** | 舊版疊加式訓練 | **標準 `--resume` 斷點接續** | 每個 Stage 精準跑 30 輪，學習率曲線平滑延續 |
| **訓練批次 (Batch Size)** | `16` | **`8`** (防顯存溢出) | 1024x1024 在 Colab T4/L4 GPU 上最穩定之批次設定 |
| **Google Drive 獨立存放** | `MyDrive/YOLO_ChestXray` | **`MyDrive/YOLO_ChestXray_Exp2_1024`** | 資料夾完全獨立，保證不覆蓋 Exp1 的原始權重與圖表 |

---

## 🚀 階段訓練進度表 (Stage Schedule)

| 階段 (Stage) | 起始輪次 | 目標輪次 | 本階段執行輪數 | 接續權重來源 |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | Epoch 0 | Epoch 30 | 30 Epochs | `yolov7.pt` (預訓練) |
| **Stage 2** | Epoch 30 | Epoch 60 | 30 Epochs | `checkpoints/checkpoint_30.pt` (`--resume`) |
| **Stage 3** | Epoch 60 | Epoch 90 | 30 Epochs | `checkpoints/checkpoint_60.pt` (`--resume`) |
| **Stage 4** | Epoch 90 | Epoch 120 | 30 Epochs | `checkpoints/checkpoint_90.pt` (`--resume`) |
| **Stage 5** | Epoch 120 | Epoch 150 | 30 Epochs | `checkpoints/checkpoint_120.pt` (`--resume`) |

---

## 💻 快速開始 (Colab 執行)

可以直接點擊下方按鈕或開啟 Notebook 執行：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wupinlai/YOLO_ChestXray/blob/main/notebooks/YOLO_ChestXray_Colab_Exp2_1024x1024.ipynb)

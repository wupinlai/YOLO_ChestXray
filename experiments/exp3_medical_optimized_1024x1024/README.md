# YOLO_ChestXray - 醫學領域專項最佳化對照組 (Experiment 3: Medical Optimized 1024x1024)

本實驗組為針對胸部 X 光（CXR）病灶判讀特徵設計的**第二個對照組 (Exp 3)**。
在解析度 `1024x1024` 與 `150 輪 (Epochs)` 的基礎上，全面導入針對醫學影像特性的超參數微調（Medical Hyperparameters）與自適應對比度增強（CLAHE），以解決微小結節（Nodule）、細微浸潤（Infiltration）與胸膜黏連等低對比難題。

---

## 📌 針對胸部 X 光病灶判讀之最佳參數建議與三組實驗對比

| 參數維度 | Exp 1: 初始實驗組 (Baseline) | Exp 2: 驗證組 (High-Res 1024) | Exp 3: 醫學專項最佳化組 (Medical Optimized) | 醫學影像判讀之核心動機與臨床意義 |
| :--- | :--- | :--- | :--- | :--- |
| **輸入解析度** | `640 x 640` | `1024 x 1024` | **`1024 x 1024`** | 高解析度大幅保留微小結節與細微紋理，避免下採樣失真 |
| **Anchor 機制** | AutoAnchor (COCO Default) | AutoAnchor (COCO Default) | **Medical AutoAnchor (k-means + GA)** | 針對 14 類胸腔病灶（如巨大 Cardiomegaly vs 微小 Nodule）自動重擬最佳長寬比例 Prior |
| **CLAHE 對比增強** | ❌ 未啟用 | ❌ 未啟用 | **✅ 啟用 (`clipLimit=2.0`, `tileGridSize=(8, 8)`)** | 消除肋骨與縱隔腔的高動態範圍壓迫，顯著提升肺泡組織與微小陰影之邊界可見度 |
| **初始學習率 (`lr0`)** | `0.01` | `0.01` | **`0.005` (溫和微調)** | 避免破壞預訓練卷積特徵抽取器，有利於醫學細微特徵之平穩收斂 |
| **最終學習率 (`lrf`)** | `0.1` ($lr_{final}=0.001$) | `0.1` ($lr_{final}=0.001$) | **`0.01` ($lr_{final}=0.00005$)** | 深度訓練後期給予更微細的權重更新，提升 mAP@0.5:0.95 邊界精準度 |
| **色彩增強 (`hsv_h/s/v`)**| `h:0.015, s:0.7, v:0.4` | `h:0.015, s:0.7, v:0.4` | **`h:0.0, s:0.0, v:0.2` (關閉色彩抖動)** | X 光片為單色灰階，關閉飽和度/色調抖動，僅保留真實亮度模擬 |
| **空間增強 (`flipud/mixup`)**| `flipud:0.0, mixup:0.0` | `flipud:0.0, mixup:0.0` | **`flipud:0.0, mixup:0.0, mosaic:0.5`** | 嚴格維持胸腔上方（肺尖）與下方（橫膈膜）解剖方位；避免 Mixup 產生虛假病灶 |
| **Google Drive 存放** | `MyDrive/YOLO_ChestXray` | `MyDrive/YOLO_ChestXray_Exp2_1024` | **`MyDrive/YOLO_ChestXray_Exp3_Medical`** | 獨立目錄存放，權重與報告完全隔離 |

---

## 🚀 階段訓練進度表 (Stage Schedule)

| 階段 (Stage) | 起始輪次 | 目標輪次 | 本階段執行輪數 | 超參數設定檔 | 接續權重來源 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | Epoch 0 | Epoch 30 | 30 Epochs | `configs/hyp.medical.yaml` | `yolov7.pt` (預訓練) |
| **Stage 2** | Epoch 30 | Epoch 60 | 30 Epochs | `configs/hyp.medical.yaml` | `checkpoints/checkpoint_30.pt` (`--resume`) |
| **Stage 3** | Epoch 60 | Epoch 90 | 30 Epochs | `configs/hyp.medical.yaml` | `checkpoints/checkpoint_60.pt` (`--resume`) |
| **Stage 4** | Epoch 90 | Epoch 120 | 30 Epochs | `configs/hyp.medical.yaml` | `checkpoints/checkpoint_90.pt` (`--resume`) |
| **Stage 5** | Epoch 120 | Epoch 150 | 30 Epochs | `configs/hyp.medical.yaml` | `checkpoints/checkpoint_120.pt` (`--resume`) |

---

## 💻 快速開始 (Colab 執行)

可以直接點擊下方按鈕在 Colab 開啟執行：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wupinlai/YOLO_ChestXray/blob/main/notebooks/YOLO_ChestXray_Colab_Exp3_Medical_Optimized.ipynb)

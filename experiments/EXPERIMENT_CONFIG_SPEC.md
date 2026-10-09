# YOLO_ChestXray: 三組實驗完整配置規範與對照手冊 (Experiment Configuration Specification)

本文件詳實記錄 **YOLO_ChestXray** 專案中三組科學對照實驗（Exp 1 初始實驗組、Exp 2 驗證對照組、Exp 3 醫學專項最佳化對照組）的所有超參數（Hyperparameters）、前處理流程（Preprocessing）、錨點配置（Anchors）、模型架構與階段排程設定。

---

## 📊 一、三組實驗核心參數全景對照矩陣 (Master Parameter Comparison Matrix)

| 參數類別 | 參數名稱 (YAML Key / Arg) | Exp 1: 初始實驗組 (Baseline) | Exp 2: 驗證對照組 (High-Res) | Exp 3: 醫學專項最佳化組 (Medical) | 醫學影像設計動機與臨床影響 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **輸入影像** | `img-size` (Pixels) | `640 x 640` | `1024 x 1024` | **`1024 x 1024`** | 高解析度減少特徵下採樣損失，保留微小結節與細微血管紋理 |
| **批次大小** | `batch-size` (Train/Val) | `16 / 32` | `8 / 16` | **`8 / 16`** | 1024 解析度下預防 GPU VRAM OOM，保障訓練穩定性 |
| **總訓練輪次** | Total Epochs | 150 輪 (累加式) | 150 輪 (5 階段 `--resume`) | **150 輪 (5 階段 `--resume`)** | 保持週期相同，消除輪次差異帶來的干擾 |
| **階段排程** | Stages / Epoch Schedule | 10 ➔ 20 ➔ 30 ➔ 40 ➔ 50 | 30 ➔ 60 ➔ 90 ➔ 120 ➔ 150 | **30 ➔ 60 ➔ 90 ➔ 120 ➔ 150** | 每個階段固定 30 輪，耗時均等，支援斷點接續 |
| **前處理增強** | CLAHE `clipLimit` | ❌ 未啟用 (None) | ❌ 未啟用 (None) | **`2.0`** | 限制局部直方圖對比度上限，避免過度放大肺泡高頻雜訊 |
| | CLAHE `tileGridSize` | ❌ 未啟用 (None) | ❌ 未啟用 (None) | **`(8, 8)`** | 局部 8x8 區塊均衡化，消除縱隔腔與骨骼之高動態壓迫 |
| **優化器** | Optimizer Type | SGD (`torch.optim.SGD`) | SGD (`torch.optim.SGD`) | **SGD (`torch.optim.SGD`)** | 具備更佳泛化性能與平滑損失地貌特徵 |
| | `lr0` (初始學習率) | `0.01` | `0.01` | **`0.005` (微調減半)** | 避免破壞預訓練卷積特徵，促進醫學病灶細微收斂 |
| | `lrf` (最終學習率倍率) | `0.1` ($lr_{end}=0.001$) | `0.1` ($lr_{end}=0.001$) | **`0.01` ($lr_{end}=0.00005$)** | 深度訓練後期給予更微小的權重微調，提升邊界 IoU 精確度 |
| | `momentum` | `0.937` | `0.937` | **`0.937`** | SGD 慣性動量因子 |
| | `weight_decay` | `0.0005` | `0.0005` | **`0.0005`** | L2 正則化權重衰減 |
| | `warmup_epochs` | `3.0` | `3.0` | **`3.0`** | 前 3 輪線性爬升至初始學習率 |
| | `warmup_momentum` | `0.8` | `0.8` | **`0.8`** | 熱身階段初始動量 |
| | `warmup_bias_lr` | `0.1` | `0.1` | **`0.1`** | 偏置項熱身學習率 |
| **損失函數權重**| `box` (Box Loss Gain) | `0.05` | `0.05` | **`0.05`** | 邊界框 CIoU 損失權重 |
| | `cls` (Cls Loss Gain) | `0.5` | `0.5` | **`0.5`** | 分類 BCE 損失權重 |
| | `cls_pw` (Positive Weight) | `1.0` | `1.0` | **`1.0`** | 分類正樣本損失加權係數 |
| | `obj` (Obj Loss Gain) | `1.0` | `1.0` | **`1.0`** | 物件置信度損失權重 |
| | `obj_pw` (Positive Weight) | `1.0` | `1.0` | **`1.0`** | 物件正樣本損失加權係數 |
| | `iou_t` (IoU Threshold) | `0.20` | `0.20` | **`0.20`** | 標籤分配 IoU 門檻值 |
| | `anchor_t` (Anchor Multiplier)| `4.0` | `4.0` | **`4.0`** | 錨點長寬比分配寬容度 |
| | `fl_gamma` (Focal Loss) | `0.0` | `0.0` | **`0.0`** | Focal loss gamma |
| **色彩資料增強**| `hsv_h` (Hue Shift) | `0.015` | `0.015` | **`0.0` (禁用)** | X 光為灰階成像，禁止非生理性色彩色相偏移 |
| | `hsv_s` (Saturation Shift) | `0.7` | `0.7` | **`0.0` (禁用)** | 灰階單色影像無飽和度，禁止飽和度抖動 |
| | `hsv_v` (Value/Brightness) | `0.4` | `0.4` | **`0.2` (微幅亮度)** | 模擬臨床射線管曝光條件（kVp/mAs）之明暗差異 |
| **幾何資料增強**| `degrees` (Rotation) | `0.0` | `0.0` | **`3.0` (±3°)** | 模擬患者站立或臥位攝影時之微幅體位傾斜 |
| | `translate` (Translation) | `0.1` | `0.1` | **`0.05` (±5%)** | 微幅平移增強 |
| | `scale` (Scale Gain) | `0.5` | `0.5` | **`0.1` (±10%)** | 保持解剖器官比例，防止過度縮放失真 |
| | `shear` (Shearing) | `0.0` | `0.0` | **`0.0` (禁用)** | 禁止剪切形變，避免扭曲胸廓骨骼結構 |
| | `perspective` | `0.0` | `0.0` | **`0.0` (禁用)** | 禁止透視變換 |
| | `flipud` (Vertical Flip) | `0.0` (禁用) | `0.0` (禁用) | **`0.0` (禁用)** | 胸腔具有絕對上下解剖方向（肺尖在上、膈肌在下） |
| | `fliplr` (Horizontal Flip) | `0.5` | `0.5` | **`0.5`** | 水平鏡像翻轉（左右肺對稱性） |
| | `mosaic` (Mosaic Aug) | `1.0` | `1.0` | **`0.5` (適度)** | 避免 4 圖拼接過密導致肺野結構過度凌亂 |
| | `mixup` (Mixup Aug) | `0.0` (禁用) | `0.0` (禁用) | **`0.0` (禁用)** | 混合影像會創造重疊虛假病灶，醫學上嚴格禁止 |
| | `copy_paste` | `0.0` (禁用) | `0.0` (禁用) | **`0.0` (禁用)** | 禁用隨機貼上 |
| **錨點 (Anchors)**| Anchor Priority | P3, P4, P5 (9 組) | P3, P4, P5 (9 組) | **P3, P4, P5 (9 組)** | 3 尺度特徵圖輸出 (8x, 16x, 32x 下採樣) |
| | AutoAnchor 機制 | ✅ 自動計算 (BPR<0.98) | ✅ 自動計算 (BPR<0.98) | **✅ 自動計算 (k-means + GA 基因演算法)** | 依 14 類標註框自動演化最適 Anchor |
| **雲端儲存路徑**| Google Drive Directory | `MyDrive/YOLO_ChestXray` | `MyDrive/YOLO_ChestXray_Exp2_1024` | **`MyDrive/YOLO_ChestXray_Exp3_Medical`** | 三組資料夾完全獨立隔離，互不覆蓋 |

---

## 🔬 二、各實驗組詳細設定與執行指令

### 1. 初始實驗組 (Exp 1: Baseline 640x640)
* **專屬目錄**：[`experiments/exp1_baseline_640x640/`](./exp1_baseline_640x640/)
* **超參數設定檔**：`data/hyp.scratch.p5.yaml`
* **模型結構定義檔**：`cfg/training/yolov7.yaml`
* **資料集定義檔**：`configs/chestxray.yaml`
* **Google Drive 路徑**：`/content/drive/MyDrive/YOLO_ChestXray`
* **執行命令**：
  ```bash
  python scripts/train.py --weights yolov7.pt --hyp data/hyp.scratch.p5.yaml --epochs 50 --batch-size 16 --img-size 640 640
  ```

---

### 2. 驗證對照組 (Exp 2: High-Resolution 1024x1024)
* **專屬目錄**：[`experiments/exp2_validation_1024x1024/`](./exp2_validation_1024x1024/)
* **超參數設定檔**：`data/hyp.scratch.p5.yaml`
* **解析度提升**：`--img-size 1024 1024`，批次縮減 `--batch-size 8`
* **Google Drive 路徑**：`/content/drive/MyDrive/YOLO_ChestXray_Exp2_1024`
* **階段排程命令**：
  ```bash
  # Stage 1 (0 ~ 30 Epochs)
  python scripts/train.py --weights yolov7.pt --stage 1 --epochs 30 --img-size 1024 1024 --batch-size 8 --name stage_1
  # Stage 2 (30 ~ 60 Epochs)
  python scripts/train.py --resume checkpoints/checkpoint_30.pt --stage 2 --epochs 60 --img-size 1024 1024 --batch-size 8 --name stage_2
  # Stage 3 (60 ~ 90 Epochs)
  python scripts/train.py --resume checkpoints/checkpoint_60.pt --stage 3 --epochs 90 --img-size 1024 1024 --batch-size 8 --name stage_3
  # Stage 4 (90 ~ 120 Epochs)
  python scripts/train.py --resume checkpoints/checkpoint_90.pt --stage 4 --epochs 120 --img-size 1024 1024 --batch-size 8 --name stage_4
  # Stage 5 (120 ~ 150 Epochs)
  python scripts/train.py --resume checkpoints/checkpoint_120.pt --stage 5 --epochs 150 --img-size 1024 1024 --batch-size 8 --name stage_5
  ```

---

### 3. 醫學專項最佳化對照組 (Exp 3: Medical Optimized 1024x1024)
* **專屬目錄**：[`experiments/exp3_medical_optimized_1024x1024/`](./exp3_medical_optimized_1024x1024/)
* **超參數設定檔**：[`configs/hyp.medical.yaml`](../configs/hyp.medical.yaml)
* **前處理增強**：CLAHE `clipLimit=2.0`, `tileGridSize=(8, 8)`
  ```bash
  python scripts/apply_clahe.py --src-dir datasets/chestxray8/train/images --dst-dir datasets/chestxray8/train/images --clip-limit 2.0 --tile-grid-size 8 8
  python scripts/apply_clahe.py --src-dir datasets/chestxray8/val/images --dst-dir datasets/chestxray8/val/images --clip-limit 2.0 --tile-grid-size 8 8
  ```
* **Google Drive 路徑**：`/content/drive/MyDrive/YOLO_ChestXray_Exp3_Medical`
* **階段排程命令**：
  ```bash
  # Stage 1 (0 ~ 30 Epochs)
  python scripts/train.py --weights yolov7.pt --hyp configs/hyp.medical.yaml --stage 1 --epochs 30 --img-size 1024 1024 --batch-size 8 --name stage_1
  # Stage 2 (30 ~ 60 Epochs)
  python scripts/train.py --resume checkpoints/checkpoint_30.pt --hyp configs/hyp.medical.yaml --stage 2 --epochs 60 --img-size 1024 1024 --batch-size 8 --name stage_2
  # Stage 3 (60 ~ 90 Epochs)
  python scripts/train.py --resume checkpoints/checkpoint_60.pt --hyp configs/hyp.medical.yaml --stage 3 --epochs 90 --img-size 1024 1024 --batch-size 8 --name stage_3
  # Stage 4 (90 ~ 120 Epochs)
  python scripts/train.py --resume checkpoints/checkpoint_90.pt --hyp configs/hyp.medical.yaml --stage 4 --epochs 120 --img-size 1024 1024 --batch-size 8 --name stage_4
  # Stage 5 (120 ~ 150 Epochs)
  python scripts/train.py --resume checkpoints/checkpoint_120.pt --hyp configs/hyp.medical.yaml --stage 5 --epochs 150 --img-size 1024 1024 --batch-size 8 --name stage_5
  ```

---

## 📦 三、各實驗組產出資產與 Checkpoint 規範

每個實驗組別執行完畢後均會產生以下標準化資產：
1. **模型權重檔**：
   - `checkpoints/best_model.pt`（驗證集 mAP 最佳模型）
   - `checkpoints/checkpoint_30.pt` ~ `checkpoint_150.pt`（各 Stage 斷點權重）
2. **診斷報告與圖表**：
   - `reports/final_report/final_report.pdf`（300 DPI 16 頁診斷分析 Master PDF 報告）
   - `reports/confusion_matrix.png`、`reports/PR_curve.png`、`reports/F1_curve.png`
   - `reports/metrics_history.csv`（完整 150 輪訓練與驗證指標歷史紀錄）
   - `reports/top5_detections.jpg`、`reports/worst_iou_cases.jpg`、`reports/fp_gallery.jpg`、`reports/fn_gallery.jpg`
3. **發布壓縮包 (GitHub Release Assets)**：
   - Exp 1: `Release_50_Plan1_v4.0_assets.zip`
   - Exp 2: `Release_Exp2_1024x1024_150epochs.zip`
   - Exp 3: `Release_Exp3_Medical_1024x1024_150epochs.zip`

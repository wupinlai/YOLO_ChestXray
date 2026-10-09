# 📝 YOLO_ChestXray 專案重要修改與版本演進紀錄 (Changelog)

本文檔詳實記錄本次針對胸部 X 光（CXR）病灶檢測專案所進行之核心架構修復、三組科學實驗規劃、超參數最佳化與雲端儲存隔離機制之變更歷程。

---

## 📅 版本紀錄總覽

| Commit Hash | 模組 / 範疇 | 修改主題與摘要 |
| :--- | :--- | :--- |
| `e3fa335` | `scripts/train.py` | 修正 YOLOv7 階段訓練斷點接續機制（切換為 `--resume` 避免從 0 輪重複起跑） |
| `05326b5` | `experiments/exp2` | 建立 Exp 2 驗證對照組（1024x1024 高解析度、150 輪訓練、獨立 Notebook） |
| `418a41f` | `configs/`, `exp3` | 建立 Exp 3 醫學專項最佳化組（新增 `hyp.medical.yaml`、CLAHE 直方圖增強器） |
| `b22b13d` | `experiments/` | 產出 `EXPERIMENT_CONFIG_SPEC.md` 三組實驗完整超參數規範與對照表 |
| `69961c4` | `scripts/train.py` | 修正 Stage 快取搜尋機制，新增 `--drive-dir` 參數杜絕跨實驗跳過訓練問題 |
| `54e4a15` | `scripts/validate.py` | 驗證模組全面適配 `--drive-dir`，保證 Google Drive 權重與報告 100% 獨立隔離 |

---

## 🔍 詳細修改內容與技術細節

### 1. 斷點接續與訓練輪次修復 (`scripts/train.py`)
* **修改前問題**：
  * 原先在階段訓練中使用 `--weights checkpoints/stageX.pt --epochs Y`，導致 YOLOv7 將權重當作全新遷移學習起點，每次從 `Epoch 0` 重新計算至指定的目標輪次（累積計算達到 $10+20+30+40+50=150$ 次，且階段耗時呈 1x, 2x, 3x, 4x, 5x 遞增）。
* **修改後實作**：
  * 當偵測到接續權重時，自動切換為官方標準指令 `python train.py --resume {weights_path} --epochs {opt.epochs}`。
  * 啟用 YOLOv7 內部的 `start_epoch = ckpt['epoch'] + 1`，使每個階段精準執行固定輪數，且學習率退火曲線平滑接續。

---

### 2. 初始實驗組保存與對照組建立 (`experiments/`)
* **Exp 1 (初始實驗組 / Baseline)**：
  * 完整保留原始程式碼、權重與產出報告（640x640 解析度，150 輪累積訓練）。
  * 存放於 `experiments/exp1_baseline_640x640/`。
* **Exp 2 (高解析度驗證對照組 / High-Res Validation)**：
  * 輸入解析度提升至 `1024 x 1024`（大幅強化微小結節、浸潤等病灶特徵）。
  * 總輪次設定為 150 輪（5 階段 x 每階段 30 輪：`30 ➔ 60 ➔ 90 ➔ 120 ➔ 150`）。
  * 訓練批次調整為 `batch-size 8`，適配 Colab T4/L4 GPU 顯存防 OOM。
  * 專屬 Notebook：[`notebooks/YOLO_ChestXray_Colab_Exp2_1024x1024.ipynb`](../notebooks/YOLO_ChestXray_Colab_Exp2_1024x1024.ipynb)。
* **Exp 3 (醫學專項最佳化對照組 / Medical Domain Optimized)**：
  * 在 1024 解析度與 150 輪基礎上，導入針對胸部 X 光特性的醫學領域最佳化配置。
  * 專屬超參數：[`configs/hyp.medical.yaml`](../configs/hyp.medical.yaml)。
  * 自適應對比度增強：[`scripts/apply_clahe.py`](../scripts/apply_clahe.py) (`clipLimit=2.0`, `tileGridSize=(8, 8)`)。
  * 專屬 Notebook：[`notebooks/YOLO_ChestXray_Colab_Exp3_Medical_Optimized.ipynb`](../notebooks/YOLO_ChestXray_Colab_Exp3_Medical_Optimized.ipynb)。

---

### 3. 醫學領域專項超參數調優 (`configs/hyp.medical.yaml`)
針對胸部 X 光物理成像特點，調整下列關鍵參數：
* **初始學習率 (`lr0`)**：由 `0.01` 降至 `0.005`，進行溫和遷移學習，保護預訓練卷積特徵。
* **最終學習率 (`lrf`)**：由 `0.1` 降至 `0.01`（最終學習率達 $0.00005$），提升後期邊界框 IoU 精確度。
* **色彩抖動 (`hsv_h: 0.0, hsv_s: 0.0`)**：針對單色灰階 X 光停用色調與飽和度抖動，避免產生偽影。
* **解剖方位約束 (`flipud: 0.0, mixup: 0.0`)**：嚴格鎖定上下方位（肺尖在上、膈肌在下）；禁用 Mixup 避免創造重疊假病灶。

---

### 4. Google Drive 雲端儲存與快取搜尋嚴格隔離 (`scripts/train.py`, `scripts/validate.py`)
* **修改前問題**：
  * 全局快取檢查寫死搜尋 `/content/drive/MyDrive/YOLO_ChestXray`，導致 Exp 2 Stage 1 (目標 30 輪) 誤讀 Exp 1 的 `checkpoint_30.pt` 而觸發略過。
* **修改後實作**：
  * 在 `train.py` 與 `validate.py` 新增 `--drive-dir` 參數。
  * 各實驗僅與自己的 Google Drive 目錄互動，三組路徑完全獨立：
    - Exp 1: `/content/drive/MyDrive/YOLO_ChestXray`
    - Exp 2: `/content/drive/MyDrive/YOLO_ChestXray_Exp2_1024`
    - Exp 3: `/content/drive/MyDrive/YOLO_ChestXray_Exp3_Medical`
  * 訓練前自動清理本地殘留，保證新實驗從 Epoch 0 乾淨起跑。

---

### 5. 完整文檔與配置規範
* 新增 [`experiments/EXPERIMENT_CONFIG_SPEC.md`](../experiments/EXPERIMENT_CONFIG_SPEC.md) 詳實記錄三組實驗的 25+ 項超參數、損失權重、資料增強與階段指令。
* 更新專案首頁 [`README.md`](../README.md)，提供三組實驗的一鍵 Colab 啟動徽章與對照摘要。

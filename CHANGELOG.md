# 📝 YOLO_ChestXray 專案重要修改與版本演進紀錄 (Changelog)

本文檔詳實記錄本次針對胸部 X 光（CXR）病灶檢測專案所進行之核心架構修復、三組科學實驗規劃、超參數最佳化與雲端儲存隔離機制之變更歷程。

---

## 📅 版本紀錄總覽

| Commit Hash | 模組 / 範疇 | 修改主題與摘要 |
| :--- | :--- | :--- |
| `564893c` | `scripts/train.py` | 鎖定每階段固定 30 輪訓練預算，並加入 ANSI 游標清行原位更新（解決終端機洗版與輪數膨脹） |
| `dc90c82` | `train.py` | 在專案根目錄維護原生相容 PyTorch 2.6 的 `train.py`，徹底修復 `torch.load` 遞迴覆蓋崩潰 |

---

## 🔍 詳細修改內容與技術細節

### 1. Stage 2 跨階段銜接與架構重大修復（昨日至今日）
在 Exp 2 高解析度（1024x1024）環境下，執行 Stage 2（載入 Stage 1 的 `checkpoint_30.pt` 繼續訓練）時觸發了一系列底層相容性與流程問題，已全數徹底修復：

#### A. PyTorch 2.6 與 `torch.load` 無窮遞迴崩潰修復
* **問題現象**：
  * Colab 升級 PyTorch 2.6 後預設 `weights_only=True` 導致自定義模型載入失敗。
  * 多模組各自執行 `_orig_torch_load = torch.load`，在跨模組互相調用時引發自我包裝的無窮遞迴死循環（`RecursionError: maximum recursion depth exceeded (988 calls)`），導致 Stage 2 數秒內閃退。
* **解決方案**：
  * 引入**全域單例指針機制**（`torch._orig_torch_load_unwrapped`），確保原始 `torch.load` 指針僅被捕獲一次，徹底消除遞迴陷阱。
  * 註冊 `numpy._core.multiarray._reconstruct` 至 PyTorch `add_safe_globals` 安全名單。
  * 在倉庫根目錄直接維護已原生修復好的 [`train.py`](file:///d:/AI%20Project/研究方法/YOLO_ChestXray/train.py)，並在 Colab Step 3 加入 `!git checkout train.py`，避免被官方舊程式碼覆蓋。

#### B. 訓練輪數膨脹控制（精準鎖定每 Stage 執行 30 輪）
* **問題現象**：
  * Stage 2 傳入 `--epochs 60` 且使用微調權重載入時，YOLO 將其解讀為「從 0 跑滿 60 個新輪次」，導致 Stage 2 耗時加倍（80 分鐘），且 5 階段總輪數會膨脹至 450 輪。
* **解決方案**：
  * 在 [`scripts/train.py`](file:///d:/AI%20Project/研究方法/YOLO_ChestXray/scripts/train.py) 中重構輪數計算邏輯，將底層傳給 `train.py` 的輪數固定為 **30 個 Epochs**（顯示 `0/29` ~ `29/29`）。
  * 產出時自動對應累計目標（Stage 2 存為 `checkpoint_60.pt`，Stage 3 存為 `checkpoint_90.pt`...），5 個 Stage 剛好精準達成 150 輪總訓練量。

#### C. 終端機即時輸出與進度條原位覆蓋（解決洗版問題）
* **問題現象**：
  * 原先使用 `subprocess.run` 造成日誌緩衝延遲；改用串流後，因非 TTY 管道使 tqdm 的 `\r` 變成換行，每個 Epoch 洗出 79 行輸出。
* **解決方案**：
  * 使用 `subprocess.Popen(..., bufsize=1)` 實現逐行無延遲串流。
  * 導入 ANSI 終端清行與游標歸位指令（`\r\033[K`），訓練 Batch 與驗證進度會在**單一行原地動態更新**，只有在 Epoch 完成輸出統計表格時才整齊換行，輸出極致清晰。

---

### 2. 斷點接續與訓練輪次修復 (`scripts/train.py`)
* **修改前問題**：
  * 原先在階段訓練中使用 `--weights checkpoints/stageX.pt --epochs Y`，導致 YOLOv7 將權重當作全新遷移學習起點，每次從 `Epoch 0` 重新計算至指定的目標輪次。
* **修改後實作**：
  * 當偵測到接續權重時，自動切換為官方標準指令 `python train.py --resume {weights_path} --epochs {opt.epochs}`。
  * 啟用 YOLOv7 內部的 `start_epoch = ckpt['epoch'] + 1`，使每個階段精準執行固定輪數，且學習率退火曲線平滑接續。

---

### 3. 初始實驗組保存與對照組建立 (`experiments/`)
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

### 4. 醫學領域專項超參數調優 (`configs/hyp.medical.yaml`)
針對胸部 X 光物理成像特點，調整下列關鍵參數：
* **初始學習率 (`lr0`)**：由 `0.01` 降至 `0.005`，進行溫和遷移學習，保護預訓練卷積特徵。
* **最終學習率 (`lrf`)**：由 `0.1` 降至 `0.01`（最終學習率達 $0.00005$），提升後期邊界框 IoU 精確度。
* **色彩抖動 (`hsv_h: 0.0, hsv_s: 0.0`)**：針對單色灰階 X 光停用色調與飽和度抖動，避免產生偽影。
* **解剖方位約束 (`flipud: 0.0, mixup: 0.0`)**：嚴格鎖定上下方位（肺尖在上、膈肌在下）；禁用 Mixup 避免創造重疊假病灶。

---

### 5. Google Drive 雲端儲存與快取搜尋嚴格隔離 (`scripts/train.py`, `scripts/validate.py`)
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

### 6. 完整文檔與配置規範
* 新增 [`experiments/EXPERIMENT_CONFIG_SPEC.md`](../experiments/EXPERIMENT_CONFIG_SPEC.md) 詳實記錄三組實驗的 25+ 項超參數、損失權重、資料增強與階段指令。
* 更新專案首頁 [`README.md`](../README.md)，提供三組實驗的一鍵 Colab 啟動徽章與對照摘要。

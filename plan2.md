# YOLO_ChestXray: Google Colab 階段式接續訓練與執行工作流 (Plan 2)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wupinlai/YOLO_ChestXray/blob/main/notebooks/YOLO_ChestXray_Colab_Training.ipynb)

本文件定義在 **Google Colab (GPU 環境)** 上進行 ChestXray8 YOLOv7 模型訓練、斷點接續、Google Drive 自動備份、指標驗證與 GitHub Release 發布的標準作業流程 (Standard Operating Procedure, SOP)。

---

## 🔗 一鍵開啟 Google Colab 執行環境

點擊下方按鈕或連結，即可直接於 Google Colab 載入訓練 Notebook：

👉 **[開啟 Google Colab 訓練 Notebook (YOLO_ChestXray_Colab_Training.ipynb)](https://colab.research.google.com/github/wupinlai/YOLO_ChestXray/blob/main/notebooks/YOLO_ChestXray_Colab_Training.ipynb)**

---

## 📋 完整工作流程 (Execution Workflow)

```text
[本地 Antigravity IDE] ➔ [GitHub 儲存庫]
                               ↓
                   [Google Colab (T4/L4 GPU)]
                               ↓
                   [掛載 Google Drive 持久化備份]
                               ↓
                   [下載 Kaggle ChestXray8 資料集]
                               ↓
                   ┌─────────────────────────────┐
                   │   Stage 1: Epoch 1 ~ 10     │ ➔ 存至 checkpoint_10.pt & Drive
                   └──────────────┬──────────────┘
                                  ↓
                   ┌─────────────────────────────┐
                   │   Stage 2: Epoch 11 ~ 20    │ ➔ 存至 checkpoint_20.pt & Drive (斷點接續)
                   └──────────────┬──────────────┘
                                  ↓
                   ┌─────────────────────────────┐
                   │   Stage 3: Epoch 21 ~ 30    │ ➔ 存至 checkpoint_30.pt & Drive
                   └──────────────┬──────────────┘
                                  ↓
                   ┌─────────────────────────────┐
                   │   Stage 4: Epoch 31 ~ 40    │ ➔ 存至 checkpoint_40.pt & Drive
                   └──────────────┬──────────────┘
                                  ↓
                   ┌─────────────────────────────┐
                   │   Stage 5: Epoch 41 ~ 50    │ ➔ 存至 checkpoint_50.pt & Drive
                   └──────────────┬──────────────┘
                                  ↓
                   [模型驗證 (test.py) & 指標評估]
                                  ↓
                   [X 光影像病灶推論 (detect.py)]
                                  ↓
                   [打包 Release 資產並發布 GitHub Release]
```

---

## 🛠️ 詳細步驟指引 (Step-by-Step Guide)

### 步驟 1：檢查 GPU 執行階段
1. 進入 Colab 後，點選上方選單 **「執行階段 (Runtime)」** ➔ **「變更執行階段類型 (Change runtime type)」**。
2. 將硬體加速器選為 **GPU**（例如 T4 GPU）。
3. 執行第 1 儲存格確認 GPU 正常啟用：
   ```bash
   !nvidia-smi
   ```

---

### 步驟 2：掛載 Google Drive（實現雲端權重持久化）
執行第 2 儲存格以掛載 Google 雲端硬碟，確保訓練過程即便遭遇 Colab 斷線，所有權重與報表都不會遺失：
- **備份目錄**：`MyDrive/YOLO_ChestXray/`
  - `checkpoints/`：存放各階段 `checkpoint_xx.pt`、`best.pt` 與 `last.pt`
  - `results/`：存放驗證報表與繪圖（`results.png`, `confusion_matrix.png` 等）
  - `releases/`：存放已打包的 Release 壓縮檔

---

### 步驟 3：安裝環境與下載 ChestXray8 資料集
1. **複製 YOLOv7 與依賴**：自動下載官方預訓練權重 `yolov7.pt`。
2. **Kaggle API 整合**：
   - 上傳 `kaggle.json`（若雲端硬碟已存有則自動載入）。
   - 自動下載並解壓縮 [YOLO Annotated ChestXray-8 Dataset](https://www.kaggle.com/datasets/spritan1/yolo-annotated-chestxray-8-object-detection/data)。
3. **配置檔確認**：自動產生 `configs/chestxray.yaml`，對應 14 種胸腔病灶類別。

---

### 步驟 4：分階段訓練與斷點接續 (Resumable Training)

| 階段 | Epoch 區間 | 載入基礎權重 | 產出 Checkpoint | 雲端同步檔案 |
| :--- | :---: | :--- | :--- | :--- |
| **Stage 1** | 1 ~ 10 | `yolov7.pt` | `runs/train/stage_1/weights/last.pt` | `MyDrive/.../checkpoints/checkpoint_10.pt` |
| **Stage 2** | 11 ~ 20 | `checkpoint_10.pt` | `runs/train/stage_2/weights/last.pt` | `MyDrive/.../checkpoints/checkpoint_20.pt` |
| **Stage 3** | 21 ~ 30 | `checkpoint_20.pt` | `runs/train/stage_3/weights/last.pt` | `MyDrive/.../checkpoints/checkpoint_30.pt` |
| **Stage 4** | 31 ~ 40 | `checkpoint_30.pt` | `runs/train/stage_4/weights/last.pt` | `MyDrive/.../checkpoints/checkpoint_40.pt` |
| **Stage 5** | 41 ~ 50 | `checkpoint_40.pt` | `runs/train/stage_5/weights/last.pt` | `MyDrive/.../checkpoints/checkpoint_50.pt` |

#### 🚨 意外斷線緊急恢復指令：
若訓練中途發生 Colab 斷線，重啟後只需執行：
```bash
!python train.py --resume /content/drive/MyDrive/YOLO_ChestXray/checkpoints/last.pt
```

---

### 步驟 5：模型驗證與指標評估 (Validation)
在各階段訓練完成後，執行驗證單元格評估病灶偵測效果：
```bash
!python test.py \
  --data configs/chestxray.yaml \
  --weights /content/drive/MyDrive/YOLO_ChestXray/checkpoints/best.pt \
  --batch-size 32 \
  --img 640 \
  --conf 0.001 \
  --iou 0.65 \
  --project runs/test \
  --name val_results \
  --verbose
```
- **評估指標**：Precision, Recall, mAP@0.5, mAP@0.5:0.95
- **視覺化圖表**：`confusion_matrix.png`, `PR_curve.png`, `F1_curve.png`, `results.png`

---

### 步驟 6：推論展示 (Inference Visualization)
使用訓練好的 `best.pt` 針對驗證集 X 光片進行推論：
```bash
!python detect.py \
  --weights /content/drive/MyDrive/YOLO_ChestXray/checkpoints/best.pt \
  --source datasets/chestxray8/val/images \
  --conf 0.25 \
  --img-size 640 \
  --project runs/detect \
  --name exp_val
```
Notebook 將直接在輸出區域渲染標註有病灶類別與信心度（Confidence Score）的胸腔 X 光影像。

---

### 步驟 7：打包與發布 GitHub Release
執行第 9 儲存格將當前階段的成果打包：
```bash
!zip -j /content/drive/MyDrive/YOLO_ChestXray/releases/Release_20_assets.zip \
  /content/drive/MyDrive/YOLO_ChestXray/checkpoints/checkpoint_20.pt \
  /content/drive/MyDrive/YOLO_ChestXray/checkpoints/best.pt \
  runs/test/val_results/*.png \
  runs/test/val_results/*.csv
```
打包完成後，可至 [GitHub Releases 頁面](https://github.com/wupinlai/YOLO_ChestXray/releases) 建立新標籤（如 `Release_20`）並上傳此 zip 或權重檔。

---

## 🎯 早停決策規則 (Early Stopping Criteria)

每 10 個 Epoch 檢視一次驗證指標 `mAP@0.5`：
- 若指標持續提升（如 `0.32` ➔ `0.48` ➔ `0.59`）➔ 繼續執行下一階段。
- 若連續兩階段指標持平或下降（如 `0.61` ➔ `0.61`）➔ 判定模型收斂，終止訓練並以 `best.pt` 作為最終成果。

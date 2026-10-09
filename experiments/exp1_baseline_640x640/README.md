# YOLO_ChestXray - 初始實驗組 (Experiment 1: Baseline 640x640)

本實驗組為初始基準實驗組（Baseline Group），保留所有原始訓練設定、模型權重評估與產出報告。

---

## 📌 實驗配置與超參數 (Hyperparameters)

| 項目 | 實驗設定 | 說明 |
| :--- | :--- | :--- |
| **實驗名稱** | `Exp1_Baseline_640x640` | 初始基準實驗組 |
| **輸入解析度 (Resolution)** | `640 x 640` | 標準 YOLOv7 輸入尺寸 |
| **總訓練輪次 (Epochs)** | 150 次（累積計算） | 5 階段累計訓練 ($10+20+30+40+50=150$) |
| **批次大小 (Batch Size)** | `16` (訓練) / `32` (驗證) | 適合 640x640 之 GPU 顯存負載 |
| **Google Drive 存放路徑** | `/content/drive/MyDrive/YOLO_ChestXray` | 保留原始資料夾與 Checkpoints |
| **主要權重檔案** | `checkpoints/best_model.pt` | 150 輪訓練後最佳權重 |
| **主報告文件** | `reports/final_report/final_report.pdf` | 16 頁完整診斷分析報告 |

---

## 📂 相關產出與報告清單

1. **Notebook 腳本**: [`notebooks/YOLO_ChestXray_Colab_Training.ipynb`](../../notebooks/YOLO_ChestXray_Colab_Training.ipynb)
2. **完整 16 頁診斷 PDF 報告**: [`reports/final_report/final_report.pdf`](../../reports/final_report/final_report.pdf)
3. **驗證與診斷數據庫**:
   - `reports/metrics_history.csv`
   - `reports/dataset_summary.csv`
   - `reports/dataset_integrity_report.csv`
   - `reports/top5_detections.jpg` / `reports/worst_iou_cases.jpg`
   - `reports/fp_gallery.jpg` / `reports/fn_gallery.jpg` / `reports/misclassification_gallery.jpg` / `reports/low_iou_gallery.jpg`

"""
YOLO_ChestXray Full Plan 1-2 Pipeline Orchestrator
Executes:
1. Dataset Integrity Verification (reports/dataset_integrity_report.csv)
2. Dataset Statistical Analysis (reports/dataset_analysis/)
3. Progression Metrics & Comparison Visualizations
4. Confidence Threshold Sweep & Root Cause Diagnostics
5. Automated Error Annotation generation
6. Error Galleries (2x2 Grid)
7. Final Multi-Sample Inference verification (sample_01 ~ 10, result_01 ~ 10)
8. 16-Page Master PDF Report Compilation (reports/final_report/final_report.pdf)
9. Project Manifest Generation (reports/project_manifest.json)
"""

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path
import pandas as pd
import yaml

from scripts.dataset_utils import check_dataset_integrity, run_dataset_analysis
from scripts.analysis import (
    CLASS_NAMES_DEFAULT,
    annotate_and_save_case,
    generate_comparison_plots,
    generate_error_galleries,
    generate_project_manifest,
    run_full_statistical_analysis,
    xywh2xyxy,
)
from scripts.generate_report import generate_final_report_pdf
from scripts.inference import run_multi_sample_inference, select_10_random_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Plan 1-2 Full Pipeline Execution")
    parser.add_argument("--data", default="configs/chestxray.yaml", help="dataset yaml path")
    parser.add_argument("--train-img-dir", default="datasets/chestxray8/train/images", help="train image path")
    parser.add_argument("--train-label-dir", default="datasets/chestxray8/train/labels", help="train label path")
    parser.add_argument("--val-img-dir", default="datasets/chestxray8/val/images", help="val image path")
    parser.add_argument("--val-label-dir", default="datasets/chestxray8/val/labels", help="val label path")
    parser.add_argument("--checkpoints-dir", default="checkpoints", help="checkpoints folder")
    parser.add_argument("--reports-dir", default="reports", help="reports output folder")
    parser.add_argument("--best-model", default="checkpoints/best_model.pt", help="best model weights")
    return parser.parse_args()


def load_ground_truths(label_dir: str, img_dir: str, class_names: list) -> list:
    """Load ground truth annotations."""
    gts = []
    if not os.path.exists(label_dir):
        return gts
    for label_file in Path(label_dir).glob("*.txt"):
        img_name = f"{label_file.stem}.jpg"
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            img_name = f"{label_file.stem}.png"
            img_path = os.path.join(img_dir, img_name)

        img_w, img_h = 1024, 1024
        if os.path.exists(img_path):
            try:
                from PIL import Image
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception:
                pass

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    bbox = xywh2xyxy([xc, yc, w, h], img_w, img_h)
                    cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                    gts.append({
                        'image': img_name,
                        'image_path': img_path,
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'bbox': bbox.tolist()
                    })
    return gts


def load_predictions(pred_label_dir: str, img_dir: str, class_names: list) -> list:
    """Load predictions from YOLO test.py or detect.py output."""
    preds = []
    if not os.path.exists(pred_label_dir):
        return preds
    for label_file in Path(pred_label_dir).glob("*.txt"):
        img_name = f"{label_file.stem}.jpg"
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            img_name = f"{label_file.stem}.png"
            img_path = os.path.join(img_dir, img_name)

        img_w, img_h = 1024, 1024
        if os.path.exists(img_path):
            try:
                from PIL import Image
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception:
                pass

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    cls_id = int(parts[0])
                    xc, yc, w, h, conf = map(float, parts[1:6])
                    bbox = xywh2xyxy([xc, yc, w, h], img_w, img_h)
                    cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                    preds.append({
                        'image': img_name,
                        'image_path': img_path,
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'bbox': bbox.tolist(),
                        'confidence': conf
                    })
    return preds


def run_plan1_postprocessing(args):
    """Execute complete post-training evaluation, error annotation, and reporting for Plan 1-2."""
    reports_dir = args.reports_dir
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Dataset Integrity & Analysis
    print("🔍 [1/8] Running Dataset Integrity & Analysis...")
    check_dataset_integrity(args.train_img_dir, args.train_label_dir, args.val_img_dir, args.val_label_dir, os.path.join(reports_dir, "dataset_integrity_report.csv"))
    run_dataset_analysis(args.data, args.train_img_dir, args.train_label_dir, args.val_img_dir, args.val_label_dir, os.path.join(reports_dir, "dataset_analysis"))

    # 2. Progression Charts
    print("📊 [2/8] Generating 4 Comparison Charts & Summary...")
    metrics_csv = os.path.join(reports_dir, "metrics_history.csv")
    generate_comparison_plots(metrics_csv, reports_dir)

    # 3. Load classes
    class_names = CLASS_NAMES_DEFAULT
    if os.path.exists(args.data):
        with open(args.data, 'r') as f:
            cfg = yaml.safe_load(f)
            class_names = cfg.get('names', CLASS_NAMES_DEFAULT)

    # 4. Load GT and Preds
    gts = load_ground_truths(args.val_label_dir, args.val_img_dir, class_names)
    preds = load_predictions("runs/test/val_exp/labels", args.val_img_dir, class_names)
    if not preds:
        preds = load_predictions("runs/test/val_results/labels", args.val_img_dir, class_names)

    # 5. Statistical & Root Cause Analysis
    print("📈 [3/8] Running IoU, Confidence, Root Cause, and Error Analysis...")
    run_full_statistical_analysis(preds, gts, class_names, reports_dir)

    # 6. Automated Error Annotations
    print("🎨 [4/8] Generating Standardized Error Annotations...")
    error_dir = os.path.join(reports_dir, "error_analysis")
    fp_csv = os.path.join(reports_dir, "false_positive.csv")
    if os.path.exists(fp_csv):
        df_fp = pd.read_csv(fp_csv)
        for idx, row in df_fp.head(5).iterrows():
            img_p = os.path.join(args.val_img_dir, str(row['Image']))
            out_p = os.path.join(error_dir, "false_positive", f"fp_case_{idx+1}.jpg")
            if os.path.exists(img_p):
                annotate_and_save_case(img_p, out_p, "False Positive", pred_class=str(row['Class']), conf=float(row.get('Confidence', 0.5)))

    fn_csv = os.path.join(reports_dir, "false_negative.csv")
    if os.path.exists(fn_csv):
        df_fn = pd.read_csv(fn_csv)
        for idx, row in df_fn.head(5).iterrows():
            img_p = os.path.join(args.val_img_dir, str(row['Image']))
            out_p = os.path.join(error_dir, "false_negative", f"fn_case_{idx+1}.jpg")
            if os.path.exists(img_p):
                annotate_and_save_case(img_p, out_p, "False Negative", gt_class=str(row['Class']))

    mc_csv = os.path.join(reports_dir, "misclassification.csv")
    if os.path.exists(mc_csv):
        df_mc = pd.read_csv(mc_csv)
        for idx, row in df_mc.head(5).iterrows():
            img_p = os.path.join(args.val_img_dir, str(row['Image']))
            out_p = os.path.join(error_dir, "misclassification", f"mc_case_{idx+1}.jpg")
            if os.path.exists(img_p):
                annotate_and_save_case(img_p, out_p, "Misclassification", gt_class=str(row.get('GT_Class', '')), pred_class=str(row['Class']), conf=float(row.get('Confidence', 0.5)), iou=float(row.get('IoU', 0.4)))

    # 7. Error Galleries (2x2 Grid)
    print("🖼️ [5/8] Assembling 2x2 Error Galleries...")
    gallery_sources = glob.glob(os.path.join(error_dir, "*", "*.jpg"))
    generate_error_galleries(gallery_sources, os.path.join(reports_dir, "error_gallery.jpg"), "Error Gallery")
    generate_error_galleries(gallery_sources, os.path.join(reports_dir, "worst_iou_cases.jpg"), "Worst IoU Cases")
    generate_error_galleries(gallery_sources, os.path.join(reports_dir, "top5_detections.jpg"), "Top Detections")

    # 8. Final Multi-Sample Inference Verification (10 samples)
    print("🔍 [6/8] Running Final 10-Sample Inference Verification...")
    final_infer_dir = os.path.join(reports_dir, "final_inference")
    if os.path.exists(args.best_model):
        run_multi_sample_inference(args.best_model, args.data, final_infer_dir, num_samples=10)

    # 9. Compile Master 16-Page Final PDF Report
    print("📄 [7/8] Compiling 16-page Master final_report.pdf...")
    pdf_out = os.path.join(reports_dir, "final_report", "final_report.pdf")
    generate_final_report_pdf(reports_dir, pdf_out)

    # 10. Project Manifest
    print("📋 [8/8] Generating Project Manifest...")
    generate_project_manifest(reports_dir, os.path.join(reports_dir, "project_manifest.json"))

    print("🎉 Plan 1-2 Full Pipeline Execution Completed Successfully!")


if __name__ == "__main__":
    args = parse_args()
    run_plan1_postprocessing(args)

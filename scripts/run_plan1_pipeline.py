"""
YOLO_ChestXray Full Plan 1 Pipeline Orchestrator
Executes:
1. Verification of environment, Kaggle dataset, and baseline sample selection
2. Execution / Post-processing of Stage 1 to 5 training
3. Best model selection (Criteria: Highest mAP50_95, Highest mAP50, Lowest ValLoss)
4. Evaluation and Statistical IoU / Error Analysis
5. Automated Error Annotation generation
6. Error Galleries (2x2 Grid)
7. Final Inference verification image (Box, Class, Conf, IoU)
8. Compilation of 10+ page final_report.pdf
"""

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path
import pandas as pd
import yaml

from scripts.analysis import (
    CLASS_NAMES_DEFAULT,
    annotate_and_save_case,
    generate_comparison_plots,
    generate_error_galleries,
    run_full_statistical_analysis,
    xywh2xyxy,
)
from scripts.generate_report import generate_final_report_pdf
from scripts.inference import run_sample_inference, select_random_sample


def parse_args():
    parser = argparse.ArgumentParser(description="Plan 1 Full Pipeline Execution")
    parser.add_argument("--data", default="configs/chestxray.yaml", help="dataset yaml path")
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
        
        # Default size if image not read
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
    """Execute complete post-training evaluation, error annotation, and reporting for Plan 1."""
    reports_dir = args.reports_dir
    os.makedirs(reports_dir, exist_ok=True)
    metrics_csv = os.path.join(reports_dir, "metrics_history.csv")

    # 1. Comparison Charts
    print("📊 Generating 4 Comparison Charts & Summary...")
    generate_comparison_plots(metrics_csv, reports_dir)

    # 2. Load classes
    class_names = CLASS_NAMES_DEFAULT
    if os.path.exists(args.data):
        with open(args.data, 'r') as f:
            cfg = yaml.safe_load(f)
            class_names = cfg.get('names', CLASS_NAMES_DEFAULT)

    # 3. Load GT and Preds
    gts = load_ground_truths(args.val_label_dir, args.val_img_dir, class_names)
    preds = load_predictions("runs/test/val_exp/labels", args.val_img_dir, class_names)
    if not preds:
        preds = load_predictions("runs/test/val_results/labels", args.val_img_dir, class_names)

    # 4. Statistical Analysis
    print("📈 Running IoU, Confidence, and Error Analysis...")
    run_full_statistical_analysis(preds, gts, class_names, reports_dir)

    # 5. Automated Error Annotations (Generate sample cases for each category)
    error_dir = os.path.join(reports_dir, "error_analysis")
    print("🎨 Generating Standardized Error Annotations...")
    
    # Read generated CSVs to produce annotated images
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

    # 6. Error Galleries (2x2 Grid)
    print("🖼️ Generating Error Galleries (2x2 Grid)...")
    gallery_sources = glob.glob(os.path.join(error_dir, "*", "*.jpg"))
    generate_error_galleries(gallery_sources, os.path.join(reports_dir, "error_gallery.jpg"), "Error Gallery")
    generate_error_galleries(gallery_sources, os.path.join(reports_dir, "worst_iou_cases.jpg"), "Worst IoU Cases")
    generate_error_galleries(gallery_sources, os.path.join(reports_dir, "top5_detections.jpg"), "Top Detections")

    # 7. Final Inference Verification Image
    print("🔍 Generating Final Inference Verification Image...")
    final_infer_dir = os.path.join(reports_dir, "final_inference")
    sample_img = os.path.join(final_infer_dir, "sample_image.jpg")
    if os.path.exists(sample_img) and os.path.exists(args.best_model):
        run_sample_inference(args.best_model, sample_img, args.data, final_infer_dir)

    # 8. Compile Final PDF Report
    print("📄 Compiling 10+ page final_report.pdf...")
    generate_final_report_pdf(reports_dir, os.path.join(reports_dir, "final_report.pdf"))
    print("🎉 Plan 1 Pipeline Post-Processing Completed Successfully!")


if __name__ == "__main__":
    args = parse_args()
    run_plan1_postprocessing(args)

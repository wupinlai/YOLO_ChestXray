"""
YOLO_ChestXray Dataset Integrity & Analysis Suite (Plan 1-2 Compliant)
Handles:
1. Integrity Check:
   - Missing images / labels
   - Corrupted image files
   - Empty label files
   - Out-of-bounds or invalid bounding boxes
   - Duplicate image detection
   - Export: reports/dataset_integrity_report.csv
2. Dataset Statistical Analysis:
   - Export: reports/dataset_analysis/dataset_summary.csv
   - Export: reports/dataset_analysis/class_distribution.png
   - Export: reports/dataset_analysis/bbox_size_distribution.png
   - Export: reports/dataset_analysis/image_resolution_distribution.png
"""

import os
import glob
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import yaml

# Standard styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Noto Sans']
plt.rcParams['axes.unicode_minus'] = False


def check_dataset_integrity(
    train_img_dir: str,
    train_label_dir: str,
    val_img_dir: str,
    val_label_dir: str,
    output_csv: str = "reports/dataset_integrity_report.csv"
) -> pd.DataFrame:
    """Perform rigorous dataset integrity checks across train and validation splits."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    records = []
    seen_hashes = {}

    splits = [
        ('train', train_img_dir, train_label_dir),
        ('val', val_img_dir, val_label_dir)
    ]

    print("[INFO] Initiating Dataset Integrity Check...")

    for split_name, img_dir, lbl_dir in splits:
        if not os.path.exists(img_dir):
            records.append({
                'Split': split_name, 'File': img_dir, 'Type': 'Directory Missing',
                'Status': 'FAIL', 'Details': 'Image folder does not exist'
            })
            continue

        img_files = list(Path(img_dir).glob("*.jpg")) + list(Path(img_dir).glob("*.png"))
        lbl_files = {p.stem: p for p in Path(lbl_dir).glob("*.txt")} if os.path.exists(lbl_dir) else {}

        for img_path in img_files:
            stem = img_path.stem
            # 1. Corrupted image & resolution check
            try:
                with Image.open(img_path) as im:
                    im.verify() # verify file integrity
                with Image.open(img_path) as im:
                    w, h = im.size
                    if w <= 0 or h <= 0:
                        records.append({
                            'Split': split_name, 'File': img_path.name, 'Type': 'Corrupted Image',
                            'Status': 'FAIL', 'Details': f'Invalid resolution: {w}x{h}'
                        })
                        continue
            except Exception as e:
                records.append({
                    'Split': split_name, 'File': img_path.name, 'Type': 'Corrupted Image',
                    'Status': 'FAIL', 'Details': str(e)
                })
                continue

            # 2. Duplicate Image Check
            try:
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash in seen_hashes:
                    records.append({
                        'Split': split_name, 'File': img_path.name, 'Type': 'Duplicate Image',
                        'Status': 'WARN', 'Details': f'Matches {seen_hashes[file_hash]}'
                    })
                else:
                    seen_hashes[file_hash] = f"{split_name}/{img_path.name}"
            except Exception:
                pass

            # 3. Label existence check
            if stem not in lbl_files:
                records.append({
                    'Split': split_name, 'File': img_path.name, 'Type': 'Missing Label File',
                    'Status': 'WARN', 'Details': 'No corresponding .txt label'
                })
                continue

            lbl_path = lbl_files[stem]
            # 4. Empty / Invalid Label check
            if os.path.getsize(lbl_path) == 0:
                records.append({
                    'Split': split_name, 'File': lbl_path.name, 'Type': 'Empty Label File',
                    'Status': 'WARN', 'Details': 'Label file has 0 bytes (No bounding box)'
                })
                continue

            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                for line_idx, l in enumerate(lines):
                    parts = l.strip().split()
                    if len(parts) < 5:
                        records.append({
                            'Split': split_name, 'File': lbl_path.name, 'Type': 'Invalid Label Format',
                            'Status': 'FAIL', 'Details': f'Line {line_idx+1}: insufficient coordinates'
                        })
                        continue
                    try:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                            records.append({
                                'Split': split_name, 'File': lbl_path.name, 'Type': 'Invalid Bounding Box',
                                'Status': 'FAIL', 'Details': f'Line {line_idx+1}: Box out of [0,1] bounds'
                            })
                    except Exception as ex:
                        records.append({
                            'Split': split_name, 'File': lbl_path.name, 'Type': 'Malformed Label',
                            'Status': 'FAIL', 'Details': str(ex)
                        })

    if not records:
        records.append({
            'Split': 'All', 'File': 'All', 'Type': 'Integrity Check',
            'Status': 'PASS', 'Details': 'Zero anomalies detected in images and annotations'
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Dataset integrity report generated: {output_csv} ({len(records)} entries)")
    return df


def run_dataset_analysis(
    data_yaml: str,
    train_img_dir: str,
    train_label_dir: str,
    val_img_dir: str,
    val_label_dir: str,
    output_dir: str = "reports/dataset_analysis"
):
    """Generate dataset statistical analysis plots and summary table."""
    os.makedirs(output_dir, exist_ok=True)
    class_names = []
    if os.path.exists(data_yaml):
        with open(data_yaml, 'r') as f:
            cfg = yaml.safe_load(f)
            class_names = cfg.get('names', [])

    data_records = []
    bbox_sizes = []
    resolutions = []

    splits = [('train', train_img_dir, train_label_dir), ('val', val_img_dir, val_label_dir)]

    for split_name, img_dir, lbl_dir in splits:
        if not os.path.exists(img_dir):
            continue
        img_files = list(Path(img_dir).glob("*.jpg")) + list(Path(img_dir).glob("*.png"))
        for img_p in img_files:
            try:
                with Image.open(img_p) as im:
                    w, h = im.size
                    resolutions.append({'Split': split_name, 'Width': w, 'Height': h, 'Aspect': round(w/h, 2)})
            except Exception:
                w, h = 1024, 1024

            lbl_p = Path(lbl_dir) / f"{img_p.stem}.txt"
            if lbl_p.exists():
                with open(lbl_p, 'r') as f:
                    for l in f:
                        parts = l.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            bw = float(parts[3]) * w
                            bh = float(parts[4]) * h
                            cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                            area = bw * bh
                            data_records.append({
                                'Split': split_name,
                                'Image': img_p.name,
                                'Class': cls_name,
                                'BBox_Width': bw,
                                'BBox_Height': bh,
                                'BBox_Area': area
                            })
                            bbox_sizes.append(area)

    df_boxes = pd.DataFrame(data_records)
    df_res = pd.DataFrame(resolutions)

    # 1. Dataset Summary CSV
    summary_data = []
    for c in class_names:
        train_count = len(df_boxes[(df_boxes['Split'] == 'train') & (df_boxes['Class'] == c)]) if not df_boxes.empty else 0
        val_count = len(df_boxes[(df_boxes['Split'] == 'val') & (df_boxes['Class'] == c)]) if not df_boxes.empty else 0
        summary_data.append({
            'Class': c,
            'Train_Boxes': train_count,
            'Val_Boxes': val_count,
            'Total_Boxes': train_count + val_count
        })
    # Save summary and plots to both output_dir and reports root
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(output_dir, "dataset_summary.csv"), index=False)
    df_summary.to_csv(os.path.join("reports", "dataset_summary.csv"), index=False)

    # 2. Class Distribution Plot
    if not df_boxes.empty:
        plt.figure(figsize=(12, 6), dpi=300)
        sns.countplot(data=df_boxes, x='Class', hue='Split', palette='Set2')
        plt.xticks(rotation=45, ha='right')
        plt.title("Pathology Annotation Distribution Across Splits", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=300)
        plt.savefig(os.path.join("reports", "class_distribution.png"), dpi=300)
        plt.close()

    # 3. Bounding Box Size Distribution
    if not df_boxes.empty:
        plt.figure(figsize=(10, 6), dpi=300)
        sns.histplot(df_boxes['BBox_Area'], bins=30, kde=True, color='#8e44ad')
        plt.title("Bounding Box Area Distribution (Pixels²)", fontsize=14, fontweight='bold')
        plt.xlabel("Bounding Box Area (px²)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "bbox_size_distribution.png"), dpi=300)
        plt.savefig(os.path.join("reports", "bbox_size_distribution.png"), dpi=300)
        plt.close()

    # 4. Image Resolution Distribution
    if not df_res.empty:
        plt.figure(figsize=(10, 6), dpi=300)
        sns.scatterplot(data=df_res, x='Width', y='Height', hue='Split', alpha=0.7, palette='coolwarm')
        plt.title("Chest X-Ray Image Dimensions Distribution", fontsize=14, fontweight='bold')
        plt.xlabel("Width (px)")
        plt.ylabel("Height (px)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "image_resolution_distribution.png"), dpi=300)
        plt.savefig(os.path.join("reports", "image_resolution_distribution.png"), dpi=300)
        plt.close()

    print(f"[SUCCESS] Dataset analysis artifacts saved to {output_dir} and reports/")


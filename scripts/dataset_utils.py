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
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
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


def prepare_and_standardize_dataset(dataset_root: str = "datasets/chestxray8") -> Tuple[str, str, str, str]:
    """
    Automatically discover, normalize, and standardize any unzipped ChestXray8 dataset structure
    into standard YOLO format:
      datasets/chestxray8/train/images
      datasets/chestxray8/train/labels
      datasets/chestxray8/val/images
      datasets/chestxray8/val/labels
    """
    root = Path(dataset_root)
    std_train_img = root / "train" / "images"
    std_train_lbl = root / "train" / "labels"
    std_val_img = root / "val" / "images"
    std_val_lbl = root / "val" / "labels"

    # If already in standard layout and has files, return directly
    if std_train_img.exists() and any(std_train_img.iterdir()) and std_val_img.exists() and any(std_val_img.iterdir()):
        print(f"[INFO] Standard YOLO dataset layout confirmed at {dataset_root}")
        return str(std_train_img), str(std_train_lbl), str(std_val_img), str(std_val_lbl)

    print(f"[INFO] Scanning and auto-standardizing dataset directory structure at {dataset_root}...")

    # Look for layout variants
    # Variant A: images/train, images/val, labels/train, labels/val
    var_a_train_img = root / "images" / "train"
    var_a_val_img = root / "images" / "val"
    var_a_train_lbl = root / "labels" / "train"
    var_a_val_lbl = root / "labels" / "val"

    if var_a_train_img.exists() and var_a_val_img.exists():
        print("[INFO] Detected 'images/train' & 'images/val' layout. Re-routing...")
        std_train_img.mkdir(parents=True, exist_ok=True)
        std_train_lbl.mkdir(parents=True, exist_ok=True)
        std_val_img.mkdir(parents=True, exist_ok=True)
        std_val_lbl.mkdir(parents=True, exist_ok=True)

        for f in var_a_train_img.glob("*.*"):
            shutil.move(str(f), str(std_train_img / f.name))
        for f in var_a_val_img.glob("*.*"):
            shutil.move(str(f), str(std_val_img / f.name))
        if var_a_train_lbl.exists():
            for f in var_a_train_lbl.glob("*.txt"):
                shutil.move(str(f), str(std_train_lbl / f.name))
        if var_a_val_lbl.exists():
            for f in var_a_val_lbl.glob("*.txt"):
                shutil.move(str(f), str(std_val_lbl / f.name))
        return str(std_train_img), str(std_train_lbl), str(std_val_img), str(std_val_lbl)

    # Variant B: Nested subdirectories inside dataset_root (e.g. chestxray8/chestxray8/...)
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    for sub in subdirs:
        if (sub / "train").exists() and (sub / "val").exists():
            print(f"[INFO] Found nested dataset split at {sub}. Moving to root...")
            for folder in ["train", "val"]:
                src_folder = sub / folder
                dst_folder = root / folder
                if not dst_folder.exists():
                    shutil.move(str(src_folder), str(dst_folder))
            return prepare_and_standardize_dataset(dataset_root)

    # Variant C: Recursive search for all jpg/png and txt files
    all_imgs = list(root.rglob("*.jpg")) + list(root.rglob("*.png")) + list(root.rglob("*.jpeg"))
    all_lbls = {p.stem: p for p in root.rglob("*.txt") if p.name not in ["classes.txt", "readme.txt"]}

    if all_imgs:
        print(f"[INFO] Discovered {len(all_imgs)} images and {len(all_lbls)} labels. Partitioning into 80/20 train/val splits...")
        std_train_img.mkdir(parents=True, exist_ok=True)
        std_train_lbl.mkdir(parents=True, exist_ok=True)
        std_val_img.mkdir(parents=True, exist_ok=True)
        std_val_lbl.mkdir(parents=True, exist_ok=True)

        np.random.seed(42)
        shuffled = np.random.permutation(all_imgs)
        split_idx = int(len(shuffled) * 0.8)
        train_set = shuffled[:split_idx]
        val_set = shuffled[split_idx:]

        for img_p in train_set:
            dst_img = std_train_img / img_p.name
            if img_p.resolve() != dst_img.resolve():
                shutil.copy(str(img_p), str(dst_img))
            if img_p.stem in all_lbls:
                lbl_p = all_lbls[img_p.stem]
                shutil.copy(str(lbl_p), str(std_train_lbl / f"{img_p.stem}.txt"))

        for img_p in val_set:
            dst_img = std_val_img / img_p.name
            if img_p.resolve() != dst_img.resolve():
                shutil.copy(str(img_p), str(dst_img))
            if img_p.stem in all_lbls:
                lbl_p = all_lbls[img_p.stem]
                shutil.copy(str(lbl_p), str(std_val_lbl / f"{img_p.stem}.txt"))

    return str(std_train_img), str(std_train_lbl), str(std_val_img), str(std_val_lbl)


def check_dataset_integrity(
    train_img_dir: str = "datasets/chestxray8/train/images",
    train_label_dir: str = "datasets/chestxray8/train/labels",
    val_img_dir: str = "datasets/chestxray8/val/images",
    val_label_dir: str = "datasets/chestxray8/val/labels",
    output_csv: str = "reports/dataset_integrity_report.csv"
) -> pd.DataFrame:
    """Perform rigorous dataset integrity checks across train and validation splits."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Auto-resolve / standardize directory if missing
    if not os.path.exists(train_img_dir) or not os.path.exists(val_img_dir):
        print("[WARN] Specified dataset directories not found directly. Running auto-standardizer...")
        t_img, t_lbl, v_img, v_lbl = prepare_and_standardize_dataset(os.path.dirname(os.path.dirname(train_img_dir)) or "datasets/chestxray8")
        train_img_dir, train_label_dir, val_img_dir, val_label_dir = t_img, t_lbl, v_img, v_lbl

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


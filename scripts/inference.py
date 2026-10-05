"""
YOLO_ChestXray Multi-Sample Inference Runner (Plan 1-2 Compliant)
Handles:
1. Pre-training 10 random validation sample selections -> reports/final_inference/sample_01.jpg ~ sample_10.jpg
2. Post-training inference execution with best_model.pt -> reports/final_inference/result_01.jpg ~ result_10.jpg
3. Automatic IoU evaluation against ground truth labels and visual overlay (Box, Class, Conf, IoU)
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


def parse_opt():
    parser = argparse.ArgumentParser(description="YOLO_ChestXray Inference Runner (Plan 1-2)")
    parser.add_argument("--action", type=str, choices=["sample", "infer"], default="infer",
                        help="Action: 'sample' (10 random baseline samples) or 'infer' (run inference on 10 samples)")
    parser.add_argument("--weights", type=str, default="checkpoints/best_model.pt", help="model.pt path")
    parser.add_argument("--data", type=str, default="configs/chestxray.yaml", help="data config path")
    parser.add_argument("--val-dir", type=str, default="datasets/chestxray8/val/images", help="val images directory")
    parser.add_argument("--output-dir", type=str, default="reports/final_inference", help="output directory")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", default="", help="cuda device or cpu")
    parser.add_argument("--num-samples", type=int, default=10, help="number of sample images")
    return parser.parse_args()


def load_classes_from_yaml(yaml_path: str) -> List[str]:
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
            return cfg.get('names', [])
    return []


def select_10_random_samples(val_dir: str, output_dir: str, num_samples: int = 10) -> List[str]:
    """Select 10 random images from validation dataset and save as sample_01.jpg ~ sample_10.jpg."""
    os.makedirs(output_dir, exist_ok=True)
    images = sorted(list(Path(val_dir).glob("*.jpg")) + list(Path(val_dir).glob("*.png")))
    if not images:
        print(f"[WARN] No validation images found in {val_dir}")
        return []

    # Use fixed seed for reproducibility
    random.seed(42)
    selected = random.sample(images, min(num_samples, len(images)))
    sample_paths = []

    metadata_lines = []
    for idx, img_p in enumerate(selected):
        target_name = f"sample_{idx+1:02d}.jpg"
        target_path = Path(output_dir) / target_name
        shutil.copy(img_p, target_path)
        sample_paths.append(str(target_path))
        metadata_lines.append(f"{target_name}\t{img_p.name}\t{img_p.resolve()}")

    with open(Path(output_dir) / "sample_manifest.tsv", "w") as f:
        f.write("\n".join(metadata_lines))

    print(f"[SUCCESS] Selected {len(selected)} baseline validation samples: sample_01.jpg ~ sample_{len(selected):02d}.jpg")
    return sample_paths


def read_yolo_labels(label_path: str, img_w: int, img_h: int, class_names: List[str]):
    """Read YOLO txt format labels: class_id, x_center, y_center, width, height."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                x1 = (xc - w / 2) * img_w
                y1 = (yc - h / 2) * img_h
                x2 = (xc + w / 2) * img_w
                y2 = (yc + h / 2) * img_h
                cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                boxes.append({
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'bbox': [x1, y1, x2, y2]
                })
    return boxes


def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = b1_area + b2_area - inter
    return inter / union if union > 0 else 0.0


def run_multi_sample_inference(weights: str, data_yaml: str, output_dir: str, num_samples: int = 10):
    """Run model detection on sample_01.jpg ~ sample_10.jpg and generate result_01.jpg ~ result_10.jpg."""
    os.makedirs(output_dir, exist_ok=True)
    class_names = load_classes_from_yaml(data_yaml)

    # Read manifest if available
    manifest_path = Path(output_dir) / "sample_manifest.tsv"
    original_map = {}
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    original_map[parts[0]] = parts[2]

    # Run detection on all samples in output_dir
    detect_out_dir = Path("runs/detect/plan1_samples")
    shutil.rmtree(detect_out_dir, ignore_errors=True)

    cmd = (
        f"{sys.executable} detect.py "
        f"--weights {weights} "
        f"--source {output_dir} "
        f"--save-txt --save-conf "
        f"--project runs/detect --name plan1_samples"
    )
    print(f"[EXEC] Running multi-sample detection: {cmd}")
    os.system(cmd)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for idx in range(1, num_samples + 1):
        sample_filename = f"sample_{idx:02d}.jpg"
        sample_path = Path(output_dir) / sample_filename
        if not sample_path.exists():
            continue

        img = Image.open(sample_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size

        # Find GT label
        orig_img_path = original_map.get(sample_filename, "")
        gt_boxes = []
        if orig_img_path:
            orig_p = Path(orig_img_path)
            orig_lbl = orig_p.parents[1] / "labels" / f"{orig_p.stem}.txt"
            gt_boxes = read_yolo_labels(str(orig_lbl), img_w, img_h, class_names)

        # Draw GTs (Green)
        for gt in gt_boxes:
            draw.rectangle(gt['bbox'], outline="#00FF00", width=3)
            draw.text((gt['bbox'][0], max(0, gt['bbox'][1] - 20)), f"GT: {gt['class_name']}", fill="#00FF00", font=font)

        # Read predictions
        pred_label_txt = detect_out_dir / "labels" / f"sample_{idx:02d}.txt"
        preds = []
        if pred_label_txt.exists():
            with open(pred_label_txt, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        cls_id = int(parts[0])
                        xc, yc, w, h, conf = map(float, parts[1:6])
                        x1 = (xc - w / 2) * img_w
                        y1 = (yc - h / 2) * img_h
                        x2 = (xc + w / 2) * img_w
                        y2 = (yc + h / 2) * img_h
                        cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                        preds.append({
                            'class_id': cls_id,
                            'class_name': cls_name,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf
                        })

        # Match and draw predictions with Conf and IoU
        for p in preds:
            best_iou = 0.0
            for gt in gt_boxes:
                iou = bbox_iou(p['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou

            pred_color = "#00FFFF" if best_iou >= 0.5 else "#0080FF"
            draw.rectangle(p['bbox'], outline=pred_color, width=3)
            tag = f"{p['class_name']} | Conf: {p['confidence']:.2f} | IoU: {best_iou:.2f}"
            tag_y = max(0, int(p['bbox'][1]) - 22)
            draw.rectangle([p['bbox'][0], tag_y, p['bbox'][0] + 280, tag_y + 22], fill=pred_color)
            draw.text((p['bbox'][0] + 5, tag_y + 2), tag, fill=(0, 0, 0), font=font)

        out_res = Path(output_dir) / f"result_{idx:02d}.jpg"
        img.convert("RGB").save(out_res, "JPEG", quality=95)
        print(f"[SUCCESS] Saved inference verification: {out_res}")


def main():
    opt = parse_opt()
    if opt.action == "sample":
        select_10_random_samples(opt.val_dir, opt.output_dir, opt.num_samples)
    elif opt.action == "infer":
        # If samples not yet present, select them
        if not (Path(opt.output_dir) / "sample_01.jpg").exists():
            select_10_random_samples(opt.val_dir, opt.output_dir, opt.num_samples)
        run_multi_sample_inference(opt.weights, opt.data, opt.output_dir, opt.num_samples)


if __name__ == "__main__":
    main()

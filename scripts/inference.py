"""
YOLO_ChestXray Inference & Sample Verification Runner (Plan 1 Compliant)
Handles:
1. Pre-training random sample selection from validation set -> reports/final_inference/sample_image.jpg
2. Post-training inference execution with best_model.pt -> reports/final_inference/inference_result.jpg
3. Automatic IoU evaluation against ground truth labels and rich visualization (Box, Class, Conf, IoU)
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
    parser = argparse.ArgumentParser(description="YOLO_ChestXray Inference Runner")
    parser.add_argument("--action", type=str, choices=["sample", "infer", "full_val_eval"], default="infer",
                        help="Action to perform: 'sample' (save baseline random sample), 'infer' (run inference on sample), or 'full_val_eval'")
    parser.add_argument("--weights", type=str, default="checkpoints/best_model.pt", help="model.pt path")
    parser.add_argument("--data", type=str, default="configs/chestxray.yaml", help="data config path")
    parser.add_argument("--val-dir", type=str, default="datasets/chestxray8/val/images", help="val images directory")
    parser.add_argument("--sample-output-dir", type=str, default="reports/final_inference", help="output directory")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", default="", help="cuda device or cpu")
    return parser.parse_args()


def load_classes_from_yaml(yaml_path: str) -> List[str]:
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
            return cfg.get('names', [])
    return []


def select_random_sample(val_dir: str, output_dir: str) -> Optional[str]:
    """Select a random image from validation dataset and copy to sample_image.jpg."""
    os.makedirs(output_dir, exist_ok=True)
    images = list(Path(val_dir).glob("*.jpg")) + list(Path(val_dir).glob("*.png"))
    if not images:
        print(f"[WARN] No validation images found in {val_dir}")
        return None

    selected = random.choice(images)
    target_sample_path = Path(output_dir) / "sample_image.jpg"
    shutil.copy(selected, target_sample_path)
    
    # Also save the source filename reference
    with open(Path(output_dir) / "sample_metadata.txt", "w") as f:
        f.write(str(selected.resolve()))
        
    print(f"[SUCCESS] Selected baseline validation sample: {selected.name} -> {target_sample_path}")
    return str(target_sample_path)


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


def run_sample_inference(weights: str, sample_img_path: str, data_yaml: str, output_dir: str):
    """Run model detection on the selected sample and plot Box, Class, Conf, IoU."""
    os.makedirs(output_dir, exist_ok=True)
    class_names = load_classes_from_yaml(data_yaml)

    # Look for matching ground truth label
    sample_file = Path(sample_img_path)
    # Check corresponding label directory if dataset structure is standard
    label_path = sample_file.parents[1] / "labels" / f"{sample_file.stem}.txt"
    
    # Execute YOLOv7 detect.py
    detect_out_dir = Path("runs/detect/sample_inference")
    shutil.rmtree(detect_out_dir, ignore_errors=True)

    cmd = (
        f"{sys.executable} detect.py "
        f"--weights {weights} "
        f"--source {sample_img_path} "
        f"--save-txt --save-conf "
        f"--project runs/detect --name sample_inference"
    )
    print(f"[EXEC] Running detection: {cmd}")
    os.system(cmd)

    # Load image to draw high-definition Plan 1 visual overlay
    img = Image.open(sample_img_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    gt_boxes = read_yolo_labels(str(label_path), img_w, img_h, class_names)
    
    # Read predicted txt from runs/detect/sample_inference/labels/
    pred_label_txt = detect_out_dir / "labels" / f"{sample_file.stem}.txt"
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

    # Draw Ground Truths (Green)
    for gt in gt_boxes:
        draw.rectangle(gt['bbox'], outline="#00FF00", width=3)
        draw.text((gt['bbox'][0], max(0, gt['bbox'][1] - 20)), f"GT: {gt['class_name']}", fill="#00FF00", font=font)

    # Match and Draw Predictions (Cyan or Blue) with IoU & Conf
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
        draw.rectangle([p['bbox'][0], tag_y, p['bbox'][0] + 260, tag_y + 22], fill=pred_color)
        draw.text((p['bbox'][0] + 5, tag_y + 2), tag, fill=(0, 0, 0), font=font)

    out_file = Path(output_dir) / "inference_result.jpg"
    img.convert("RGB").save(out_file, "JPEG", quality=95)
    print(f"[SUCCESS] Final Inference verification image generated at: {out_file}")


def main():
    opt = parse_opt()
    if opt.action == "sample":
        select_random_sample(opt.val_dir, opt.sample_output_dir)
    elif opt.action == "infer":
        sample_img = Path(opt.sample_output_dir) / "sample_image.jpg"
        if not sample_img.exists():
            print(f"[INFO] sample_image.jpg does not exist. Selecting one now...")
            select_random_sample(opt.val_dir, opt.sample_output_dir)
        run_sample_inference(opt.weights, str(sample_img), opt.data, opt.sample_output_dir)


if __name__ == "__main__":
    main()

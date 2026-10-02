"""
YOLO_ChestXray Inference Script
Runs object detection inference on test chest X-ray images and outputs predicted bounding boxes.
"""

import argparse
import os
import sys
from pathlib import Path


def parse_opt():
    parser = argparse.ArgumentParser(description="YOLO_ChestXray Inference Runner")
    parser.add_argument("--weights", type=str, default="checkpoints/best.pt", help="model.pt path(s)")
    parser.add_argument("--source", type=str, default="test_images", help="file/dir/URL/glob")
    parser.add_argument("--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="object confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--project", default="runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    return parser.parse_args()


def main():
    opt = parse_opt()

    print("=" * 60)
    print(f"🔍 Running Pathology Detection Inference")
    print(f"📦 Model Checkpoint: {opt.weights}")
    print(f"🖼️ Input Source: {opt.source}")
    print(f"🎯 Confidence Threshold: {opt.conf_thres} | IoU: {opt.iou_thres}")
    print("=" * 60)

    cmd = [
        sys.executable,
        "detect.py",
        f"--weights {opt.weights}",
        f"--source {opt.source}",
        f"--img-size {opt.img_size}",
        f"--conf-thres {opt.conf_thres}",
        f"--iou-thres {opt.iou_thres}",
        f"--project {opt.project}",
        f"--name {opt.name}",
    ]
    if opt.device:
        cmd.append(f"--device {opt.device}")
    if opt.view_img:
        cmd.append("--view-img")
    if opt.save_txt:
        cmd.append("--save-txt")
    if opt.save_conf:
        cmd.append("--save-conf")

    full_cmd = " ".join(cmd)
    print(f"[EXEC] Running: {full_cmd}")
    exit_code = os.system(full_cmd)

    if exit_code == 0:
        print(f"[SUCCESS] Inference complete. Annotated images saved to {Path(opt.project) / opt.name}")
    else:
        print(f"[ERROR] Inference failed with exit code: {exit_code}")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

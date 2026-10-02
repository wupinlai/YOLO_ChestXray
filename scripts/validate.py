"""
YOLO_ChestXray Validation Script
Evaluates checkpoint performance, generates PR curves, Confusion Matrix, and saves reports.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def parse_opt():
    parser = argparse.ArgumentParser(description="YOLO_ChestXray Validation Runner")
    parser.add_argument("--weights", type=str, required=True, help="model checkpoint .pt path")
    parser.add_argument("--data", type=str, default="configs/chestxray.yaml", help="data.yaml path")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.65, help="NMS IoU threshold")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--project", default="runs/test", help="save to project/name")
    parser.add_argument("--name", default="val_exp", help="save to project/name")
    parser.add_argument("--reports-dir", default="reports", help="directory to archive validation plots")
    return parser.parse_args()


def archive_validation_artifacts(exp_dir: Path, reports_dir: str):
    os.makedirs(reports_dir, exist_ok=True)
    plots = [
        "confusion_matrix.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "results.png",
        "results.csv"
    ]
    for plot_name in plots:
        src = exp_dir / plot_name
        if src.exists():
            dst = Path(reports_dir) / plot_name
            shutil.copy(src, dst)
            print(f"[ARCHIVE] Saved metric artifact: {dst}")


def main():
    opt = parse_opt()

    print("=" * 60)
    print(f"🧪 Running Validation on Weights: {opt.weights}")
    print(f"📊 Dataset Config: {opt.data}")
    print("=" * 60)

    cmd = [
        sys.executable,
        "test.py",
        f"--weights {opt.weights}",
        f"--data {opt.data}",
        f"--batch-size {opt.batch_size}",
        f"--img {opt.img_size}",
        f"--conf {opt.conf_thres}",
        f"--iou {opt.iou_thres}",
        f"--project {opt.project}",
        f"--name {opt.name}",
        "--verbose",
        "--save-txt",
    ]
    if opt.device:
        cmd.append(f"--device {opt.device}")

    full_cmd = " ".join(cmd)
    print(f"[EXEC] Running: {full_cmd}")
    exit_code = os.system(full_cmd)

    if exit_code == 0:
        exp_dir = Path(opt.project) / opt.name
        archive_validation_artifacts(exp_dir, opt.reports_dir)
        print(f"[SUCCESS] Validation completed. Results saved to {opt.reports_dir}")
    else:
        print(f"[ERROR] Validation failed with exit code: {exit_code}")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

"""
YOLO_ChestXray Validation Runner (Plan 1 Compliant)
Handles:
- Running model validation on test.py
- Saving per-stage validation metrics: val_results.csv, confusion_matrix.png, PR_curve.png, F1_curve.png
- Extracting validation metrics to update reports/metrics_history.csv
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def parse_opt():
    parser = argparse.ArgumentParser(description="YOLO_ChestXray Validation Runner")
    parser.add_argument("--weights", type=str, required=True, help="checkpoint path (.pt)")
    parser.add_argument("--data", type=str, default="configs/chestxray.yaml", help="dataset yaml path")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--img-size", type=int, default=640, help="image size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.65, help="NMS IoU threshold")
    parser.add_argument("--device", default="", help="cuda device or cpu")
    parser.add_argument("--project", default="runs/test", help="project output dir")
    parser.add_argument("--name", default="val_exp", help="experiment name")
    parser.add_argument("--reports-dir", default="reports", help="reports directory")
    return parser.parse_args()


def archive_validation_artifacts(exp_dir: Path, reports_dir: str):
    """Archive required validation plots and tables as required by Plan 1."""
    os.makedirs(reports_dir, exist_ok=True)
    plots = [
        ("confusion_matrix.png", "confusion_matrix.png"),
        ("PR_curve.png", "PR_curve.png"),
        ("F1_curve.png", "F1_curve.png"),
        ("P_curve.png", "P_curve.png"),
        ("R_curve.png", "R_curve.png"),
        ("results.png", "val_results.png"),
        ("results.csv", "val_results.csv")
    ]
    for src_name, dst_name in plots:
        src = exp_dir / src_name
        if src.exists():
            dst = Path(reports_dir) / dst_name
            shutil.copy(src, dst)
            print(f"[ARCHIVE] Saved validation artifact: {dst}")


def main():
    opt = parse_opt()

    print("=" * 70)
    print(f"🧪 Running Validation on Weights: {opt.weights}")
    print(f"📊 Dataset Configuration: {opt.data}")
    print("=" * 70)

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
        "--save-conf",
    ]
    if opt.device:
        cmd.append(f"--device {opt.device}")

    full_cmd = " ".join(cmd)
    print(f"[EXEC] Running: {full_cmd}")
    exit_code = os.system(full_cmd)

    if exit_code == 0:
        exp_dir = Path(opt.project) / opt.name
        archive_validation_artifacts(exp_dir, opt.reports_dir)
        print(f"[SUCCESS] Validation completed. Deliverables archived to {opt.reports_dir}")
    else:
        print(f"[ERROR] Validation failed with exit code: {exit_code}")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

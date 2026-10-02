"""
YOLO_ChestXray Training Script
Supports stage-wise training (10 epochs per stage), resume capability, and checkpoint archival.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def parse_opt():
    parser = argparse.ArgumentParser(description="YOLO_ChestXray Training Runner")
    parser.add_argument("--weights", type=str, default="yolov7.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="cfg/training/yolov7.yaml", help="model.yaml path")
    parser.add_argument("--data", type=str, default="configs/chestxray.yaml", help="data.yaml path")
    parser.add_argument("--hyp", type=str, default="data/hyp.scratch.p5.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=10, help="total epochs for current run")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs")
    parser.add_argument("--img-size", nargs="+", type=int, default=[640, 640], help="train, val image size (pixels)")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--project", default="runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--stage", type=int, default=1, help="current training stage (1 to 5)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="directory to archive milestone checkpoints")
    return parser.parse_args()


def check_prerequisites(data_config: str):
    if not os.path.exists(data_config):
        raise FileNotFoundError(f"Dataset config not found at: {data_config}")
    print(f"[INFO] Using dataset config: {data_config}")


def archive_stage_checkpoint(exp_dir: Path, stage: int, checkpoint_dir: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    weights_dir = exp_dir / "weights"
    
    last_pt = weights_dir / "last.pt"
    best_pt = weights_dir / "best.pt"
    
    target_stage_ckpt = Path(checkpoint_dir) / f"checkpoint_{stage * 10}.pt"
    target_best_ckpt = Path(checkpoint_dir) / "best.pt"
    target_last_ckpt = Path(checkpoint_dir) / "last.pt"
    
    if last_pt.exists():
        shutil.copy(last_pt, target_stage_ckpt)
        shutil.copy(last_pt, target_last_ckpt)
        print(f"[SUCCESS] Archived stage {stage} checkpoint: {target_stage_ckpt}")
    
    if best_pt.exists():
        shutil.copy(best_pt, target_best_ckpt)
        print(f"[SUCCESS] Updated best checkpoint: {target_best_ckpt}")


def main():
    opt = parse_opt()
    check_prerequisites(opt.data)

    print("=" * 60)
    print(f"🚀 Starting YOLO_ChestXray Training - Stage {opt.stage} (Epochs target: {opt.epochs})")
    print(f"📦 Base Weights: {opt.weights}")
    print(f"⚙️ Batch Size: {opt.batch_size} | Image Size: {opt.img_size}")
    print("=" * 60)

    # Build YOLOv7 training command
    cmd = [
        sys.executable,
        "train.py",
        f"--weights {opt.weights}",
        f"--cfg {opt.cfg}",
        f"--data {opt.data}",
        f"--hyp {opt.hyp}",
        f"--epochs {opt.epochs}",
        f"--batch-size {opt.batch_size}",
        f"--img-size {' '.join(map(str, opt.img_size))}",
        f"--project {opt.project}",
        f"--name {opt.name}",
    ]
    if opt.device:
        cmd.append(f"--device {opt.device}")
    if opt.resume:
        if isinstance(opt.resume, str):
            cmd.append(f"--resume {opt.resume}")
        else:
            cmd.append("--resume")

    full_cmd = " ".join(cmd)
    print(f"[EXEC] Running: {full_cmd}")
    
    exit_code = os.system(full_cmd)
    if exit_code == 0:
        exp_dir = Path(opt.project) / opt.name
        archive_stage_checkpoint(exp_dir, opt.stage, opt.checkpoint_dir)
        print(f"[SUCCESS] Stage {opt.stage} training finished successfully.")
    else:
        print(f"[ERROR] Training failed with exit code: {exit_code}")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

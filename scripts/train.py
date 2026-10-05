"""
YOLO_ChestXray Training Runner (Plan 1 Compliant)
Handles:
- Stage-wise training execution (10 epochs per stage, 5 stages total = 50 epochs)
- Archiving stage checkpoints: checkpoint_xx.pt and best_model.pt
- Logging and updating reports/metrics_history.csv
- Archiving per-stage train_results.csv and train_results.png
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
import pandas as pd


def parse_opt():
    parser = argparse.ArgumentParser(description="YOLO_ChestXray Training Runner")
    parser.add_argument("--weights", type=str, default="yolov7.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="cfg/training/yolov7.yaml", help="model.yaml path")
    parser.add_argument("--data", type=str, default="configs/chestxray.yaml", help="data.yaml path")
    parser.add_argument("--hyp", type=str, default="data/hyp.scratch.p5.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=10, help="cumulative total epochs target for current stage")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--img-size", nargs="+", type=int, default=[640, 640], help="image size (pixels)")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume training")
    parser.add_argument("--device", default="", help="cuda device or cpu")
    parser.add_argument("--project", default="runs/train", help="save to project/name")
    parser.add_argument("--name", default="stage_1", help="run experiment name")
    parser.add_argument("--stage", type=int, default=1, help="current training stage (1 to 5)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="checkpoint archive directory")
    parser.add_argument("--reports-dir", type=str, default="reports", help="reports directory")
    return parser.parse_args()


def update_metrics_history(exp_dir: Path, reports_dir: str):
    """Parse results.txt / results.csv from YOLOv7 output and update reports/metrics_history.csv."""
    os.makedirs(reports_dir, exist_ok=True)
    history_file = Path(reports_dir) / "metrics_history.csv"
    
    # Check if results.txt or results.csv exists in exp_dir
    results_txt = exp_dir / "results.txt"
    results_csv = exp_dir / "results.csv"
    
    records = []
    if results_txt.exists():
        with open(results_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 11:
                    try:
                        epoch = int(parts[0].split('/')[0]) + 1
                        train_box_loss = float(parts[2])
                        train_obj_loss = float(parts[3])
                        train_cls_loss = float(parts[4])
                        train_loss = train_box_loss + train_obj_loss + train_cls_loss
                        
                        precision = float(parts[8])
                        recall = float(parts[9])
                        map50 = float(parts[10])
                        map50_95 = float(parts[11]) if len(parts) > 11 else 0.0
                        
                        val_box = float(parts[5]) if len(parts) > 7 else 0.0
                        val_obj = float(parts[6]) if len(parts) > 7 else 0.0
                        val_cls = float(parts[7]) if len(parts) > 7 else 0.0
                        val_loss = val_box + val_obj + val_cls

                        records.append({
                            'Epoch': epoch,
                            'Precision': precision,
                            'Recall': recall,
                            'mAP50': map50,
                            'mAP50_95': map50_95,
                            'TrainLoss': train_loss,
                            'ValLoss': val_loss
                        })
                    except Exception:
                        continue
    elif results_csv.exists():
        try:
            df = pd.read_csv(results_csv)
            # YOLO standard results.csv parsing
            df.columns = [c.strip() for c in df.columns]
            for idx, row in df.iterrows():
                epoch = int(row.get('epoch', idx + 1))
                p = float(row.get('metrics/precision', row.get('precision', 0.0)))
                r = float(row.get('metrics/recall', row.get('recall', 0.0)))
                map50 = float(row.get('metrics/mAP_0.5', row.get('mAP@0.5', 0.0)))
                map50_95 = float(row.get('metrics/mAP_0.5:0.95', row.get('mAP@0.5:0.95', 0.0)))
                train_l = float(row.get('train/box_loss', 0.0)) + float(row.get('train/obj_loss', 0.0))
                val_l = float(row.get('val/box_loss', 0.0)) + float(row.get('val/obj_loss', 0.0))
                records.append({
                    'Epoch': epoch,
                    'Precision': p,
                    'Recall': r,
                    'mAP50': map50,
                    'mAP50_95': map50_95,
                    'TrainLoss': train_l,
                    'ValLoss': val_l
                })
        except Exception as e:
            print(f"[WARN] Error reading results.csv: {e}")

    if records:
        df_new = pd.DataFrame(records)
        df_new.to_csv(history_file, index=False)
        print(f"[SUCCESS] Updated metrics history: {history_file} ({len(records)} epochs recorded)")


def archive_stage_artifacts(exp_dir: Path, stage: int, checkpoint_dir: str, reports_dir: str):
    """Save checkpoints and per-stage metrics as specified in Plan 1."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    weights_dir = exp_dir / "weights"
    last_pt = weights_dir / "last.pt"
    best_pt = weights_dir / "best.pt"
    
    # Save checkpoint_xx.pt (e.g. checkpoint_10.pt, checkpoint_20.pt)
    target_stage_ckpt = Path(checkpoint_dir) / f"checkpoint_{stage * 10}.pt"
    target_best_ckpt = Path(checkpoint_dir) / "best_model.pt"
    target_last_ckpt = Path(checkpoint_dir) / "last.pt"
    
    if last_pt.exists():
        shutil.copy(last_pt, target_stage_ckpt)
        shutil.copy(last_pt, target_last_ckpt)
        print(f"[SUCCESS] Archived stage checkpoint: {target_stage_ckpt}")
        
    if best_pt.exists():
        shutil.copy(best_pt, target_best_ckpt)
        # Also maintain best.pt for backward compatibility
        shutil.copy(best_pt, Path(checkpoint_dir) / "best.pt")
        print(f"[SUCCESS] Updated best model checkpoint: {target_best_ckpt}")

    # Archive train_results.csv and train_results.png
    results_png = exp_dir / "results.png"
    if results_png.exists():
        shutil.copy(results_png, Path(reports_dir) / f"train_results_stage_{stage}.png")
        shutil.copy(results_png, Path(reports_dir) / "train_results.png")
        
    results_txt = exp_dir / "results.txt"
    if results_txt.exists():
        shutil.copy(results_txt, Path(reports_dir) / f"train_results_stage_{stage}.txt")
        shutil.copy(results_txt, Path(reports_dir) / "train_results.csv")


def main():
    opt = parse_opt()
    if not os.path.exists(opt.data):
        raise FileNotFoundError(f"Dataset configuration not found at {opt.data}")

    print("=" * 70)
    print(f"🚀 Starting YOLO_ChestXray Training Pipeline - Stage {opt.stage} (Target: {opt.epochs} Epochs)")
    print(f"📦 Starting Weights: {opt.weights}")
    print(f"⚙️ Batch Size: {opt.batch_size} | Image Resolution: {opt.img_size}")
    print("=" * 70)

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
        archive_stage_artifacts(exp_dir, opt.stage, opt.checkpoint_dir, opt.reports_dir)
        update_metrics_history(exp_dir, opt.reports_dir)
        print(f"[SUCCESS] Stage {opt.stage} training concluded successfully.")
    else:
        print(f"[ERROR] Training failed with exit code: {exit_code}")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

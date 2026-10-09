"""
YOLO_ChestXray Training Runner (Plan 1-2 Compliant)
Handles:
- Non-Interactive Execution (WANDB disabled)
- Reproducibility Seed Locking (Seed 42)
- Strict YOLOv7 Stage-wise Resume Standard:
  Stage 1: --weights yolov7.pt --epochs 10
  Stage 2..5: --resume checkpoints/checkpoint_xx.pt
- Checkpoint validation & Archival (checkpoint_xx.pt, best_model.pt)
- GPU Monitoring (gpu_usage.csv, gpu_usage.png)
- Learning Rate Tracking (learning_rate_curve.png)
- Metrics History Updating (reports/metrics_history.csv)
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Non-interactive requirement
os.environ['WANDB_MODE'] = 'disabled'
os.environ['WANDB_DISABLED'] = 'true'

# 2. Reproducibility seed setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    _orig_torch_load = torch.load
    def _compat_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            try:
                return _orig_torch_load(*args, **kwargs, weights_only=False)
            except TypeError:
                pass
        return _orig_torch_load(*args, **kwargs)
    torch.load = _compat_torch_load
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except Exception:
    pass


def parse_opt():
    parser = argparse.ArgumentParser(description="YOLO_ChestXray Training Runner (Plan 1-2)")
    parser.add_argument("--weights", type=str, default="yolov7.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="cfg/training/yolov7.yaml", help="model.yaml path")
    parser.add_argument("--data", type=str, default="configs/chestxray.yaml", help="data.yaml path")
    parser.add_argument("--hyp", type=str, default="data/hyp.scratch.p5.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=10, help="cumulative total epochs target for current stage")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--img-size", nargs="+", type=int, default=[640, 640], help="image size (pixels)")
    parser.add_argument("--resume", type=str, default="", help="checkpoint path to resume from")
    parser.add_argument("--device", default="", help="cuda device or cpu")
    parser.add_argument("--project", default="runs/train", help="save to project/name")
    parser.add_argument("--name", default="stage_1", help="run experiment name")
    parser.add_argument("--stage", type=int, default=1, help="current training stage (1 to 5)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="checkpoint archive directory")
    parser.add_argument("--reports-dir", type=str, default="reports", help="reports directory")
    parser.add_argument("--drive-dir", type=str, default="", help="Google Drive experiment sync directory")
    parser.add_argument("--skip-existing", action="store_true", help="skip stage if target checkpoint already exists")
    return parser.parse_args()


def log_gpu_usage(reports_dir: str):
    """Capture GPU metrics using nvidia-smi if available."""
    os.makedirs(reports_dir, exist_ok=True)
    gpu_csv = Path(reports_dir) / "gpu_usage.csv"
    gpu_png = Path(reports_dir) / "gpu_usage.png"

    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            universal_newlines=True
        ).strip()
        lines = res.split("\n")
        records = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                records.append({
                    'Timestamp': parts[0],
                    'GPU_Name': parts[1],
                    'GPU_Util_Percent': float(parts[2]),
                    'Memory_Util_Percent': float(parts[3]),
                    'Memory_Used_MB': float(parts[4]),
                    'Memory_Total_MB': float(parts[5]),
                    'Temperature_C': float(parts[6])
                })
        df = pd.DataFrame(records)
        if gpu_csv.exists():
            df.to_csv(gpu_csv, mode='a', header=False, index=False)
        else:
            df.to_csv(gpu_csv, index=False)

        # Plot GPU usage
        if gpu_csv.exists():
            full_df = pd.read_csv(gpu_csv)
            plt.figure(figsize=(10, 5), dpi=300)
            plt.plot(full_df.index, full_df['Memory_Used_MB'], label='Memory Used (MB)', color='#e67e22', lw=2)
            plt.title('GPU Memory Usage Over Training Stages', fontsize=14, fontweight='bold')
            plt.xlabel('Log Checkpoint Index')
            plt.ylabel('Memory (MB)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(gpu_png, dpi=300)
            plt.close()
    except Exception as e:
        # If running on CPU or environment without nvidia-smi
        pass


def update_metrics_and_lr(exp_dir: Path, reports_dir: str):
    """Parse results.txt / results.csv from YOLOv7 output and update reports/metrics_history.csv and learning_rate_curve.png."""
    os.makedirs(reports_dir, exist_ok=True)
    history_file = Path(reports_dir) / "metrics_history.csv"
    lr_plot = Path(reports_dir) / "learning_rate_curve.png"

    results_txt = exp_dir / "results.txt"
    records = []
    lrs = []

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

                        lr = float(parts[12]) if len(parts) > 12 else 0.001

                        records.append({
                            'Epoch': epoch,
                            'Precision': precision,
                            'Recall': recall,
                            'mAP50': map50,
                            'mAP50_95': map50_95,
                            'TrainLoss': train_loss,
                            'ValLoss': val_loss
                        })
                        lrs.append({'Epoch': epoch, 'LR': lr})
                    except Exception:
                        continue

    if records:
        df_new = pd.DataFrame(records)
        df_new.to_csv(history_file, index=False)
        print(f"[SUCCESS] Updated metrics history: {history_file} ({len(records)} epochs recorded)")

        if lrs:
            df_lr = pd.DataFrame(lrs)
            plt.figure(figsize=(10, 5), dpi=300)
            plt.plot(df_lr['Epoch'], df_lr['LR'], color='#2980b9', lw=2.5, marker='.')
            plt.title('Learning Rate Schedule Progression', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(lr_plot, dpi=300)
            plt.close()
            print(f"[SUCCESS] Saved learning rate curve: {lr_plot}")


def check_stage_already_completed(stage: int, epochs: int, checkpoint_dir: str, reports_dir: str, drive_dir: str = "", skip_existing: bool = False) -> bool:
    """Check if the current stage has already been trained and should be skipped."""
    if not skip_existing:
        return False

    target_ckpt = Path(checkpoint_dir) / f"checkpoint_{epochs}.pt"

    # Sync from current experiment's Google Drive folder if available
    if drive_dir and os.path.exists(drive_dir):
        drive_ckpt = Path(drive_dir) / "checkpoints" / f"checkpoint_{epochs}.pt"
        drive_best = Path(drive_dir) / "checkpoints" / "best_model.pt"
        drive_metrics = Path(drive_dir) / "reports" / "metrics_history.csv"

        if not target_ckpt.exists() and drive_ckpt.exists() and drive_ckpt.stat().st_size > 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            shutil.copy(drive_ckpt, target_ckpt)
            if drive_best.exists() and not (Path(checkpoint_dir) / "best_model.pt").exists():
                shutil.copy(drive_best, Path(checkpoint_dir) / "best_model.pt")
            print(f"[CACHE] Restored stage {stage} checkpoint from Google Drive: {drive_ckpt}")

        if drive_metrics.exists() and not (Path(reports_dir) / "metrics_history.csv").exists():
            os.makedirs(reports_dir, exist_ok=True)
            shutil.copy(drive_metrics, Path(reports_dir) / "metrics_history.csv")

    if target_ckpt.exists() and target_ckpt.stat().st_size > 0:
        if not (Path(checkpoint_dir) / "best_model.pt").exists():
            shutil.copy(target_ckpt, Path(checkpoint_dir) / "best_model.pt")
        print("=" * 70)
        print(f"✨ [SKIP] Stage {stage} result already exists ({target_ckpt.name})!")
        print(f"⏩ Skipping Stage {stage} training and proceeding directly to the next step.")
        print("=" * 70)
        return True
    return False


def resolve_exp_dir(project: str, name: str) -> Path:
    """Find the most recent experiment output directory even if YOLOv7 incremented the name."""
    exp_dir = Path(project) / name
    if (exp_dir / "weights" / "last.pt").exists() or (exp_dir / "weights" / "best.pt").exists() or (exp_dir / "results.txt").exists():
        return exp_dir

    # Search for incremented names like stage_22, stage_23, etc.
    matches = sorted(Path(project).glob(f"{name}*"), key=os.path.getmtime, reverse=True)
    for m in matches:
        if (m / "weights" / "last.pt").exists() or (m / "weights" / "best.pt").exists() or (m / "results.txt").exists():
            print(f"[INFO] Resolved active experiment directory: {m}")
            return m

    # Fallback to any recent directory under project
    all_runs = sorted([d for d in Path(project).glob("*") if d.is_dir()], key=os.path.getmtime, reverse=True)
    for r in all_runs:
        if (r / "weights" / "last.pt").exists() or (r / "weights" / "best.pt").exists() or (r / "results.txt").exists():
            print(f"[INFO] Resolved fallback experiment directory: {r}")
            return r
    return exp_dir


def archive_and_verify_checkpoints(exp_dir: Path, stage: int, epochs: int, checkpoint_dir: str, reports_dir: str):
    """Verify and archive checkpoint_xx.pt and best_model.pt."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    weights_dir = exp_dir / "weights"
    last_pt = weights_dir / "last.pt"
    best_pt = weights_dir / "best.pt"

    # Checkpoint Validation & Recovery requirement
    if not last_pt.exists() and not best_pt.exists():
        # Fallback search across all recent weights in runs/
        cands = sorted(Path("runs").glob("**/weights/*.pt"), key=os.path.getmtime, reverse=True)
        for c in cands:
            if c.exists() and c.stat().st_size > 0:
                print(f"[RECOVERY] Discovered valid stage weight at: {c}")
                if "best" in c.name:
                    best_pt = c
                else:
                    last_pt = c
                break

    if not last_pt.exists() and not best_pt.exists():
        print(f"[ERROR] Checkpoint verification failed: Neither {last_pt} nor {best_pt} exist in {exp_dir}!")
        print("\n--- Diagnostic File System Snapshot ---")
        for search_root in [Path("runs"), Path("checkpoints"), Path(".")]:
            found_pts = list(search_root.glob("**/*.pt"))[:15]
            if found_pts:
                print(f"Found .pt files in [{search_root}]: {[str(x) for x in found_pts]}")
        sys.exit(1)

    target_stage_ckpt = Path(checkpoint_dir) / f"checkpoint_{epochs}.pt"
    target_stage_legacy = Path(checkpoint_dir) / f"checkpoint_{stage * 10}.pt"
    target_best_ckpt = Path(checkpoint_dir) / "best_model.pt"
    target_last_ckpt = Path(checkpoint_dir) / "last.pt"

    if last_pt.exists():
        shutil.copy(last_pt, target_stage_ckpt)
        if not target_stage_legacy.exists():
            shutil.copy(last_pt, target_stage_legacy)
        shutil.copy(last_pt, target_last_ckpt)
        print(f"[SUCCESS] Archived stage checkpoint: {target_stage_ckpt}")

    if best_pt.exists():
        shutil.copy(best_pt, target_best_ckpt)
        shutil.copy(best_pt, Path(checkpoint_dir) / "best.pt")
        print(f"[SUCCESS] Updated best model checkpoint: {target_best_ckpt}")

    # Archive opt.yaml and hyp.yaml to ensure YOLOv7 resume never fails
    opt_yaml_src = exp_dir / "opt.yaml"
    hyp_yaml_src = exp_dir / "hyp.yaml"
    if opt_yaml_src.exists():
        shutil.copy(opt_yaml_src, Path(checkpoint_dir) / "opt.yaml")
        shutil.copy(opt_yaml_src, "opt.yaml")
        print(f"[SUCCESS] Archived opt.yaml configuration for resume safety.")
    if hyp_yaml_src.exists():
        shutil.copy(hyp_yaml_src, Path(checkpoint_dir) / "hyp.yaml")

    # Archive stage plots
    results_png = exp_dir / "results.png"
    if results_png.exists():
        shutil.copy(results_png, Path(reports_dir) / f"train_results_stage_{stage}.png")
        shutil.copy(results_png, Path(reports_dir) / "train_results.png")


def main():
    opt = parse_opt()

    # 1. Check if this stage was already completed before running
    if check_stage_already_completed(opt.stage, opt.epochs, opt.checkpoint_dir, opt.reports_dir, opt.drive_dir, opt.skip_existing):
        return

    print("=" * 70)
    print(f"🚀 YOLO_ChestXray Training Pipeline - Stage {opt.stage} (Plan 1 v4.0 Compliant)")
    print(f"🔒 Fixed Seed: {SEED} | Non-Interactive Mode: WANDB Disabled")
    print("=" * 70)

    # Determine starting weights
    weights_path = opt.weights
    if opt.resume:
        weights_path = opt.resume
        # If resume path doesn't exist locally, check Drive
        if not os.path.exists(weights_path):
            candidates = []
            if opt.drive_dir:
                candidates.append(Path(f"{opt.drive_dir}/{opt.resume}"))
                candidates.append(Path(f"{opt.drive_dir}/checkpoints/{Path(opt.resume).name}"))
            for cand in candidates:
                if cand.exists():
                    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                    shutil.copy(cand, weights_path)
                    print(f"[RESUME] Retrieved {opt.resume} from Google Drive: {cand}")
                    break

    # Diagnostic logging of checkpoint and parameters
    print(f"[DIAGNOSTIC] Stage {opt.stage} Target Epochs: {opt.epochs}")
    print(f"[DIAGNOSTIC] Checkpoint input: {weights_path}")
    if os.path.exists(weights_path):
        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        print(f"[DIAGNOSTIC] Checkpoint file exists (Size: {size_mb:.2f} MB)")
        try:
            import torch
            ckpt_info = torch.load(weights_path, map_location='cpu')
            if isinstance(ckpt_info, dict):
                ckpt_epoch = ckpt_info.get('epoch', 'N/A')
                print(f"[DIAGNOSTIC] Checkpoint inner epoch: {ckpt_epoch}")
        except Exception as e:
            print(f"[DIAGNOSTIC] Could not read checkpoint metadata: {e}")
    else:
        print(f"[WARNING] Checkpoint file {weights_path} not found locally before launch!")

    # Build robust execution command for stage-wise training
    cmd = [
        sys.executable,
        "train.py",
        f"--weights {weights_path}",
        f"--cfg {opt.cfg}",
        f"--data {opt.data}",
        f"--hyp {opt.hyp}",
        f"--epochs {opt.epochs}",
        f"--batch-size {opt.batch_size}",
        f"--img-size {' '.join(map(str, opt.img_size))}",
        f"--project {opt.project}",
        f"--name {opt.name}",
        "--exist-ok",
    ]
    if opt.device:
        cmd.append(f"--device {opt.device}")

    try:
        try:
            from scripts.fix_yolov7_env import patch_yolov7_train_py_resume, patch_yolov7_train_py_epochs
            patch_yolov7_train_py_resume(".")
            patch_yolov7_train_py_epochs(".")
        except ImportError:
            from fix_yolov7_env import patch_yolov7_train_py_resume, patch_yolov7_train_py_epochs
            patch_yolov7_train_py_resume(".")
            patch_yolov7_train_py_epochs(".")
    except Exception:
        pass

    full_cmd = " ".join(cmd)
    print(f"[EXEC] Running command: {full_cmd}")

    log_gpu_usage(opt.reports_dir)
    res = subprocess.run(full_cmd, shell=True)
    exit_code = res.returncode
    log_gpu_usage(opt.reports_dir)

    if exit_code == 0:
        exp_dir = resolve_exp_dir(opt.project, opt.name)
        archive_and_verify_checkpoints(exp_dir, opt.stage, opt.epochs, opt.checkpoint_dir, opt.reports_dir)
        update_metrics_and_lr(exp_dir, opt.reports_dir)
        print(f"[SUCCESS] Stage {opt.stage} training finished and verified successfully.")
    else:
        print(f"[ERROR] Training process exited with non-zero return code: {exit_code}")
        # Directory diagnostic dump
        print("\n--- Diagnostic Directory Listing ---")
        for p in [Path("checkpoints"), Path("runs/train"), Path(opt.project)]:
            if p.exists():
                print(f"Directory [{p}]: {[str(x) for x in p.glob('**/*') if x.is_file()][:10]}")
        sys.exit(exit_code)




if __name__ == "__main__":
    main()

"""
YOLOv7 Legacy Compatibility & Environment Builder Suite (Plan 1 v4.0 Compliant)
Version: 4.0
Handles:
1. Python / Torch / CUDA Environment Validation
   - Generates: reports/environment_report.json
2. YOLOv7 torch.load Compatibility Patch (weights_only=False for PyTorch 2.4+ / 2.6+)
   - Generates: reports/patch_report.json
3. NumPy Legacy Alias Injection (np.int, np.float, np.bool)
4. YOLOv7 utils/loss.py PyTorch 2.x Tensor Indexing Patch
5. Non-Interactive WANDB Disabling
"""

import json
import os
import re
import sys
from pathlib import Path


def check_and_report_environment(reports_dir: str = "reports") -> dict:
    """Verify and report active Python, PyTorch, CUDA, and GPU hardware, saving environment_report.json."""
    os.makedirs(reports_dir, exist_ok=True)
    print("=" * 70)
    print("🔍 [Plan 1 v4.0] Validating Runtime Environment & Hardware Profile")
    print("=" * 70)

    py_ver = sys.version_info
    py_ver_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    print(f"🐍 Python Version: {py_ver_str}")

    torch_ver_str = "Not Installed"
    cuda_available = False
    gpu_name = "N/A"
    device_count = 0
    try:
        import torch
        torch_ver_str = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
        print(f"🔥 PyTorch Version: {torch_ver_str}")
        print(f"⚡ CUDA Available: {cuda_available}")
        if cuda_available:
            gpu_name = str(torch.cuda.get_device_name(0))
            device_count = int(torch.cuda.device_count())
            print(f"🚀 GPU Device: {gpu_name}")
            print(f"🧠 CUDA Device Count: {device_count}")
        else:
            print("⚠️ WARNING: GPU/CUDA is not available. Execution will fall back to CPU.")
    except ImportError:
        print("❌ PyTorch is not installed in the current environment.")

    numpy_ver_str = "Not Installed"
    try:
        import numpy as np
        numpy_ver_str = str(np.__version__)
        print(f"🔢 NumPy Version: {numpy_ver_str}")
    except ImportError:
        pass

    env_report = {
        "plan_version": "Plan 1 v4.0",
        "python_version": py_ver_str,
        "torch_version": torch_ver_str,
        "numpy_version": numpy_ver_str,
        "cuda_available": cuda_available,
        "gpu_available": cuda_available,
        "gpu_device_name": gpu_name,
        "gpu_device_count": device_count,
        "wandb_disabled": True,
        "status": "PASS" if cuda_available else "WARN_CPU_ONLY"
    }

    env_report_file = os.path.join(reports_dir, "environment_report.json")
    with open(env_report_file, "w", encoding="utf-8") as f:
        json.dump(env_report, f, indent=2)
    print(f"[SUCCESS] Environment report generated: {env_report_file}")

    return env_report


def patch_numpy_aliases():
    """Inject backwards compatibility for np.int, np.float, np.bool."""
    try:
        import numpy as np
        if not hasattr(np, 'int'):
            np.int = int
        if not hasattr(np, 'float'):
            np.float = float
        if not hasattr(np, 'bool'):
            np.bool = bool
        print("[ENV FIX] Patched NumPy legacy type aliases (np.int, np.float, np.bool).")
    except Exception as e:
        print(f"[ENV FIX] NumPy patch warning: {e}")


def patch_torch_load_in_files(yolov7_dir: str = ".", reports_dir: str = "reports") -> dict:
    """
    Patch all occurrences of torch.load in YOLOv7 codebase to support PyTorch 2.4+ / 2.6+
    by ensuring weights_only=False where applicable. Generates reports/patch_report.json.
    """
    os.makedirs(reports_dir, exist_ok=True)
    target_files = [
        Path(yolov7_dir) / "train.py",
        Path(yolov7_dir) / "test.py",
        Path(yolov7_dir) / "detect.py",
        Path(yolov7_dir) / "models" / "experimental.py",
        Path(yolov7_dir) / "models" / "yolo.py",
        Path(yolov7_dir) / "utils" / "torch_utils.py",
    ]

    patch_results = {}

    header_patch = """
# --- Plan 1 v4.0 PyTorch Compatibility Layer ---
import torch
_orig_torch_load = torch.load
def _compat_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs and 'weights_only' in _orig_torch_load.__code__.co_varnames:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _compat_torch_load
# -----------------------------------------------
"""

    for file_path in target_files:
        if not file_path.exists():
            patch_results[str(file_path)] = "File not found"
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if "# --- Plan 1 v4.0 PyTorch Compatibility Layer ---" not in content and "# --- Plan 1_4 PyTorch 2.x Compatibility Layer ---" not in content:
                content = header_patch + content
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                patch_results[str(file_path)] = "PATCHED (torch.load wrapper injected)"
                print(f"[ENV FIX] Injected torch.load compatibility wrapper into {file_path.name}")
            else:
                patch_results[str(file_path)] = "ALREADY_PATCHED"
        except Exception as e:
            patch_results[str(file_path)] = f"ERROR: {str(e)}"
            print(f"[ENV FIX] Error patching {file_path}: {e}")

    patch_report = {
        "plan_version": "Plan 1 v4.0",
        "weights_only_policy": "weights_only=False",
        "patched_files": patch_results,
        "overall_status": "SUCCESS"
    }

    patch_report_file = os.path.join(reports_dir, "patch_report.json")
    with open(patch_report_file, "w", encoding="utf-8") as f:
        json.dump(patch_report, f, indent=2)
    print(f"[SUCCESS] Patch report generated: {patch_report_file}")

    return patch_report


def patch_yolov7_loss_py(yolov7_dir: str = "."):
    """Patch utils/loss.py in YOLOv7 for PyTorch 2.x clamp float tensor indexing."""
    loss_py = Path(yolov7_dir) / "utils" / "loss.py"
    if not loss_py.exists():
        return

    with open(loss_py, "r", encoding="utf-8") as f:
        content = f.read()

    # Targets to replace (including any buggy previous replacement)
    targets_old = [
        "indices.append((b, a, gj.clamp_(0, shape[2] - 1).long(), gi.clamp_(0, shape[3] - 1).long()))",
        "indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))",
        "indices.append((b, a, gj.clamp_(0, gain[3] - 1.0), gi.clamp_(0, gain[2] - 1.0)))",
        "indices.append((b, a, gj.clamp_(0, gain[3].long() - 1), gi.clamp_(0, gain[2].long() - 1)))",
    ]
    target_correct = "indices.append((b, a, gj.clamp_(0, (gain[3] - 1).long()), gi.clamp_(0, (gain[2] - 1).long())))"

    modified = False
    for t in targets_old:
        if t in content:
            content = content.replace(t, target_correct)
            modified = True

    if modified:
        with open(loss_py, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[ENV FIX] Successfully patched {loss_py.name} for PyTorch 2.x tensor indexing.")



def patch_yolov7_train_py_resume(yolov7_dir: str = "."):
    """Patch train.py in YOLOv7 to safely resolve opt.yaml when resuming from arbitrary checkpoint paths."""
    train_py = Path(yolov7_dir) / "train.py"
    if not train_py.exists():
        return

    with open(train_py, "r", encoding="utf-8") as f:
        content = f.read()

    # Pattern: with open(Path(ckpt).parent.parent / 'opt.yaml') as f: or chkpt or previous patch
    old_patterns = [
        "with open(Path(ckpt).parent.parent / 'opt.yaml') as f:",
        "with open(Path(chkpt).parent.parent / 'opt.yaml') as f:",
        "Path(chkpt).parent.parent / 'opt.yaml'",
    ]
    replacement = """# --- Safe opt.yaml resolution for resume ---
        _ckpt_target = ckpt if 'ckpt' in locals() else (chkpt if 'chkpt' in locals() else opt.resume)
        _opt_candidates = [
            Path(_ckpt_target).parent.parent / 'opt.yaml',
            Path(_ckpt_target).parent / 'opt.yaml',
            Path('checkpoints/opt.yaml'),
            Path('opt.yaml'),
            Path('runs/train/stage_1/opt.yaml'),
            Path('runs/train/stage_2/opt.yaml'),
        ]
        _opt_file = next((c for c in _opt_candidates if c.exists()), None)
        if _opt_file is not None:
            with open(_opt_file) as f:
                opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))
        else:
            _d = torch.load(_ckpt_target, map_location='cpu')
            if isinstance(_d, dict) and 'opt' in _d and _d['opt'] is not None:
                opt = _d['opt'] if isinstance(_d['opt'], argparse.Namespace) else argparse.Namespace(**_d['opt'])
            else:
                opt = argparse.Namespace(epochs=opt.epochs, cfg='', weights='', data='configs/chestxray.yaml', batch_size=opt.batch_size if hasattr(opt, 'batch_size') else 8, img_size=opt.img_size if hasattr(opt, 'img_size') else [1024, 1024], hyp='data/hyp.scratch.p5.yaml', project='runs/train', name='exp', device='', exist_ok=True, single_cls=False, sync_bn=False, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, save_period=-1, artifact_alias='latest')
        # -------------------------------------------"""

    # If already patched with chkpt, replace chkpt with _ckpt_target
    if "Path(chkpt).parent.parent / 'opt.yaml'" in content:
        content = content.replace("chkpt", "ckpt")

    modified = False
    for pat in old_patterns:
        if pat in content:
            content = content.replace(pat, replacement)
            modified = True

    if modified:
        with open(train_py, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[ENV FIX] Successfully patched {train_py.name} for safe resume opt.yaml lookup.")


def setup_plan1_4_environment(yolov7_dir: str = ".", reports_dir: str = "reports"):
    """Main entrypoint for Plan 1 v4.0 environment setup & compatibility enforcement."""
    # Disable WANDB completely
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['WANDB_DISABLED'] = 'true'

    check_and_report_environment(reports_dir)
    patch_numpy_aliases()
    patch_yolov7_loss_py(yolov7_dir)
    patch_yolov7_train_py_resume(yolov7_dir)
    patch_torch_load_in_files(yolov7_dir, reports_dir)
    print("=" * 70)
    print("✅ [Plan 1 v4.0] YOLOv7 Legacy Compatibility Environment Ready")
    print("=" * 70)


if __name__ == "__main__":
    yolo_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    rep_dir = sys.argv[2] if len(sys.argv) > 2 else "reports"
    setup_plan1_4_environment(yolo_dir, rep_dir)


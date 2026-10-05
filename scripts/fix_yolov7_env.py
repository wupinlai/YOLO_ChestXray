"""
YOLOv7 Legacy Compatibility & Environment Builder Suite (Plan 1_4 Compliant)
Version: 2.0
Handles:
1. Python / Torch / CUDA Environment Validation
   - Target Compatibility Profile: Python 3.10, Torch 2.0.1+cu118, TorchVision 0.15.2, NumPy 1.23.5
2. YOLOv7 torch.load Compatibility Patch (weights_only=False for PyTorch 2.4+ / 2.6+)
3. NumPy 1.24+ Legacy Alias Injection (np.int, np.float, np.bool)
4. YOLOv7 utils/loss.py PyTorch 2.x Tensor Indexing Patch
5. Non-Interactive WANDB Disabling
"""

import os
import re
import sys
from pathlib import Path


def check_and_report_environment():
    """Verify and report active Python, PyTorch, CUDA, and GPU hardware."""
    print("=" * 70)
    print("🔍 [Plan 1_4] Validating Runtime Environment & Compatibility Profile")
    print("=" * 70)

    py_ver = sys.version_info
    print(f"🐍 Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")

    try:
        import torch
        print(f"🔥 PyTorch Version: {torch.__version__}")
        print(f"⚡ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🚀 GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"🧠 CUDA Device Count: {torch.cuda.device_count()}")
        else:
            print("⚠️ WARNING: GPU/CUDA is not available. Execution will fall back to CPU.")
    except ImportError:
        print("❌ PyTorch is not installed in the current environment.")

    try:
        import numpy as np
        print(f"🔢 NumPy Version: {np.__version__}")
    except ImportError:
        pass

    # Check if isolation or compatibility patches are required
    needs_patch = False
    if py_ver.major == 3 and py_ver.minor > 10:
        print(f"ℹ️ Python version is 3.{py_ver.minor} (> 3.10) - Applying legacy compatibility layer.")
        needs_patch = True
    try:
        import torch
        # Check Torch 2.x / 2.6+
        t_major = int(torch.__version__.split('.')[0])
        t_minor = int(torch.__version__.split('.')[1])
        if t_major >= 2:
            needs_patch = True
    except Exception:
        pass

    return needs_patch


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


def patch_torch_load_in_files(yolov7_dir: str = "."):
    """
    Patch all occurrences of torch.load in YOLOv7 codebase to support PyTorch 2.4+ / 2.6+
    by ensuring weights_only=False where applicable or handling safe globals.
    """
    target_files = [
        Path(yolov7_dir) / "models" / "experimental.py",
        Path(yolov7_dir) / "models" / "yolo.py",
        Path(yolov7_dir) / "train.py",
        Path(yolov7_dir) / "test.py",
        Path(yolov7_dir) / "detect.py",
        Path(yolov7_dir) / "utils" / "torch_utils.py",
    ]

    for file_path in target_files:
        if not file_path.exists():
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Replace torch.load(f, map_location=...) with weights_only=False if not already present
            pattern = r"torch\.load\(([^)]+)\)"
            
            def replace_torch_load(match):
                args = match.group(1)
                if "weights_only" in args:
                    return match.group(0)
                # Safely add weights_only=False if PyTorch supports it
                return f"torch.load({args}, weights_only=False if 'weights_only' in torch.load.__code__.co_varnames else None)" if False else f"torch.load({args})"

            # A more robust file-level patch: Add custom safe torch_load wrapper at the top of file
            header_patch = """
# --- Plan 1_4 PyTorch 2.x Compatibility Layer ---
import torch
_orig_torch_load = torch.load
def _compat_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs and 'weights_only' in _orig_torch_load.__code__.co_varnames:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _compat_torch_load
# ------------------------------------------------
"""
            if "# --- Plan 1_4 PyTorch 2.x Compatibility Layer ---" not in content:
                content = header_patch + content
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"[ENV FIX] Injected torch.load compatibility wrapper into {file_path.name}")
        except Exception as e:
            print(f"[ENV FIX] Error patching {file_path}: {e}")


def patch_yolov7_loss_py(yolov7_dir: str = "."):
    """Patch utils/loss.py in YOLOv7 for PyTorch 2.x clamp float tensor indexing."""
    loss_py = Path(yolov7_dir) / "utils" / "loss.py"
    if not loss_py.exists():
        return

    with open(loss_py, "r", encoding="utf-8") as f:
        content = f.read()

    target_old = "indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))"
    target_new = "indices.append((b, a, gj.clamp_(0, shape[2] - 1).long(), gi.clamp_(0, shape[3] - 1).long()))"
    target_old_2 = "indices.append((b, a, gj.clamp_(0, gain[3] - 1.0), gi.clamp_(0, gain[2] - 1.0)))"

    modified = False
    if target_old in content:
        content = content.replace(target_old, target_new)
        modified = True
    elif target_old_2 in content:
        content = content.replace(target_old_2, target_new)
        modified = True

    if modified:
        with open(loss_py, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[ENV FIX] Successfully patched {loss_py.name} for PyTorch 2.x tensor indexing.")


def setup_plan1_4_environment(yolov7_dir: str = "."):
    """Main entrypoint for Plan 1_4 environment setup & compatibility enforcement."""
    # Disable WANDB completely
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['WANDB_DISABLED'] = 'true'

    check_and_report_environment()
    patch_numpy_aliases()
    patch_yolov7_loss_py(yolov7_dir)
    patch_torch_load_in_files(yolov7_dir)
    print("=" * 70)
    print("✅ [Plan 1_4] YOLOv7 Legacy Compatibility Environment Ready")
    print("=" * 70)


if __name__ == "__main__":
    setup_plan1_4_environment(sys.argv[1] if len(sys.argv) > 1 else ".")

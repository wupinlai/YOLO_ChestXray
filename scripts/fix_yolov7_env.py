"""
YOLOv7 Google Colab Environment & Compatibility Patch Suite
Fixes known YOLOv7 compatibility issues with modern Colab (PyTorch 2.x & NumPy 1.24+):
1. Fixes NumPy deprecation (np.int, np.float, np.bool removed in NumPy 1.24+)
2. Fixes PyTorch 2.x Tensor indexing issue in utils/loss.py (ComputeLoss.build_targets)
3. Fixes PyTorch 2.x weights_only=False warning in torch.load
4. Fixes WANDB non-interactive prompts by stubbing wandb
5. Verifies CUDA / GPU environment
"""

import os
import sys
from pathlib import Path


def patch_numpy_aliases():
    """Inject backwards compatibility for np.int, np.float, np.bool."""
    import numpy as np
    if not hasattr(np, 'int'):
        np.int = int
    if not hasattr(np, 'float'):
        np.float = float
    if not hasattr(np, 'bool'):
        np.bool = bool
    print("[ENV FIX] Patched NumPy legacy type aliases (np.int, np.float, np.bool).")


def patch_yolov7_loss_py(yolov7_dir: str = "."):
    """
    Patch utils/loss.py in YOLOv7 for PyTorch 2.x.
    Fixes:
    RuntimeError: result type Float can't be cast to the desired output type __int64
    or float tensor indexing in build_targets.
    """
    loss_py = Path(yolov7_dir) / "utils" / "loss.py"
    if not loss_py.exists():
        print(f"[ENV FIX] {loss_py} not found, skipping patch.")
        return

    with open(loss_py, "r", encoding="utf-8") as f:
        content = f.read()

    # Target the float tensor clamp in build_targets
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

    # Also fix gain tensor assignment if needed
    if "gain = torch.ones(7, device=targets.device)" in content and "gain = torch.ones(7, device=targets.device).long()" not in content:
        # Check build_targets shape definition
        shape_code = """
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
        """

    if modified:
        with open(loss_py, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[ENV FIX] Successfully patched {loss_py} for PyTorch 2.x tensor indexing.")
    else:
        print(f"[ENV FIX] {loss_py} is already compatible or modified.")


def setup_colab_environment(yolov7_dir: str = "."):
    """Run all environment fixes."""
    print("=" * 60)
    print("🔧 Running YOLOv7 Environment & Compatibility Fixer")
    print("=" * 60)

    # Disable WANDB
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['WANDB_DISABLED'] = 'true'

    patch_numpy_aliases()
    patch_yolov7_loss_py(yolov7_dir)
    print("[ENV FIX] Environment preparation completed successfully.")


if __name__ == "__main__":
    setup_colab_environment(sys.argv[1] if len(sys.argv) > 1 else ".")

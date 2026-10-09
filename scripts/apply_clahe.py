"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) Preprocessor for Chest X-Ray Images
Enhances local tissue contrast for subtle pulmonary lesions (Nodules, Infiltrations, Fibrosis).
Default Parameters:
  --clip-limit 2.0
  --tile-grid-size 8 8
"""

import argparse
import os
import glob
from pathlib import Path
from typing import Tuple
from PIL import Image
import numpy as np


def apply_clahe_to_image(img_path: Path, out_path: Path, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
    """Apply CLAHE to a single image using OpenCV if available, or fallback to high-quality PIL equalization."""
    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        
        # Convert to LAB color space or Grayscale to equalize Luminance channel
        if len(img.shape) == 2 or img.shape[2] == 1:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced = clahe.apply(img)
        else:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l_enhanced = clahe.apply(l)
            enhanced_lab = cv2.merge((l_enhanced, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), enhanced)
        return True
    except Exception as e:
        # Fallback to PIL ImageOps
        try:
            from PIL import ImageOps
            img = Image.open(img_path).convert('RGB')
            enhanced = ImageOps.autocontrast(img, cutoff=2)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            enhanced.save(out_path, quality=95)
            return True
        except Exception:
            return False


def main():
    parser = argparse.ArgumentParser(description="Apply CLAHE Enhancement to Chest X-ray Dataset")
    parser.add_argument("--src-dir", type=str, required=True, help="source images directory")
    parser.add_argument("--dst-dir", type=str, required=True, help="destination images directory")
    parser.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE clip limit (default: 2.0)")
    parser.add_argument("--tile-grid-size", nargs=2, type=int, default=[8, 8], help="CLAHE tile grid size (default: 8 8)")
    args = parser.parse_args()

    src_p = Path(args.src_dir)
    dst_p = Path(args.dst_dir)
    dst_p.mkdir(parents=True, exist_ok=True)

    img_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"]
    img_files = []
    for ext in img_extensions:
        img_files.extend(list(src_p.glob(ext)))

    print(f"🔬 Applying CLAHE Enhancement: clipLimit={args.clip_limit}, tileGridSize={tuple(args.tile_grid_size)}")
    print(f"📁 Processing {len(img_files)} images from {src_p} -> {dst_p}")

    success_cnt = 0
    for idx, img_f in enumerate(img_files):
        out_f = dst_p / img_f.name
        if apply_clahe_to_image(img_f, out_f, clip_limit=args.clip_limit, tile_grid_size=tuple(args.tile_grid_size)):
            success_cnt += 1
        if (idx + 1) % 500 == 0 or (idx + 1) == len(img_files):
            print(f"[{idx+1}/{len(img_files)}] Processed {success_cnt} images...")

    print(f"✨ [DONE] Successfully processed {success_cnt}/{len(img_files)} images.")


if __name__ == "__main__":
    main()

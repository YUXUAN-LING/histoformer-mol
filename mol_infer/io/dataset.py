# mol_infer/io/dataset.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

# Public type expected by mol_infer.io.__init__
ImagePair = Tuple[str, Optional[str]]

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def build_pairs(input_path: str, pair_list: Optional[str] = None, gt_root: Optional[str] = None) -> List[ImagePair]:
    """
    Return list of (lq_path, gt_path_or_None).
    - If pair_list provided: parse "lq [gt]" per line.
    - Else:
        * if input_path is file: [(input_path, None)]
        * if input_path is dir: enumerate images -> [(img, None)]
    """
    if pair_list:
        pairs: List[ImagePair] = []
        with open(pair_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 1:
                    lq, gt = parts[0], None
                else:
                    lq, gt = parts[0], parts[1]
                if gt_root and gt is not None and (not os.path.isabs(gt)):
                    gt = os.path.join(gt_root, gt)
                pairs.append((lq, gt))
        return pairs

    p = Path(input_path)
    if p.is_file():
        return [(str(p), None)]

    if not p.is_dir():
        raise FileNotFoundError(f"[dataset] input not found: {input_path}")

    files: List[str] = []
    for fn in sorted(os.listdir(str(p))):
        if fn.lower().endswith(_IMG_EXTS):
            files.append(str(p / fn))
    return [(fp, None) for fp in files]


def load_image_tensor(path: str, device: Optional[str] = None) -> torch.Tensor:
    """
    Load RGB image -> torch tensor in [0,1], shape [1,3,H,W].
    """
    if path is None:
        raise ValueError("path is None")
    img = Image.open(path).convert("RGB")
    t = T.ToTensor()(img).unsqueeze(0)  # [1,3,H,W]
    if device:
        t = t.to(device)
    return t


__all__ = ["ImagePair", "build_pairs", "load_image_tensor"]

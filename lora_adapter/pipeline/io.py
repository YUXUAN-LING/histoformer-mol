# -*- coding: utf-8 -*-
"""lora_adapters.pipeline.io

I/O utilities:
- read RGB image -> torch tensor in [0,1]
- save tensor -> png
- save triplet concat (lq | base | output)

Keep this file small; you can replace with your own dataset/dataloader later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image


def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def pil_to_tensor(img: Image.Image, device: str = "cpu") -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def read_image(path: Union[str, Path], device: str = "cpu") -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return pil_to_tensor(img, device=device)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    # expects [1,C,H,W] or [C,H,W]
    if t.ndim == 4:
        t = t[0]
    t = t.detach().clamp(0, 1).cpu()
    arr = (t.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def save_tensor_image(t: torch.Tensor, path: Union[str, Path]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    img = tensor_to_pil(t)
    img.save(path)


def concat_horiz(*imgs: Image.Image) -> Image.Image:
    widths = [im.width for im in imgs]
    heights = [im.height for im in imgs]
    H = max(heights)
    W = sum(widths)
    out = Image.new("RGB", (W, H))
    x = 0
    for im in imgs:
        if im.height != H:
            im = im.resize((im.width, H), resample=Image.BILINEAR)
        out.paste(im, (x, 0))
        x += im.width
    return out


def save_triplet(
    lq: torch.Tensor,
    base: torch.Tensor,
    out: torch.Tensor,
    path: Union[str, Path],
    label_lq: str = "LQ",
    label_base: str = "BASE",
    label_out: str = "OUT",
) -> None:
    """Save a horizontal concat: [lq | base | out]."""
    path = Path(path)
    ensure_dir(path.parent)

    pil_lq = tensor_to_pil(lq)
    pil_base = tensor_to_pil(base)
    pil_out = tensor_to_pil(out)

    # (optional) add simple captions in the future; keep minimal for now
    cat = concat_horiz(pil_lq, pil_base, pil_out)
    cat.save(path)

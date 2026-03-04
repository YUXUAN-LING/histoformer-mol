# lora_adapters/vis_utils.py
# -*- coding: utf-8 -*-
"""
Lightweight visualization helpers for inference scripts.

Design goals:
- Keep infer scripts clean (no PIL/IO clutter).
- Default: save a single triplet concat image (LQ | BASE | MIX).
- Optional: save single images via flags (base/mix and optionally lq).

Expected tensor range: [0,1], shape [B,3,H,W] or [3,H,W].
"""

from __future__ import annotations

import os
from typing import Optional, Dict

import torch
from torchvision import transforms

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:
    _PIL_OK = False

# Reuse your repo's save_image if available (consistent output)
try:
    from lora_adapters.utils import save_image as _save_image_repo
except Exception:
    _save_image_repo = None


def _ensure_bchw(t: torch.Tensor) -> torch.Tensor:
    """Convert [3,H,W] -> [1,3,H,W]. Keep [B,3,H,W] as-is."""
    if t.dim() == 3:
        return t.unsqueeze(0)
    return t


def _clamp01_bchw(t: torch.Tensor) -> torch.Tensor:
    t = _ensure_bchw(t)
    return t.detach().float().cpu().clamp(0, 1)


def make_triplet_concat(lq: torch.Tensor, base: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    """
    Return [B,3,H,3W] tensor concatenated along width dimension.
    Inputs can be [B,3,H,W] or [3,H,W].
    """
    lq = _clamp01_bchw(lq)
    base = _clamp01_bchw(base)
    mix = _clamp01_bchw(mix)
    return torch.cat([lq, base, mix], dim=3)


def _tensor_to_pil(t_bchw: torch.Tensor) -> "Image.Image":
    """Convert [1,3,H,W] to PIL."""
    t = _clamp01_bchw(t_bchw)[0]
    return transforms.ToPILImage()(t)


def _save_tensor_image(t_bchw: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t_bchw = _clamp01_bchw(t_bchw)

    if _save_image_repo is not None:
        # repo save_image expects torch tensor, usually [B,C,H,W]
        _save_image_repo(t_bchw, path)
        return

    # fallback: PIL save
    _tensor_to_pil(t_bchw).save(path)


def annotate_pil(
    img: "Image.Image",
    text: str,
    xy=(8, 8),
    font_size: int = 18,
) -> "Image.Image":
    if not _PIL_OK:
        return img
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = None
    # white text with a light shadow for readability
    x, y = xy
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return out


def save_triplet(
    out_dir: str,
    stem: str,
    lq: torch.Tensor,
    base: torch.Tensor,
    mix: torch.Tensor,
    *,
    save_concat: bool = True,
    save_singles: bool = False,
    save_lq: bool = False,
    annotate: bool = False,
    meta: Optional[Dict] = None,
):
    """
    Save images to out_dir:
      - <stem>_concat.png : (LQ | BASE | MIX)   (default)
      - <stem>_base.png, <stem>_mix.png        (optional)
      - <stem>_lq.png                          (optional)

    annotate=True writes meta text on concat (if PIL available).
    """
    os.makedirs(out_dir, exist_ok=True)

    if save_concat:
        c = make_triplet_concat(lq, base, mix)  # [B,3,H,3W]
        if annotate and meta is not None and _PIL_OK:
            img = _tensor_to_pil(c)
            lines = []
            # stable order
            for k in ["mode", "local", "global", "psnr_base", "psnr_mix", "ssim_base", "ssim_mix"]:
                if k in meta and meta[k] not in [None, ""]:
                    lines.append(f"{k}: {meta[k]}")
            img = annotate_pil(img, "\n".join(lines))
            img.save(os.path.join(out_dir, f"{stem}_concat.png"))
        else:
            _save_tensor_image(c, os.path.join(out_dir, f"{stem}_concat.png"))

    if save_singles:
        _save_tensor_image(_ensure_bchw(base), os.path.join(out_dir, f"{stem}_base.png"))
        _save_tensor_image(_ensure_bchw(mix),  os.path.join(out_dir, f"{stem}_mix.png"))
        if save_lq:
            _save_tensor_image(_ensure_bchw(lq), os.path.join(out_dir, f"{stem}_lq.png"))

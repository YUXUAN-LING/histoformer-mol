# mol_infer/io/image_save.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch


def save_triplet_images(
    output_dir: str,
    stem: str,
    lq: torch.Tensor,
    base: torch.Tensor,
    mix: torch.Tensor,
    save_concat: bool = True,
    save_singles: bool = False,
    save_lq: bool = False,
    annotate: bool = False,
    meta: Optional[Dict[str, Any]] = None,
):
    """
    适配你现有签名：
        save_triplet(
            out_dir=trip_dir,
            stem=stem,
            lq=lq_t,
            base=y_base,
            mix=y_mix,
            save_concat=True,
            save_singles=False,
            save_lq=False,
            annotate=...,
            meta=meta,
        )

    这里 output_dir 是“总输出目录”，我们内部会用：
        trip_dir = output_dir/triplets
    """
    from lora_adapters.vis_utils import save_triplet  # 你的现成实现

    trip_dir = os.path.join(output_dir, "triplets")
    os.makedirs(trip_dir, exist_ok=True)

    save_triplet(
        out_dir=trip_dir,
        stem=stem,
        lq=lq,
        base=base,
        mix=mix,
        save_concat=bool(save_concat),
        save_singles=bool(save_singles),
        save_lq=bool(save_lq),
        annotate=bool(annotate),
        meta=meta,
    )

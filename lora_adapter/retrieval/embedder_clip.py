# -*- coding: utf-8 -*-
"""lora_adapters.retrieval.embedder_clip

CLIP image embedder (open_clip).

This module is intentionally minimal and self-contained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, Optional

import numpy as np
import torch
from PIL import Image

open_clip = None  # lazy import (keeps CLI importable even if open_clip isn't installed)


class CLIPEmbedder:
    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
    ):
        global open_clip
        if open_clip is None:
            try:
                import open_clip as _open_clip
            except Exception as e:  # pragma: no cover
                raise ImportError(
                    "open_clip not found. Please install: pip install open_clip_torch"
                ) from e
            open_clip = _open_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CLIPEmbedder] loading {model_name} ({pretrained}) on {self.device}")
        model, preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.dim = int(getattr(model.visual, "output_dim", 512))
    @torch.no_grad()
    def embed_image(self, img_or_path: Union[str, Path, Image.Image]) -> np.ndarray:
        if isinstance(img_or_path, (str, Path)):
            img = Image.open(img_or_path).convert("RGB")
        else:
            img = img_or_path

        x = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)  # [1, C]
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).detach().cpu().numpy()

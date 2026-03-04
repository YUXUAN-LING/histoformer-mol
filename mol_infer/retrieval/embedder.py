# mol_infer/retrieval/embedder.py
from __future__ import annotations
from PIL import Image

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

# your existing embedder
try:
    from lora_adapters.embedding_clip import CLIPEmbedder as _CLIPEmbedder
except Exception as e:
    raise ImportError(f"Cannot import CLIPEmbedder from lora_adapters.embedding_clip: {e}")


@dataclass
class Embedder:
    """
    Minimal standardized embedder wrapper.
    Runner will call: emb = embedder.encode_image_tensor(x) -> np.ndarray (1, D)
    """
    _impl: Any
    device: str = "cuda"

    @torch.inference_mode()
    def encode_image_tensor(self, x: torch.Tensor) -> np.ndarray:
        """
        x: torch tensor [1,3,H,W] in 0..1
        returns np float32 [1,D]
        """
        # ensure on cpu for PIL conversion
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Expected [B,3,H,W], got shape={tuple(x.shape)}")

        # take first image in batch
        t = x[0].detach().clamp(0, 1).cpu()

        # convert to uint8 HWC
        t = (t * 255.0).round().to(torch.uint8)          # [3,H,W]
        hwc = t.permute(1, 2, 0).numpy()                 # [H,W,3]
        img = Image.fromarray(hwc, mode="RGB")

        # --- support both APIs: encode_image / embed_image ---
        if hasattr(self._impl, "encode_image"):
            emb = self._impl.encode_image(img)           # PIL
        elif hasattr(self._impl, "embed_image"):
            emb = self._impl.embed_image(img)            # PIL
        else:
            raise AttributeError("CLIPEmbedder has neither encode_image nor embed_image.")

        if isinstance(emb, torch.Tensor):
            emb = emb.detach().float().cpu().numpy()
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb[None, :]
        return emb


    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> np.ndarray:
        return self.encode_image_tensor(x)


def _get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    # dataclass / object
    if hasattr(obj, key):
        v = getattr(obj, key)
        return default if v is None else v
    # dict
    if isinstance(obj, dict) and key in obj:
        v = obj.get(key)
        return default if v is None else v
    return default


def build_embedder(retrieval_cfg: Any, device: str = "cuda") -> Embedder:
    """
    retrieval_cfg can be:
      - RetrievalConfig (dataclass)
      - argparse.Namespace
      - dict
      - or a plain string "clip" (legacy)
    """
    # accept legacy string
    if isinstance(retrieval_cfg, str):
        embedder_name = retrieval_cfg
        clip_model = "ViT-B-16"
        clip_pretrained = None
    else:
        embedder_name = _get(retrieval_cfg, "embedder", "clip")
        clip_model = _get(retrieval_cfg, "clip_model", "ViT-B-16")
        clip_pretrained = _get(retrieval_cfg, "clip_pretrained", None)

    name = str(embedder_name).lower()
    if name != "clip":
        raise ValueError(f"Unknown embedder: {embedder_name}")

    impl = _CLIPEmbedder(
        model_name=str(clip_model),
        pretrained=clip_pretrained,
        device=device,
    )
    return Embedder(_impl=impl, device=device)

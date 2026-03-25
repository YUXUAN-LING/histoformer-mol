from __future__ import annotations

import numpy as np
from PIL import Image

from lora_adapters.embedding_clip import CLIPEmbedder

from .base import ImageEmbedder


class ClipImageEmbedder(ImageEmbedder):
    def __init__(self, model_name: str = "ViT-B-16", pretrained: str | None = None, device: str = "cuda"):
        self.impl = CLIPEmbedder(model_name=model_name, pretrained=pretrained, device=device)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        emb = self.impl.encode_image(image)
        emb = np.asarray(emb, dtype=np.float32)
        return emb.reshape(-1)

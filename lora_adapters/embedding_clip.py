# lora_adapters/embedding_clip.py
from pathlib import Path
from typing import Union
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import open_clip  # 需要: pip install open_clip_torch


class CLIPEmbedder:
    def __init__(
        self,
        device: str = None,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai"
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CLIP] loading {model_name} ({pretrained}) on {self.device}")
        model, preprocess, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess

    @torch.no_grad()
    def embed_image(self, img_or_path: Union[str, Path, Image.Image]) -> np.ndarray:
        if isinstance(img_or_path, (str, Path)):
            img = Image.open(img_or_path).convert("RGB")
        else:
            img = img_or_path

        x = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(x)   # [1, C]
            feat = feat / feat.norm(dim=-1, keepdim=True)

        return feat.squeeze(0).cpu().numpy()

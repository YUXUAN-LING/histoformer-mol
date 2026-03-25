from __future__ import annotations

from PIL import Image
import numpy as np
import torch


def load_image_tensor(path: str) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

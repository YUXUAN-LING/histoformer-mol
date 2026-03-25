from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from PIL import Image


class ImageEmbedder(ABC):
    @abstractmethod
    def encode_image(self, image: Image.Image) -> np.ndarray:
        ...

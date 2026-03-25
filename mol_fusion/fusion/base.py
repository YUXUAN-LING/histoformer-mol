from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict


class FusionPolicy(ABC):
    @abstractmethod
    def build_layer_weights(self, *, model, dom1: str, dom2: str, context: Dict | None = None) -> Dict[str, Dict[str, float]]:
        ...


def normalize_pair(a: float, b: float):
    s = max(a + b, 1e-12)
    return a / s, b / s

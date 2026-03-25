from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict


class PairScoreFunction(ABC):
    @abstractmethod
    def compute(self, model, dom1: str, dom2: str, activation_norms: Dict[str, float] | None = None) -> Dict[str, Dict[str, float]]:
        """Return per-layer scalar score for dom1/dom2."""
        ...

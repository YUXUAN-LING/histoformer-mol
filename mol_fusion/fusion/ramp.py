from __future__ import annotations

import math
from typing import Dict

from .base import FusionPolicy, normalize_pair
from mol_fusion.utils.module_utils import collect_layer_names


def ramp_curve(pos01: float, mode: str = "sigmoid", alpha: float = 8.0, beta: float = 0.5, complement: bool = False, mirror: bool = False) -> float:
    x = min(1.0, max(0.0, float(pos01)))
    if mirror:
        x = 1.0 - abs(2 * x - 1.0)
    if mode == "linear":
        y = x
    elif mode == "sigmoid":
        y = 1.0 / (1.0 + math.exp(-alpha * (x - beta)))
    elif mode == "beta":
        # simple beta-like shape without scipy dependency
        a = max(alpha, 1e-4)
        b = max(beta, 1e-4)
        y = (x ** a) / max((x ** a) + ((1.0 - x) ** b), 1e-8)
    else:
        raise ValueError(f"Unknown ramp mode: {mode}")
    return 1.0 - y if complement else y


class PureRampPolicy(FusionPolicy):
    def __init__(self, mode: str = "sigmoid", alpha: float = 8.0, beta: float = 0.5, complement: bool = False, mirror: bool = False):
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.complement = complement
        self.mirror = mirror

    def build_layer_weights(self, *, model, dom1: str, dom2: str, context: Dict | None = None):
        names = collect_layer_names(model)
        n = max(len(names), 1)
        out = {}
        for i, name in enumerate(names):
            p = i / max(n - 1, 1)
            w2 = ramp_curve(p, mode=self.mode, alpha=self.alpha, beta=self.beta, complement=self.complement, mirror=self.mirror)
            w1 = 1.0 - w2
            w1, w2 = normalize_pair(w1, w2)
            out[name] = {dom1: w1, dom2: w2}
        return out

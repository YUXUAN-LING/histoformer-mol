from __future__ import annotations

from typing import Dict

from .base import PairScoreFunction
from .common import base_weight_abs_sum, delta_weight_abs_sum
from mol_fusion.utils.module_utils import iter_lora_named_modules


class ShiftScore(PairScoreFunction):
    def compute(self, model, dom1: str, dom2: str, activation_norms: Dict[str, float] | None = None):
        activation_norms = activation_norms or {}
        out = {}
        for name, module in iter_lora_named_modules(model):
            act = float(activation_norms.get(name, 1.0))
            bw = base_weight_abs_sum(module)
            d1 = max(delta_weight_abs_sum(module, dom1) - bw, 0.0)
            d2 = max(delta_weight_abs_sum(module, dom2) - bw, 0.0)
            out[name] = {dom1: d1 * act, dom2: d2 * act}
        return out

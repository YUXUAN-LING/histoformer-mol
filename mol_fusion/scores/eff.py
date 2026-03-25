from __future__ import annotations

from typing import Dict

from .base import PairScoreFunction
from .common import delta_weight_abs_sum
from mol_fusion.utils.module_utils import iter_lora_named_modules


class EffScore(PairScoreFunction):
    def compute(self, model, dom1: str, dom2: str, activation_norms: Dict[str, float] | None = None):
        out = {}
        for name, module in iter_lora_named_modules(model):
            s1 = delta_weight_abs_sum(module, dom1)
            s2 = delta_weight_abs_sum(module, dom2)
            den = max(s1 + s2, 1e-12)
            out[name] = {dom1: s1 / den, dom2: s2 / den}
        return out

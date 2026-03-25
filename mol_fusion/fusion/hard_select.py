from __future__ import annotations

from .base import FusionPolicy
from .ramp import PureRampPolicy


class HardSelectPolicy(FusionPolicy):
    def __init__(self, mode: str = "score", score_fn=None, ramp_policy: PureRampPolicy | None = None):
        self.mode = mode
        self.score_fn = score_fn
        self.ramp_policy = ramp_policy or PureRampPolicy()

    def build_layer_weights(self, *, model, dom1: str, dom2: str, context=None):
        if self.mode == "prior":
            ramp = self.ramp_policy.build_layer_weights(model=model, dom1=dom1, dom2=dom2, context=context)
            return {n: {dom1: float(v[dom1] >= v[dom2]), dom2: float(v[dom2] > v[dom1])} for n, v in ramp.items()}
        if self.score_fn is None:
            raise ValueError("score mode requires score_fn")
        sc = self.score_fn.compute(model, dom1, dom2, (context or {}).get("activation_norms"))
        out = {}
        for n, s in sc.items():
            out[n] = {dom1: float(s[dom1] >= s[dom2]), dom2: float(s[dom2] > s[dom1])}
        return out

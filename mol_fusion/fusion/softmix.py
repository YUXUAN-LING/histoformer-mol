from __future__ import annotations

import math
from typing import Dict

from .base import FusionPolicy, normalize_pair
from .ramp import PureRampPolicy
from mol_fusion.scores.shared_metrics import aggregate_shared


def _pair_from_scores(s1: float, s2: float, rule: str, temp: float):
    if rule == "ratio":
        return normalize_pair(max(s1, 0.0), max(s2, 0.0))
    t = max(temp, 1e-6)
    a = math.exp(s1 / t)
    b = math.exp(s2 / t)
    return normalize_pair(a, b)


class PureSoftmixPolicy(FusionPolicy):
    def __init__(self, score_fn, weight_rule: str = "softmax", temp: float = 0.7):
        self.score_fn = score_fn
        self.weight_rule = weight_rule
        self.temp = temp

    def build_layer_weights(self, *, model, dom1: str, dom2: str, context: Dict | None = None):
        layer_scores = self.score_fn.compute(model, dom1, dom2, (context or {}).get("activation_norms"))
        out = {}
        for name, ds in layer_scores.items():
            w1, w2 = _pair_from_scores(float(ds[dom1]), float(ds[dom2]), self.weight_rule, self.temp)
            out[name] = {dom1: w1, dom2: w2}
        return out


class TopKSoftmixPolicy(FusionPolicy):
    def __init__(self, score_fn, topk: int = 6, shared_metric: str = "min", nonkey_mode: str = "half", weight_rule: str = "softmax", temp: float = 0.7, ramp_policy: PureRampPolicy | None = None):
        self.score_fn = score_fn
        self.topk = int(topk)
        self.shared_metric = shared_metric
        self.nonkey_mode = nonkey_mode
        self.weight_rule = weight_rule
        self.temp = temp
        self.ramp_policy = ramp_policy or PureRampPolicy()

    def build_layer_weights(self, *, model, dom1: str, dom2: str, context: Dict | None = None):
        full_scores = self.score_fn.compute(model, dom1, dom2, (context or {}).get("activation_norms"))
        names = list(full_scores.keys())

        # semantic rule 1: topk < 0 => full softmix
        if self.topk < 0:
            return PureSoftmixPolicy(self.score_fn, weight_rule=self.weight_rule, temp=self.temp).build_layer_weights(model=model, dom1=dom1, dom2=dom2, context=context)

        ramp_weights = self.ramp_policy.build_layer_weights(model=model, dom1=dom1, dom2=dom2, context=context)

        # rank by shared metric
        ranked = sorted(
            names,
            key=lambda n: aggregate_shared(float(full_scores[n][dom1]), float(full_scores[n][dom2]), self.shared_metric),
            reverse=True,
        )
        key_layers = set(ranked[: max(0, self.topk)])
        out = {}
        for name in names:
            if name in key_layers:
                out[name] = dict(zip((dom1, dom2), _pair_from_scores(full_scores[name][dom1], full_scores[name][dom2], self.weight_rule, self.temp)))
                continue
            if self.nonkey_mode == "half":
                out[name] = {dom1: 0.5, dom2: 0.5}
            elif self.nonkey_mode == "hard-dom1":
                out[name] = {dom1: 1.0, dom2: 0.0}
            elif self.nonkey_mode == "hard-dom2":
                out[name] = {dom1: 0.0, dom2: 1.0}
            elif self.nonkey_mode == "ramp":
                out[name] = ramp_weights[name]
            else:
                raise ValueError(f"Unknown nonkey_mode={self.nonkey_mode}")
            out[name][dom1], out[name][dom2] = normalize_pair(out[name][dom1], out[name][dom2])

        # semantic rule 4: topk=0 + nonkey=ramp == pure ramp
        if self.topk == 0 and self.nonkey_mode == "ramp":
            return ramp_weights
        return out

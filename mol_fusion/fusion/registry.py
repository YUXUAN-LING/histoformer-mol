from __future__ import annotations

from mol_fusion.scores.delta import DeltaScore
from mol_fusion.scores.shift import ShiftScore
from mol_fusion.scores.eff import EffScore

from .hard_select import HardSelectPolicy
from .ramp import PureRampPolicy
from .softmix import PureSoftmixPolicy, TopKSoftmixPolicy


SCORE_REGISTRY = {
    "delta": DeltaScore,
    "shift": ShiftScore,
    "eff": EffScore,
}


def build_score(name: str):
    if name not in SCORE_REGISTRY:
        raise ValueError(f"Unknown score_type={name}")
    return SCORE_REGISTRY[name]()


def build_fusion_policy(name: str, **kwargs):
    if name == "pure_ramp":
        return PureRampPolicy(
            mode=kwargs.get("ramp_mode", "sigmoid"),
            alpha=float(kwargs.get("ramp_alpha", 8.0)),
            beta=float(kwargs.get("ramp_beta", 0.5)),
            complement=bool(kwargs.get("ramp_complement", False)),
            mirror=bool(kwargs.get("ramp_mirror", False)),
        )
    score_fn = build_score(kwargs.get("score_type", "delta"))
    if name == "pure_softmix":
        return PureSoftmixPolicy(score_fn=score_fn, weight_rule=kwargs.get("weight_rule", "softmax"), temp=float(kwargs.get("temp", 0.7)))
    if name == "topk_softmix":
        ramp = build_fusion_policy("pure_ramp", **kwargs)
        return TopKSoftmixPolicy(
            score_fn=score_fn,
            topk=int(kwargs.get("topk", 6)),
            shared_metric=kwargs.get("shared_metric", "min"),
            nonkey_mode=kwargs.get("nonkey_mode", "half"),
            weight_rule=kwargs.get("weight_rule", "softmax"),
            temp=float(kwargs.get("temp", 0.7)),
            ramp_policy=ramp,
        )
    if name == "hard_select":
        ramp = build_fusion_policy("pure_ramp", **kwargs)
        return HardSelectPolicy(mode=kwargs.get("hard_mode", "score"), score_fn=score_fn, ramp_policy=ramp)
    raise ValueError(f"Unknown fusion policy: {name}")

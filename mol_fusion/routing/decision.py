from __future__ import annotations

from typing import Dict

from .pair_selector import select_pair


def route_decision(
    retrieval_weights: Dict[str, float],
    single_tau: float,
    margin_tau: float,
    pair_policy: str = "top2",
    local_domains=(),
    global_domains=(),
):
    items = sorted(retrieval_weights.items(), key=lambda kv: kv[1], reverse=True)
    if not items:
        raise ValueError("Empty retrieval weights")
    top1, w1 = items[0]
    top2, w2 = items[1] if len(items) > 1 else (top1, 0.0)
    margin = w1 - w2
    if w1 >= single_tau:
        return {"route": "single", "dom1": top1, "dom2": None, "scores": retrieval_weights, "reason": f"top1>={single_tau}"}
    if margin >= margin_tau:
        return {"route": "single", "dom1": top1, "dom2": None, "scores": retrieval_weights, "reason": f"margin>={margin_tau}"}
    dom1, dom2 = select_pair(retrieval_weights, policy=pair_policy, local_domains=local_domains, global_domains=global_domains)
    return {"route": "pair", "dom1": dom1, "dom2": dom2, "scores": retrieval_weights, "reason": f"pair_policy={pair_policy}"}

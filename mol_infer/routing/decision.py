# mol_infer/routing/decision.py
from __future__ import annotations

from typing import List, Optional, Tuple

from mol_infer.core.types import RoutingDecision


def decide_single_or_mix(
    picks: List[Tuple[str, float]],  # already softmaxed, sorted desc
    single_tau: float,
    single_margin: float,
    mix_topk: int,
    local_domains: List[str],
    global_domains: List[str],
) -> RoutingDecision:
    """
    Rules:
      - single if:
            top1_w >= single_tau
         OR top1_w - top2_w >= single_margin
      - mix otherwise:
            choose best-local + best-global within top mix_topk
            (not necessarily top1/top2)
    """
    if not picks:
        raise ValueError("[routing] empty picks")

    top1_d, top1_w = picks[0][0], float(picks[0][1])
    top2_w = float(picks[1][1]) if len(picks) >= 2 else 0.0
    margin = top1_w - top2_w

    if top1_w >= single_tau:
        return RoutingDecision(
            mode="single",
            top1=top1_d, top1_w=top1_w, top2_w=top2_w, margin=margin,
            reason=f"top1_w({top1_w:.3f})>=tau({single_tau:.3f})",
        )

    if margin >= single_margin:
        return RoutingDecision(
            mode="single",
            top1=top1_d, top1_w=top1_w, top2_w=top2_w, margin=margin,
            reason=f"margin({margin:.3f})>=single_margin({single_margin:.3f})",
        )

    # mix selection
    top = picks[:max(1, min(mix_topk, len(picks)))]
    best_local: Optional[Tuple[str, float]] = None
    best_global: Optional[Tuple[str, float]] = None

    for d, w in top:
        if (d in local_domains) and best_local is None:
            best_local = (d, float(w))
        if (d in global_domains) and best_global is None:
            best_global = (d, float(w))
        if best_local and best_global:
            break

    # fallback if missing one side
    if best_local is None:
        # try pick first non-global as local, else top1
        for d, w in top:
            if d not in global_domains:
                best_local = (d, float(w))
                break
        if best_local is None:
            best_local = (top1_d, top1_w)

    if best_global is None:
        for d, w in top:
            if d not in local_domains:
                best_global = (d, float(w))
                break
        if best_global is None:
            best_global = (top1_d, top1_w)

    return RoutingDecision(
        mode="mix",
        top1=top1_d, top1_w=top1_w, top2_w=top2_w, margin=margin,
        local=best_local[0], local_w=best_local[1],
        global_=best_global[0], global_w=best_global[1],
        reason=f"mix: local_best={best_local[0]}({best_local[1]:.3f}), "
               f"global_best={best_global[0]}({best_global[1]:.3f})",
    )

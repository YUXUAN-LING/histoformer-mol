# -*- coding: utf-8 -*-
"""lora_adapters.decision.decide

Rule-based decision module.

Input:
  - RouteResult: sorted domain weights from retrieval.
  - thresholds: single_tau / single_margin
  - mix_topk: only consider top-k candidates when picking mix pair
  - local_domains/global_domains: domain grouping lists

Output:
  - Decision object containing mode and selected domains.

Note: This is intentionally *stateless* so you can swap decision logic easily.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lora_adapter.retrieval.router import RouteResult


@dataclass
class Decision:
    mode: str  # 'single' or 'mix'
    top1: str
    top1_w: float
    top2: Optional[str] = None
    top2_w: float = 0.0

    # mix selection
    local: Optional[str] = None
    global_: Optional[str] = None
    local_w: float = 0.0
    global_w: float = 0.0

    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["global"] = d.pop("global_")
        return d


def decide_single_or_mix(
    route: RouteResult,
    single_tau: float,
    single_margin: float,
    mix_topk: int,
    local_domains: Sequence[str],
    global_domains: Sequence[str],
) -> Decision:
    items = route.items
    if len(items) == 0:
        raise ValueError("RouteResult has 0 items")

    top1_d, top1_w = items[0].domain, float(items[0].weight)
    top2_d, top2_w = (items[1].domain, float(items[1].weight)) if len(items) >= 2 else (None, 0.0)

    # (A) single conditions
    if top1_w >= float(single_tau):
        return Decision(
            mode="single",
            top1=top1_d,
            top1_w=top1_w,
            top2=top2_d,
            top2_w=top2_w,
            reason=f"top1_w>=tau({top1_w:.4f}>={single_tau})",
        )

    if (top1_w - top2_w) >= float(single_margin):
        return Decision(
            mode="single",
            top1=top1_d,
            top1_w=top1_w,
            top2=top2_d,
            top2_w=top2_w,
            reason=f"top1-top2>=margin({top1_w-top2_w:.4f}>={single_margin})",
        )

    # (B) mix: pick best local + best global from top mix_topk candidates
    cand = items[: max(1, int(mix_topk))]
    local_set = set(local_domains)
    global_set = set(global_domains)

    best_local = None
    best_global = None
    for it in cand:
        if best_local is None and it.domain in local_set:
            best_local = it
        if best_global is None and it.domain in global_set:
            best_global = it
        if best_local is not None and best_global is not None:
            break

    if best_local is None or best_global is None:
        return Decision(
            mode="single",
            top1=top1_d,
            top1_w=top1_w,
            top2=top2_d,
            top2_w=top2_w,
            reason="mix not possible (missing local/global in candidates) -> fallback single",
        )

    if best_local.domain == best_global.domain:
        return Decision(
            mode="single",
            top1=top1_d,
            top1_w=top1_w,
            top2=top2_d,
            top2_w=top2_w,
            reason="mix not possible (local==global) -> fallback single",
        )

    return Decision(
        mode="mix",
        top1=top1_d,
        top1_w=top1_w,
        top2=top2_d,
        top2_w=top2_w,
        local=best_local.domain,
        global_=best_global.domain,
        local_w=float(best_local.weight),
        global_w=float(best_global.weight),
        reason=(
            f"mix(best_local={best_local.domain}:{best_local.weight:.4f}, "
            f"best_global={best_global.domain}:{best_global.weight:.4f})"
        ),
    )

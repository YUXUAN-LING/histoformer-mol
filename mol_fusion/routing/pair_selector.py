from __future__ import annotations

from typing import Dict, Iterable, Tuple


def _sorted(scores: Dict[str, float]):
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def select_pair(scores: Dict[str, float], policy: str = "top2", local_domains: Iterable[str] = (), global_domains: Iterable[str] = (), manual_pair: tuple[str, str] | None = None) -> Tuple[str, str]:
    if manual_pair is not None:
        return manual_pair
    items = _sorted(scores)
    if len(items) < 2:
        raise ValueError("Need at least two domains for pair selection")
    if policy == "top2":
        return items[0][0], items[1][0]
    if policy == "local_global":
        local_set, global_set = set(local_domains), set(global_domains)
        d1 = next((d for d, _ in items if d in local_set), None)
        d2 = next((d for d, _ in items if d in global_set and d != d1), None)
        if d1 and d2:
            return d1, d2
    return items[0][0], items[1][0]

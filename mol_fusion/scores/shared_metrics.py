from __future__ import annotations


def aggregate_shared(v1: float, v2: float, metric: str) -> float:
    m = metric.lower()
    if m == "min":
        return min(v1, v2)
    if m == "mean":
        return 0.5 * (v1 + v2)
    if m == "max":
        return max(v1, v2)
    if m == "gap":
        return abs(v1 - v2)
    raise ValueError(f"Unknown shared metric: {metric}")

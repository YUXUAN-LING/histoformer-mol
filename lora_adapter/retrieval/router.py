# -*- coding: utf-8 -*-
"""lora_adapters.retrieval.router

Given an image embedding and a PrototypeBank, compute domain ranking and weights.

Output is a RouteResult (sorted by weight descending).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .prototype_bank import PrototypeBank


@dataclass
class RouteItem:
    domain: str
    raw_score: float  # similarity (higher better) or distance/proximity (depends on metric)
    weight: float     # softmax weight after temperature


@dataclass
class RouteResult:
    metric: str
    temperature: float
    items: List[RouteItem]  # sorted by weight desc

    def topk(self, k: int) -> List[Tuple[str, float]]:
        return [(it.domain, float(it.weight)) for it in self.items[:max(1, k)]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "temperature": float(self.temperature),
            "items": [asdict(it) for it in self.items],
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    s = float(e.sum())
    if s <= 1e-12:
        return np.ones_like(e) / max(1, e.size)
    return e / s


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)
    return float(np.dot(a, b))


def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.sum(diff * diff)))


class Router:
    """Prototype router.

    Usage:
      bank = PrototypeBank(...)
      router = Router(bank, sim_metric='euclidean', temperature=0.07)
      rr = router.route(img_emb, topk=5)
    """

    def __init__(
        self,
        bank: PrototypeBank,
        sim_metric: str = "euclidean",
        temperature: float = 0.07,
    ):
        self.bank = bank
        self.sim_metric = sim_metric.lower()
        self.temperature = float(max(temperature, 1e-6))

        if self.sim_metric not in ("cosine", "euclidean", "l2"):
            raise ValueError(f"Unknown sim_metric: {sim_metric}")

        # cache matrix for speed
        self._names = self.bank.names()
        self._protos = self.bank.proto_matrix()  # [D, C]

    def route(self, img_emb: np.ndarray, topk: int = 5, temperature: Optional[float] = None) -> RouteResult:
        topk = max(1, int(topk))
        t = float(max(temperature if temperature is not None else self.temperature, 1e-6))

        if img_emb.ndim != 1:
            img_emb = img_emb.reshape(-1)
        img_emb = img_emb.astype(np.float32)

        if self.sim_metric == "cosine":
            raw = np.array([cosine_sim(img_emb, p) for p in self._protos], dtype=np.float32)
            # larger better
            idx = np.argsort(-raw)[:topk]
            raw_top = raw[idx]
            w = _softmax(raw_top / t)
            items = [
                RouteItem(domain=self._names[i], raw_score=float(raw[i]), weight=float(wj))
                for i, wj in zip(idx.tolist(), w.tolist())
            ]
        else:
            # distance smaller better -> convert to proximity
            dist = np.array([l2_dist(img_emb, p) for p in self._protos], dtype=np.float32)
            idx = np.argsort(dist)[:topk]
            dist_top = dist[idx]
            prox = 1.0 / np.maximum(dist_top, 1e-6)
            w = _softmax(prox / t)
            items = [
                RouteItem(domain=self._names[i], raw_score=float(dist[i]), weight=float(wj))
                for i, wj in zip(idx.tolist(), w.tolist())
            ]

        # sort by weight desc for convenience
        items.sort(key=lambda x: float(x.weight), reverse=True)
        return RouteResult(metric=self.sim_metric, temperature=t, items=items)

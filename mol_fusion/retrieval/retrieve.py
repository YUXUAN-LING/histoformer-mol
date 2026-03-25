from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from PIL import Image

from .embedders.base import ImageEmbedder
from .prototype_store import PrototypeStore
from .scorers import cosine, proximity_l2, softmax_temperature


@dataclass
class RetrievalOutput:
    scores: Dict[str, float]
    weights: Dict[str, float]

    def topk(self, k: int) -> List[tuple[str, float]]:
        return sorted(self.weights.items(), key=lambda kv: kv[1], reverse=True)[: max(1, k)]


class Retriever:
    def __init__(self, embedder: ImageEmbedder, store: PrototypeStore, sim_metric: str = "cosine", temperature: float = 0.07):
        self.embedder = embedder
        self.store = store
        self.sim_metric = sim_metric.lower()
        self.temperature = temperature

    def retrieve_image(self, image: Image.Image) -> RetrievalOutput:
        emb = self.embedder.encode_image(image)
        raw: Dict[str, float] = {}
        for dom, dp in self.store.items.items():
            vals = [cosine(emb, p) if self.sim_metric == "cosine" else proximity_l2(emb, p) for p in dp.vectors]
            raw[dom] = float(max(vals))
        names = list(raw.keys())
        arr = np.asarray([raw[n] for n in names], dtype=np.float32)
        w = softmax_temperature(arr, self.temperature)
        return RetrievalOutput(scores=raw, weights={n: float(w[i]) for i, n in enumerate(names)})

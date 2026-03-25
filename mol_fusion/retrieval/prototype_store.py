from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class DomainPrototype:
    name: str
    vectors: np.ndarray  # [K,C]


class PrototypeStore:
    def __init__(self, proto_root: str, domains: List[str], proto_tag: str | None = None):
        self.proto_root = Path(proto_root)
        self.domains = domains
        self.proto_tag = proto_tag
        self.items: Dict[str, DomainPrototype] = {}
        self._load()

    def _resolve_proto_path(self, dom: str) -> Path:
        d = self.proto_root / dom
        cand = []
        if self.proto_tag:
            cand += [d / f"prototypes_{self.proto_tag}.npy", d / f"avg_embedding_{self.proto_tag}.npy"]
        cand += [d / "prototypes.npy", d / "avg_embedding.npy"]
        for p in cand:
            if p.exists():
                return p
        raise FileNotFoundError(f"No prototype found for {dom} under {d}")

    def _load(self):
        for dom in self.domains:
            p = self._resolve_proto_path(dom)
            arr = np.load(p).astype(np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            self.items[dom] = DomainPrototype(name=dom, vectors=arr)

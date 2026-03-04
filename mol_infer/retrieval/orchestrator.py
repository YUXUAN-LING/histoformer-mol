# mol_infer/retrieval/orchestrator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from mol_infer.core.types import RetrievalResult

# try import path variants
try:
    from lora_adapters.domain_orchestrator import DomainOrchestrator as _DomainOrchestrator
except Exception:
    try:
        from lora_adapters.domain_orchestrator import DomainOrchestrator as _DomainOrchestrator  # type: ignore
    except Exception as e:
        raise ImportError(f"Cannot import DomainOrchestrator: {e}")


@dataclass
class RetrievalOrchestrator:
    """
    Thin wrapper around your existing DomainOrchestrator.

    Standardized API:
      - retrieve(img_emb, topk, temperature, return_raw_scores) -> RetrievalResult
    """
    domains: List[str]
    loradb_root: str
    sim_metric: str = "cosine"
    temperature: float = 0.07
    embedder_tag: Optional[str] = None

    def __post_init__(self):
        self._orch = _DomainOrchestrator(
            domains=self.domains,
            lora_db_path=self.loradb_root,
            sim_metric=self.sim_metric,
            temperature=self.temperature,
            embedder_tag=self.embedder_tag,
        )

    def retrieve(
        self,
        img_emb: np.ndarray,
        topk: int,
        temperature: Optional[float] = None,
        return_raw_scores: bool = False,
        norm_topk_domains: int = 0,
        include_domains: Optional[List[str]] = None,
    ) -> RetrievalResult:
        """
        Returns:
          RetrievalResult.picks: [(domain, weight)] softmaxed over topk (sorted desc)
          RetrievalResult.raw_scores: [(domain, score)] optional full list
        """
        picks = self._orch.select_topk(
            img_emb,
            top_k=topk,
            temperature=temperature,
            norm_topk_domains=norm_topk_domains,
            include_domains=include_domains,
        )

        raw_scores = None
        if return_raw_scores:
            raw_scores = self._compute_raw_scores(img_emb)

        return RetrievalResult(picks=picks, raw_scores=raw_scores)

    def _compute_raw_scores(self, img_emb: np.ndarray) -> List[Tuple[str, float]]:
        """
        DomainOrchestrator.select_topk only returns topk weights.
        For debugging we recompute raw metric values for ALL domains.
        """
        metric = getattr(self._orch, "sim_metric", "cosine")
        metric = str(metric).lower()

        out: List[Tuple[str, float]] = []
        # DomainOrchestrator stores prototypes in self.domains: Dict[str, Domain]
        dom_map = getattr(self._orch, "domains", {})
        if not dom_map:
            return out

        if metric == "cosine":
            for d, dom in dom_map.items():
                s = self._orch.cosine(img_emb, dom.avg_embedding)
                out.append((d, float(s)))
            out.sort(key=lambda x: -x[1])
            return out

        if metric in ("euclidean", "l2"):
            for d, dom in dom_map.items():
                dist = self._orch.l2(img_emb, dom.avg_embedding)
                out.append((d, float(dist)))
            # distance: smaller is better
            out.sort(key=lambda x: x[1])
            return out

        # unknown metric: still return empty
        return out

def build_orchestrator(args, domains: List[str]) -> RetrievalOrchestrator:
    """
    Factory for RetrievalOrchestrator.
    Reads config from args to stay consistent with CLI flags.
    """
    sim_metric = getattr(args, "sim_metric", "cosine")
    temperature = float(getattr(args, "temperature", 0.07))
    embedder_tag = getattr(args, "embedder_tag", None)

    # your wrapper requires loradb_root
    loradb_root = str(getattr(args, "loradb", getattr(args, "loradb_root", "")))
    if not loradb_root:
        raise ValueError("[retrieval] missing --loradb (loradb_root) for build_orchestrator")

    return RetrievalOrchestrator(
        domains=domains,
        loradb_root=loradb_root,
        sim_metric=sim_metric,
        temperature=temperature,
        embedder_tag=embedder_tag,
    )

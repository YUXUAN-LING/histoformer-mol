# mol_infer/fusion/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Protocol


@dataclass
class FusionContext:
    """
    Optional runtime context passed into fusion strategies.
    Keep it lightweight; you can extend later.
    """
    meta: Optional[Dict[str, Any]] = None


class FusionStrategy(ABC):
    """
    Base interface for all fusion strategies.

    Contract:
      forward_mix(adapter, x, local_domain, global_domain, meta=...) -> (y, stats)

    - adapter: mol_infer.lora.adapter.HistoformerLoRAAdapter
    - x: input tensor [B,3,H,W] in [0,1]
    - y: output tensor [B,3,H,W] in [0,1] (or clamped later by caller)
    - stats: JSON-serializable dict
    """
    name: str = "fusion_base"

    def __init__(self, cfg: Any):
        self.cfg = cfg

    @abstractmethod
    def forward_mix(
        self,
        adapter: Any,
        x: Any,
        local_domain: str,
        global_domain: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError

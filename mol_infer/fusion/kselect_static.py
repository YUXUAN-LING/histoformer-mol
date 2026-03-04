# mol_infer/fusion/kselect_static.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from mol_infer.fusion.base import FusionStrategy
from mol_infer.fusion.utils import (
    compute_gamma_global_paramsum,
    compute_gamma_list_paramsum,
    iter_lora_named_modules,   # re-exported from mol_infer.lora.modules via import path? (see below)
    ramp_value,
    weight_score,
)

# NOTE: iter_lora_named_modules is actually in mol_infer.lora.modules, not fusion.utils.
# To avoid confusion, import it directly here:
from mol_infer.lora.modules import iter_lora_named_modules


@dataclass
class KSelectStaticConfig:
    # weight-space score
    k_topk: int = 0
    k_score_mode: str = "topk_sum"   # topk_sum/mean/median/fro/topk_ratio
    score_eps: float = 1e-12

    # ramp schedule
    k_ramp_mode: str = "sigmoid"
    k_alpha: float = 1.0
    k_beta: float = 1.0

    # gamma balancing
    use_gamma: bool = True
    gamma_mode: str = "global"       # none/global/per_layer
    gamma_clip: float = 0.0

    # caching
    cache_plan: bool = True
    verbose: bool = False


@torch.no_grad()
def build_kselect_static_plan(
    net: Any,
    local_domain: str,
    global_domain: str,
    cfg: KSelectStaticConfig,
) -> Dict[str, Any]:
    """
    Build a static per-layer selection plan (no input dependency).

    Returns plan dict containing:
      - _mods: [(name, module)] internal
      - picks: [local/global] per layer
      - gamma_global / gamma_list
      - Sc_list / Ss_list / Ss_scaled_list / S_list
      - summary stats
    """
    mods = list(iter_lora_named_modules(net))
    if len(mods) == 0:
        raise RuntimeError("[kselect-static] found 0 LoRA modules. Injection failed?")

    # gamma
    if (not cfg.use_gamma) or (str(cfg.gamma_mode).lower() == "none"):
        gamma_global = 1.0
        gamma_list = [1.0 for _ in mods]
        gamma_mode = "none"
    else:
        gamma_global = compute_gamma_global_paramsum(net, local_domain, global_domain, eps=cfg.score_eps)
        if str(cfg.gamma_mode).lower() == "per_layer":
            gamma_list = compute_gamma_list_paramsum(
                mods,
                local_domain,
                global_domain,
                eps=cfg.score_eps,
                gamma_clip=cfg.gamma_clip,
            )
            gamma_mode = "per_layer"
        else:
            gamma_list = [gamma_global for _ in mods]
            gamma_mode = "global"

    picks: List[str] = []
    names: List[str] = []
    sc_list: List[float] = []
    ss_list: List[float] = []
    ss_scaled_list: List[float] = []
    s_list: List[float] = []

    for i, (name, m) in enumerate(mods):
        pos = 0.0 if len(mods) == 1 else float(i) / float(len(mods) - 1)
        S = ramp_value(pos, cfg.k_ramp_mode, alpha=cfg.k_alpha, beta=cfg.k_beta)
        gamma_i = float(gamma_list[i])

        Sc = weight_score(m, local_domain, k_topk=cfg.k_topk, score_mode=cfg.k_score_mode, eps=cfg.score_eps)
        Ss = weight_score(m, global_domain, k_topk=cfg.k_topk, score_mode=cfg.k_score_mode, eps=cfg.score_eps)
        Ss_scaled = float(Ss * gamma_i * S)

        pick = local_domain if Sc >= Ss_scaled else global_domain

        names.append(name)
        picks.append(pick)
        sc_list.append(float(Sc))
        ss_list.append(float(Ss))
        ss_scaled_list.append(float(Ss_scaled))
        s_list.append(float(S))

        if cfg.verbose and (i < 5 or i >= len(mods) - 2):
            print(
                f"[kselect-static][{i:03d}] pos={pos:.3f} S={S:.4f} gamma={gamma_i:.4f} "
                f"Sc={Sc:.3e} Ss={Ss:.3e} Ss'={Ss_scaled:.3e} -> pick={pick} | {name}"
            )

    n_local = sum(1 for p in picks if p == local_domain)
    n_global = len(picks) - n_local

    plan = {
        "_mods": mods,
        "names": names,
        "picks": picks,
        "gamma_global": float(gamma_global),
        "gamma_mode": gamma_mode,
        "gamma_list": gamma_list,
        "k_topk": int(cfg.k_topk),
        "score_mode": str(cfg.k_score_mode),
        "ramp_mode": str(cfg.k_ramp_mode),
        "alpha": float(cfg.k_alpha),
        "beta": float(cfg.k_beta),
        "Sc_list": sc_list,
        "Ss_list": ss_list,
        "Ss_scaled_list": ss_scaled_list,
        "S_list": s_list,
        "n_layers": int(len(picks)),
        "n_local": int(n_local),
        "n_global": int(n_global),
        "local_ratio": float(n_local / max(1, len(picks))),
        "global_ratio": float(n_global / max(1, len(picks))),
        "sc_mean": float(np.mean(sc_list)) if sc_list else 0.0,
        "ss_mean": float(np.mean(ss_list)) if ss_list else 0.0,
        "ss_scaled_mean": float(np.mean(ss_scaled_list)) if ss_scaled_list else 0.0,
        "S_first": float(s_list[0]) if s_list else 0.0,
        "S_last": float(s_list[-1]) if s_list else 0.0,
    }
    return plan


@torch.no_grad()
def apply_kselect_plan(plan: Dict[str, Any], local_domain: str, global_domain: str) -> None:
    """
    Apply 2-way hard selection plan (local/global).
    """
    mods = plan["_mods"]
    picks = plan["picks"]
    for (_, m), pick in zip(mods, picks):
        if pick == local_domain:
            m.set_domain_weights({local_domain: 1.0, global_domain: 0.0})
        else:
            m.set_domain_weights({local_domain: 0.0, global_domain: 1.0})


class KSelectStaticStrategy(FusionStrategy):
    name = "kselect_static"

    def __init__(self, cfg: KSelectStaticConfig):
        super().__init__(cfg)
        self._plan_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    @torch.no_grad()
    def forward_mix(
        self,
        adapter: Any,
        x: torch.Tensor,
        local_domain: str,
        global_domain: str,
        meta: Optional[Dict[str, Any]] = None,
    ):
        key = (local_domain, global_domain)
        if self.cfg.cache_plan and key in self._plan_cache:
            plan = self._plan_cache[key]
        else:
            plan = build_kselect_static_plan(adapter.net, local_domain, global_domain, self.cfg)
            if self.cfg.cache_plan:
                self._plan_cache[key] = plan

        apply_kselect_plan(plan, local_domain, global_domain)
        y = adapter.forward(x)

        stats = {
            "fusion": self.name,
            "kselect_mode": "static",
            "score_mode": f"weight:{plan.get('score_mode','')}",
            "gamma_mode": plan.get("gamma_mode", ""),
            "gamma": float(plan.get("gamma_global", 1.0)),
            "n_layers": int(plan.get("n_layers", 0)),
            "n_local": int(plan.get("n_local", 0)),
            "n_global": int(plan.get("n_global", 0)),
            "n_none": 0,
            "local_ratio": float(plan.get("local_ratio", 0.0)),
            "global_ratio": float(plan.get("global_ratio", 0.0)),
            "none_ratio": 0.0,
            "sc_mean": float(plan.get("sc_mean", 0.0)),
            "ss_mean": float(plan.get("ss_mean", 0.0)),
            "ss_scaled_mean": float(plan.get("ss_scaled_mean", 0.0)),
            "S_first": float(plan.get("S_first", 0.0)),
            "S_last": float(plan.get("S_last", 0.0)),
        }
        return y, stats

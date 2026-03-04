# mol_infer/fusion/kselect_activation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mol_infer.fusion.base import FusionStrategy
from mol_infer.fusion.kselect_static import KSelectStaticConfig, build_kselect_static_plan
from mol_infer.fusion.utils import (
    activation_score,
    lora_delta_output,
    pad_to_factor,
)

from mol_infer.lora.modules import iter_lora_named_modules


# --- keep names for kselect_none imports ---
def _activation_score(t: torch.Tensor, mode: str = "mean_abs", eps: float = 1e-12) -> float:
    return activation_score(t, mode=mode, eps=eps)


@torch.no_grad()
def _lora_delta_output(m: Any, domain: str, x: torch.Tensor) -> torch.Tensor:
    return lora_delta_output(m, domain, x)


@dataclass
class KSelectActivationConfig:
    """
    Activation KSelect uses a static plan (for gamma/ramp + fallback),
    and optionally overrides picks on every N-th layer using activation deltas.
    """
    static: KSelectStaticConfig = KSelectStaticConfig()

    act_score_mode: str = "mean_abs"   # mean_abs/l2/rms
    act_every_n: int = 1              # run activation decision on every N-th LoRA module
    verbose: bool = False


@torch.no_grad()
def forward_padded_with_activation_kselect(
    net: Any,
    x: torch.Tensor,
    local_domain: str,
    global_domain: str,
    plan: Dict[str, Any],
    act_score_mode: str = "mean_abs",
    act_every_n: int = 1,
    verbose: bool = False,
    factor: int = 8,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Input-adaptive 2-way KSelect (local/global):
      - for each LoRA module (optionally), compute Δy_local and Δy_global on current activation
      - compare Sc_act vs Ss_act * gamma_i * S(pos)
      - otherwise fall back to plan['picks']

    Returns (y, stats)
    """
    mods: List[Tuple[str, Any]] = plan["_mods"]
    picks_static: List[str] = plan["picks"]
    gamma_list: List[float] = plan.get("gamma_list", [1.0 for _ in mods])
    S_list: List[float] = plan.get("S_list", [1.0 for _ in mods])
    names: List[str] = plan.get("names", [""] * len(mods))

    x_in, h, w = pad_to_factor(x, factor=factor)

    mod2idx = {id(m): i for i, (_, m) in enumerate(mods)}

    chosen = [None for _ in mods]
    used_act = [False for _ in mods]
    sc_list = [0.0 for _ in mods]
    ss_list = [0.0 for _ in mods]
    ss_scaled_list = [0.0 for _ in mods]

    def _should_act(i: int) -> bool:
        if act_every_n is None or act_every_n <= 0:
            return False
        return (i % int(act_every_n)) == 0

    handles = []

    def pre_hook(module, inputs):
        i = mod2idx.get(id(module), None)
        if i is None:
            return

        xin = inputs[0]
        gamma_i = float(gamma_list[i]) if gamma_list is not None else 1.0
        S = float(S_list[i]) if S_list is not None else 1.0

        if _should_act(i):
            dl = _lora_delta_output(module, local_domain, xin)
            dg = _lora_delta_output(module, global_domain, xin)
            Sc = _activation_score(dl, mode=act_score_mode, eps=eps)
            Ss = _activation_score(dg, mode=act_score_mode, eps=eps)
            Ss_scaled = float(Ss * gamma_i * S)

            pick = local_domain if Sc >= Ss_scaled else global_domain
            used_act[i] = True
            sc_list[i] = float(Sc)
            ss_list[i] = float(Ss)
            ss_scaled_list[i] = float(Ss_scaled)

            if verbose and (i < 5 or i >= len(mods) - 2):
                print(
                    f"[kselect-act][{i:03d}] S={S:.4f} gamma={gamma_i:.4f} "
                    f"Sc_act={Sc:.3e} Ss_act={Ss:.3e} Ss'={Ss_scaled:.3e} -> pick={pick} | {names[i]}"
                )
        else:
            pick = picks_static[i]
            # log static scores if exist
            Sc = float(plan.get("Sc_list", [0.0] * len(mods))[i])
            Ss = float(plan.get("Ss_list", [0.0] * len(mods))[i])
            Ss_scaled = float(plan.get("Ss_scaled_list", [0.0] * len(mods))[i])
            sc_list[i] = Sc
            ss_list[i] = Ss
            ss_scaled_list[i] = Ss_scaled

        chosen[i] = pick

        if pick == local_domain:
            module.set_domain_weights({local_domain: 1.0, global_domain: 0.0})
        else:
            module.set_domain_weights({local_domain: 0.0, global_domain: 1.0})

    for _, m in mods:
        handles.append(m.register_forward_pre_hook(pre_hook))

    try:
        y = net(x_in)[:, :, :h, :w]
    finally:
        for hdl in handles:
            try:
                hdl.remove()
            except Exception:
                pass

    n_local = sum(1 for p in chosen if p == local_domain)
    n_global = len(chosen) - n_local
    act_cnt = sum(1 for u in used_act if u)

    stats = {
        "fusion": "kselect_activation",
        "kselect_mode": f"activation(every_n={act_every_n})",
        "score_mode": f"act:{act_score_mode} + static:{plan.get('score_mode','')}",
        "gamma_mode": plan.get("gamma_mode", ""),
        "gamma": float(plan.get("gamma_global", 1.0)),
        "n_layers": int(len(chosen)),
        "n_local": int(n_local),
        "n_global": int(n_global),
        "n_none": 0,
        "local_ratio": float(n_local / max(1, len(chosen))),
        "global_ratio": float(n_global / max(1, len(chosen))),
        "none_ratio": 0.0,
        "act_layers": int(act_cnt),
        "act_ratio": float(act_cnt / max(1, len(chosen))),
        "sc_mean": float(np.mean(sc_list)) if sc_list else 0.0,
        "ss_mean": float(np.mean(ss_list)) if ss_list else 0.0,
        "ss_scaled_mean": float(np.mean(ss_scaled_list)) if ss_scaled_list else 0.0,
        "S_first": float(plan.get("S_first", 0.0)),
        "S_last": float(plan.get("S_last", 0.0)),
    }
    return y, stats


class KSelectActivationStrategy(FusionStrategy):
    name = "kselect_activation"

    def __init__(self, cfg: KSelectActivationConfig):
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
        static_cfg = self.cfg.static

        if static_cfg.cache_plan and key in self._plan_cache:
            plan = self._plan_cache[key]
        else:
            plan = build_kselect_static_plan(adapter.net, local_domain, global_domain, static_cfg)
            if static_cfg.cache_plan:
                self._plan_cache[key] = plan

        y, stats = forward_padded_with_activation_kselect(
            adapter.net,
            x,
            local_domain=local_domain,
            global_domain=global_domain,
            plan=plan,
            act_score_mode=self.cfg.act_score_mode,
            act_every_n=self.cfg.act_every_n,
            verbose=self.cfg.verbose,
            factor=8,
            eps=static_cfg.score_eps,
        )
        return y, stats

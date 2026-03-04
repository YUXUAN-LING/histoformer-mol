# mol_infer/fusion/kselect_none.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch

from mol_infer.fusion.base import FusionStrategy
from mol_infer.fusion.kselect_static import (
    build_kselect_static_plan,
    apply_kselect_plan,
)
from mol_infer.fusion.kselect_activation import (
    forward_padded_with_activation_kselect,
)
from mol_infer.fusion.utils import none_threshold_value


@dataclass
class KSelectNoneConfig:
    # base kselect params
    kselect_mode: str = "static"          # static / activation / hybrid
    k_score_mode: str = "topk_ratio"      # topk_sum / mean / median / fro / topk_ratio
    k_topk: int = 0
    k_alpha: float = 1.0
    k_beta: float = 1.0
    k_ramp_mode: str = "sigmoid"
    use_gamma: bool = True
    gamma_mode: str = "per_layer"         # none / global / per_layer
    gamma_clip: float = 0.0
    score_eps: float = 1e-12

    # activation params
    act_score_mode: str = "mean_abs"      # mean_abs / l2 / rms
    act_every_n: int = 1                  # hybrid: only every N-th layer use activation

    # none-threshold schedule
    none_mode: str = "const"              # const / linear / sigmoid
    none_beta: float = 0.15               # base threshold
    none_alpha: float = 0.0               # ramp amplitude
    # compare metric: "abs" uses raw score; "ratio" uses max_score / (mean_score+eps)
    none_metric: str = "abs"              # abs / ratio
    none_ratio_eps: float = 1e-12


class KSelectNone(FusionStrategy):
    """
    Three-choice KSelect: local / global / none(base)

    For each LoRA module i:
      - compute Sc, Ss_scaled (static or activation)
      - compute T_none(pos)
      - if max(Sc, Ss_scaled) < T_none -> none (disable both domains)
        else pick local/global by Sc vs Ss_scaled
    """

    name: str = "kselect_none"

    def __init__(self, cfg: Optional[KSelectNoneConfig] = None):
        self.cfg = cfg or KSelectNoneConfig()

    @torch.no_grad()
    def run(
        self,
        adapter,
        x: torch.Tensor,
        local_domain: str,
        global_domain: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        adapter: your mol_infer.lora.adapter.HistoformerLoRAAdapter
        must provide:
          - net (torch.nn.Module)
          - forward_padded(x) -> y
          - set_all_domain_weights(dict or None)
        """
        net = adapter.net
        c = self.cfg

        # 1) build base static plan (gives: mods, names, gamma_list, S_list, Sc/Ss/Ss_scaled)
        plan = build_kselect_static_plan(
            net,
            local_domain=local_domain,
            global_domain=global_domain,
            k_topk=c.k_topk,
            alpha=c.k_alpha,
            beta=c.k_beta,
            ramp_mode=c.k_ramp_mode,
            score_mode=c.k_score_mode,
            use_gamma=bool(c.use_gamma),
            gamma_mode=c.gamma_mode,
            gamma_clip=c.gamma_clip,
            eps=c.score_eps,
            verbose=False,
        )

        # 2) Depending on mode, get scores per layer (Sc, Ss_scaled) and pick including none
        if c.kselect_mode == "static":
            picks, stats = self._pick_with_none_static(plan, local_domain, global_domain)
            # apply hard gating (including none)
            self._apply_picks(net, plan, picks, local_domain, global_domain)
            y = adapter.forward_padded(x)
            return y, stats

        # activation / hybrid:
        # We reuse your existing activation forward, but we need it to output chosen picks and scores.
        # Easiest: we call a new helper here that replays the activation hooks and also does none.
        y, stats = self._forward_activation_with_none(adapter, x, plan, local_domain, global_domain)
        return y, stats

    def _score_to_none_metric(self, Sc: float, Ss_scaled: float) -> float:
        c = self.cfg
        if c.none_metric == "ratio":
            meanv = 0.5 * (Sc + Ss_scaled)
            return float(max(Sc, Ss_scaled) / (meanv + c.none_ratio_eps))
        return float(max(Sc, Ss_scaled))

    def _pick_with_none_static(
        self,
        plan: Dict[str, Any],
        local_domain: str,
        global_domain: str,
    ) -> Tuple[List[str], Dict[str, Any]]:
        c = self.cfg
        mods = plan["_mods"]
        Sc_list = plan["Sc_list"]
        Ss_scaled_list = plan["Ss_scaled_list"]

        picks: List[str] = []
        n_local = n_global = n_none = 0

        for i in range(len(mods)):
            pos = 0.0 if len(mods) == 1 else float(i) / float(len(mods) - 1)
            Tn = none_threshold_value(pos, mode=c.none_mode, alpha=c.none_alpha, beta=c.none_beta)

            Sc = float(Sc_list[i])
            Ss_scaled = float(Ss_scaled_list[i])

            mval = self._score_to_none_metric(Sc, Ss_scaled)
            if mval < Tn:
                picks.append("none")
                n_none += 1
            else:
                if Sc >= Ss_scaled:
                    picks.append(local_domain)
                    n_local += 1
                else:
                    picks.append(global_domain)
                    n_global += 1

        stats = {
            "fusion": self.name,
            "kselect_mode": "static+none",
            "k_score_mode": f"weight:{plan.get('score_mode','')}",
            "gamma_mode": plan.get("gamma_mode", ""),
            "gamma": float(plan.get("gamma_global", 1.0)),
            "none_mode": c.none_mode,
            "none_beta": float(c.none_beta),
            "none_alpha": float(c.none_alpha),
            "none_metric": c.none_metric,
            "n_layers": int(len(mods)),
            "n_local": int(n_local),
            "n_global": int(n_global),
            "n_none": int(n_none),
            "local_ratio": float(n_local / max(1, len(mods))),
            "global_ratio": float(n_global / max(1, len(mods))),
            "none_ratio": float(n_none / max(1, len(mods))),
            "S_first": float(plan.get("S_first", 0.0)),
            "S_last": float(plan.get("S_last", 0.0)),
        }
        return picks, stats

    def _apply_picks(self, net, plan, picks: List[str], local_domain: str, global_domain: str):
        mods = plan["_mods"]
        for (_, m), pick in zip(mods, picks):
            if pick == "none":
                m.set_domain_weights({local_domain: 0.0, global_domain: 0.0})
            elif pick == local_domain:
                m.set_domain_weights({local_domain: 1.0, global_domain: 0.0})
            else:
                m.set_domain_weights({local_domain: 0.0, global_domain: 1.0})

    @torch.no_grad()
    def _forward_activation_with_none(
        self,
        adapter,
        x: torch.Tensor,
        plan: Dict[str, Any],
        local_domain: str,
        global_domain: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        We extend the activation kselect forward by post-processing with none threshold.
        Implementation approach:
          - run activation-kselect forward once to obtain per-layer chosen picks and scores
          - then re-run a second pass? (not acceptable)
        So we implement our own hook forward (similar to your forward_padded_with_activation_kselect),
        but now with three-choice.
        """
        net = adapter.net
        c = self.cfg

        mods = plan["_mods"]
        gamma_list = plan.get("gamma_list", [1.0 for _ in mods])
        S_list = plan.get("S_list", [1.0 for _ in mods])
        picks_static = plan.get("picks", [])

        # Ensure no leakage from other domains before per-layer gating
        # (adapter should know all domains; if not, leave as-is)
        try:
            adapter.set_all_domain_weights_zero()
        except Exception:
            # fallback: do nothing, hooks will set local/global anyway
            pass

        mod2idx = {id(m): i for i, (_, m) in enumerate(mods)}
        chosen = ["none" for _ in mods]
        used_act = [False for _ in mods]

        # store scores for stats
        sc_list = [0.0 for _ in mods]
        ss_scaled_list = [0.0 for _ in mods]
        none_list = [0.0 for _ in mods]

        def _should_act(i: int) -> bool:
            if c.act_every_n <= 0:
                return False
            return (i % c.act_every_n) == 0

        handles = []

        def pre_hook(module, inputs):
            i = mod2idx.get(id(module), None)
            if i is None:
                return
            xin = inputs[0]
            pos = 0.0 if len(mods) == 1 else float(i) / float(len(mods) - 1)
            Tn = none_threshold_value(pos, mode=c.none_mode, alpha=c.none_alpha, beta=c.none_beta)
            none_list[i] = float(Tn)

            if _should_act(i):
                used_act[i] = True
                # reuse your activation helper, but we need deltas; call the one in activation module:
                # forward_padded_with_activation_kselect already computes dl/dg internally, but doesn't expose them.
                # So we import the helper from that module by calling it is not possible.
                # => We instead call the internal utilities through module methods (LoRA up/down).
                from mol_infer.fusion.kselect_activation import _lora_delta_output, _activation_score

                dl = _lora_delta_output(module, local_domain, xin)
                dg = _lora_delta_output(module, global_domain, xin)
                Sc = _activation_score(dl, mode=c.act_score_mode, eps=c.score_eps)
                Ss = _activation_score(dg, mode=c.act_score_mode, eps=c.score_eps)

                gamma_i = float(gamma_list[i]) if gamma_list is not None else 1.0
                S = float(S_list[i]) if S_list is not None else 1.0
                Ss_scaled = float(Ss * gamma_i * S)

                sc_list[i] = float(Sc)
                ss_scaled_list[i] = float(Ss_scaled)

                mval = self._score_to_none_metric(Sc, Ss_scaled)
                if mval < Tn:
                    pick = "none"
                else:
                    pick = local_domain if Sc >= Ss_scaled else global_domain
            else:
                # static fallback (for hybrid)
                Sc = float(plan.get("Sc_list", [0.0] * len(mods))[i])
                Ss_scaled = float(plan.get("Ss_scaled_list", [0.0] * len(mods))[i])
                sc_list[i] = Sc
                ss_scaled_list[i] = Ss_scaled

                mval = self._score_to_none_metric(Sc, Ss_scaled)
                if mval < Tn:
                    pick = "none"
                else:
                    pick = picks_static[i] if i < len(picks_static) else (local_domain if Sc >= Ss_scaled else global_domain)

            chosen[i] = pick

            if pick == "none":
                module.set_domain_weights({local_domain: 0.0, global_domain: 0.0})
            elif pick == local_domain:
                module.set_domain_weights({local_domain: 1.0, global_domain: 0.0})
            else:
                module.set_domain_weights({local_domain: 0.0, global_domain: 1.0})

        for _, m in mods:
            handles.append(m.register_forward_pre_hook(pre_hook))

        try:
            y = adapter.forward_padded(x)
        finally:
            for hdl in handles:
                try:
                    hdl.remove()
                except Exception:
                    pass

        n_local = sum(1 for p in chosen if p == local_domain)
        n_global = sum(1 for p in chosen if p == global_domain)
        n_none = sum(1 for p in chosen if p == "none")
        act_cnt = sum(1 for u in used_act if u)

        stats = {
            "fusion": self.name,
            "kselect_mode": f"{c.kselect_mode}+none(act_every_n={c.act_every_n})",
            "score_mode": f"act:{c.act_score_mode} + static:{plan.get('score_mode','')}",
            "gamma_mode": plan.get("gamma_mode", ""),
            "gamma": float(plan.get("gamma_global", 1.0)),
            "none_mode": c.none_mode,
            "none_beta": float(c.none_beta),
            "none_alpha": float(c.none_alpha),
            "none_metric": c.none_metric,
            "n_layers": int(len(mods)),
            "n_local": int(n_local),
            "n_global": int(n_global),
            "n_none": int(n_none),
            "local_ratio": float(n_local / max(1, len(mods))),
            "global_ratio": float(n_global / max(1, len(mods))),
            "none_ratio": float(n_none / max(1, len(mods))),
            "act_layers": int(act_cnt),
            "act_ratio": float(act_cnt / max(1, len(mods))),
            "Tnone_first": float(none_list[0]) if none_list else 0.0,
            "Tnone_last": float(none_list[-1]) if none_list else 0.0,
            "S_first": float(plan.get("S_first", 0.0)),
            "S_last": float(plan.get("S_last", 0.0)),
        }
        return y, stats

# keep backward-compatible name
KSelectNoneStrategy = KSelectNone
__all__ = ["KSelectNoneConfig", "KSelectNone", "KSelectNoneStrategy"]

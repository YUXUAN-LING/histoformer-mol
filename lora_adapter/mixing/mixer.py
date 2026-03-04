# -*- coding: utf-8 -*-
"""lora_adapters.mixing.mixer

✅ Mixing strategies live HERE.

Pipeline contract
-----------------
The pipeline calls:
  y, mix_info = mixer.apply(mode, x indicate..., decision=Decision, route=RouteResult, **mix_args)

So you can change / add mixing strategies by editing ONLY this file.

Strategies included
-------------------
- single:          enable only top1 (or decision.top1)
- linear:          enable local + global simultaneously with fixed weights
- ramp:            depth-dependent weights (smooth transition across layers)
- kselect_static:  per-layer hard selection based on LoRA weights
- kselect_hybrid:  activation-based selection every N layers + static fallback
- kselect_activation: activation selection every layer
- act_kselect_dy:  ✅ DY-KSelect (activation-aware delta-output scoring + TopK/Tau + optional BOTH)

LoRA weight loading
-------------------
We assume the model has been injected with multi-domain LoRA slots (LoRAConv2d/LoRALinear
with ModuleDict lora_down/domain and lora_up/domain).

We load each single-domain LoRA ckpt and remap its keys into the multi-domain slots.

This design keeps retrieval/decision decoupled from mixing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from lora_adapters.lora_linear import LoRALinear, LoRAConv2d
except Exception as e:  # pragma: no cover
    raise ImportError("Cannot import LoRALinear/LoRAConv2d. Ensure lora_adapters/lora_linear.py exists.") from e

from lora_adapter.decision.decide import Decision
from lora_adapter.retrieval.router import RouteResult
from lora_adapter.retrieval.prototype_bank import PrototypeBank


# -----------------------------------------------------------------------------
# Common utilities
# -----------------------------------------------------------------------------

def iter_lora_named_modules(net):
    for name, m in net.named_modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            yield name, m


def collect_lora_modules(net) -> List[Tuple[str, Any]]:
    return list(iter_lora_named_modules(net))


def reset_all_lora_weights(net):
    """Disable LoRA by clearing domain_weights on every LoRA module."""
    for _, m in iter_lora_named_modules(net):
        # NOTE: keeping None is OK for multi-domain inference (domain_list>1 -> no LoRA).
        # If someone accidentally injected a single-domain LoRA (domain_list==1),
        # None would enable that LoRA. To be safe for all cases, we reset to an
        # empty dict which always means "no contribution".
        m.set_domain_weights({})


def set_all_lora_weights(net, weights: Dict[str, float]):
    """Set the same domain weights dict for all LoRA modules."""
    for _, m in iter_lora_named_modules(net):
        m.set_domain_weights(weights)


def forward_padded(net, x: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """Pad input by reflect to be divisible by `factor`, then crop back."""
    _, _, h, w = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else x
    y = net(x_in)
    if isinstance(y, list):
        y = y[-1]
    y = y[:, :, :h, :w]
    return y


# -----------------------------------------------------------------------------
# LoRA loading into multi-domain slots
# -----------------------------------------------------------------------------

def _unwrap_state_dict(sd: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "params" in sd and isinstance(sd["params"], dict):
        sd = sd["params"]
    if isinstance(sd, dict) and "params_ema" in sd and isinstance(sd["params_ema"], dict):
        sd = sd["params_ema"]
    # keep only tensors
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v
    return out


def _remap_lora_keys_to_domain(sd: Dict[str, torch.Tensor], domain: str) -> Dict[str, torch.Tensor]:
    """Remap keys from a single-domain LoRA checkpoint into multi-domain slots.

    Supported source key patterns (both appear in wild repos):
      - ...lora_down.weight / ...lora_up.weight
      - ...lora_down.<something>.weight / ...lora_up.<something>.weight

    Target pattern expected by our multi-domain LoRA modules:
      - ...lora_down.<domain>.weight
      - ...lora_up.<domain>.weight

    We only remap LoRA keys; other keys are ignored.
    """
    mapped: Dict[str, torch.Tensor] = {}

    for k, v in sd.items():
        if "lora_down" not in k and "lora_up" not in k:
            continue
        parts = k.split(".")
        # find token
        for i, token in enumerate(parts):
            if token in ("lora_down", "lora_up"):
                # cases:
                # 1) ... lora_down weight
                if i + 1 < len(parts) and parts[i + 1] == "weight":
                    new_parts = parts[: i + 1] + [domain] + parts[i + 1 :]
                    mapped[".".join(new_parts)] = v
                    break
                # 2) ... lora_down <dom> weight
                if i + 2 < len(parts) and parts[i + 2] == "weight":
                    new_parts = parts.copy()
                    new_parts[i + 1] = domain
                    mapped[".".join(new_parts)] = v
                    break

    return mapped


def load_all_domain_loras(net, bank: PrototypeBank, strict: bool = False):
    """Load each domain LoRA checkpoint into the model's multi-domain slots."""
    missing = []
    for d in bank.names():
        ckpt = bank.get(d).lora_path
        sd = torch.load(ckpt, map_location="cpu")
        sd = _unwrap_state_dict(sd)
        mapped = _remap_lora_keys_to_domain(sd, d)
        if not mapped:
            missing.append(d)
            continue
        net.load_state_dict(mapped, strict=strict)
    if missing:
        print(f"[Mixer] WARN: no LoRA keys remapped for domains={missing}. "
              f"(ckpt format may not include lora_down/lora_up?)")


# -----------------------------------------------------------------------------
# Mixer registry
# -----------------------------------------------------------------------------

StrategyFn = Callable[["Mixer", torch.Tensor, Decision, RouteResult, Dict[str, Any]], Tuple[torch.Tensor, Dict[str, Any]]]


class MixerRegistry:
    def __init__(self):
        self._fns: Dict[str, StrategyFn] = {}

    def register(self, name: str):
        def deco(fn: StrategyFn):
            self._fns[name] = fn
            return fn
        return deco

    def get(self, name: str) -> StrategyFn:
        if name not in self._fns:
            raise KeyError(f"Unknown mix_mode='{name}'. Available: {sorted(self._fns.keys())}")
        return self._fns[name]

    def names(self) -> List[str]:
        return sorted(self._fns.keys())


@dataclass
class Mixer:
    """Holds model reference and applies registered mixing strategies."""

    net: torch.nn.Module
    factor: int = 8  # padding factor

    def __post_init__(self):
        self.lora_modules = collect_lora_modules(self.net)
        self.registry = MIXERS

    def disable_lora(self):
        reset_all_lora_weights(self.net)

    def forward_base(self, x: torch.Tensor) -> torch.Tensor:
        self.disable_lora()
        return forward_padded(self.net, x, factor=self.factor)

    def apply(self, mix_mode: str, x: torch.Tensor, decision: Decision, route: RouteResult, **mix_args):
        fn = self.registry.get(mix_mode)
        return fn(self, x, decision, route, mix_args)


MIXERS = MixerRegistry()


# -----------------------------------------------------------------------------
# Strategy: single / linear / ramp
# -----------------------------------------------------------------------------

@MIXERS.register("single")
def mix_single(ctx: Mixer, x: torch.Tensor, decision: Decision, route: RouteResult, args: Dict[str, Any]):
    # choose domain
    domain = decision.top1
    ctx.disable_lora()
    set_all_lora_weights(ctx.net, {domain: 1.0})

    y = forward_padded(ctx.net, x, factor=ctx.factor)
    info = {
        "mix_mode": "single",
        "domain": domain,
        "weights": {domain: 1.0},
    }
    return y, info


@MIXERS.register("linear")
def mix_linear(ctx: Mixer, x: torch.Tensor, decision: Decision, route: RouteResult, args: Dict[str, Any]):
    """Enable local+global simultaneously with fixed weights."""
    ctx.disable_lora()

    if decision.mode == "single" or not decision.local or not decision.global_:
        return mix_single(ctx, x, decision, route, args)

    normalize = bool(args.get("normalize", True))
    w_local = float(args.get("w_local", decision.local_w))
    w_global = float(args.get("w_global", decision.global_w))

    if normalize:
        s = w_local + w_global
        if s > 1e-12:
            w_local /= s
            w_global /= s

    weights = {decision.local: w_local, decision.global_: w_global}
    set_all_lora_weights(ctx.net, weights)

    y = forward_padded(ctx.net, x, factor=ctx.factor)
    info = {
        "mix_mode": "linear",
        "local": decision.local,
        "global": decision.global_,
        "weights": weights,
        "normalize": normalize,
    }
    return y, info


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _ramp(pos01: float, mode: str = "sigmoid") -> float:
    pos01 = float(max(0.0, min(1.0, pos01)))
    if mode == "linear":
        return pos01
    # sigmoid centered at 0.5
    return _sigmoid(6.0 * (pos01 - 0.5))


@MIXERS.register("ramp")
def mix_ramp(ctx: Mixer, x: torch.Tensor, decision: Decision, route: RouteResult, args: Dict[str, Any]):
    """Depth-dependent weights.

    S(pos) in [0,1] controls global share.
      w_global(pos) = S(pos)
      w_local(pos)  = 1 - S(pos)

    Optional:
      - multiply by retrieval weights (decision.local_w/global_w)
      - normalize per-layer
    """
    ctx.disable_lora()

    if decision.mode == "single" or not decision.local or not decision.global_:
        return mix_single(ctx, x, decision, route, args)

    ramp_mode = str(args.get("ramp_mode", "sigmoid"))
    use_retrieval_weight = bool(args.get("use_retrieval_weight", True))
    normalize = bool(args.get("normalize", True))

    local_w0 = float(decision.local_w) if use_retrieval_weight else 1.0
    global_w0 = float(decision.global_w) if use_retrieval_weight else 1.0

    mods = ctx.lora_modules
    n = len(mods)
    for i, (_, m) in enumerate(mods):
        pos = 0.0 if n <= 1 else float(i) / float(n - 1)
        S = _ramp(pos, mode=ramp_mode)
        w_global = global_w0 * S
        w_local = local_w0 * (1.0 - S)
        if normalize:
            s = w_global + w_local
            if s > 1e-12:
                w_global /= s
                w_local /= s
        m.set_domain_weights({decision.local: w_local, decision.global_: w_global})

    y = forward_padded(ctx.net, x, factor=ctx.factor)
    info = {
        "mix_mode": "ramp",
        "local": decision.local,
        "global": decision.global_,
        "ramp_mode": ramp_mode,
        "use_retrieval_weight": use_retrieval_weight,
        "normalize": normalize,
    }
    return y, info


# -----------------------------------------------------------------------------
# KSelect v2 (ported & simplified from your infer_data_kselect_v2.py)
# -----------------------------------------------------------------------------

def _get_domain_up_down(m, domain: str):
    if domain not in m.lora_up or domain not in m.lora_down:
        raise KeyError(f"domain='{domain}' not found in LoRA module")
    return m.lora_up[domain], m.lora_down[domain]


def _lora_scale(m) -> float:
    alpha = float(getattr(m, "alpha", 1.0))
    r = int(getattr(m, "rank", getattr(m, "r", 0)) or 0)
    if r <= 0:
        return 1.0
    return alpha / float(r)


@torch.no_grad()
def _layer_param_abs_sum(m, domain: str) -> float:
    up, down = _get_domain_up_down(m, domain)
    return float(up.weight.abs().sum().item() + down.weight.abs().sum().item())


@torch.no_grad()
def _delta_abs_matrix_proxy(m, domain: str) -> Optional[torch.Tensor]:
    """Build |ΔW| proxy: |B| @ |A|.

    Works for both Linear and Conv2d LoRA (Conv kernels collapsed).
    """
    up, down = _get_domain_up_down(m, domain)
    A = down.weight
    B = up.weight

    A2 = A.reshape(A.shape[0], -1).float().abs()  # [r, in_flat]
    Bf = B.reshape(B.shape[0], -1).float().abs()  # [out, r_flat]

    if Bf.shape[1] != A2.shape[0]:
        r = A2.shape[0]
        if Bf.numel() % r == 0:
            try:
                Bf = Bf.reshape(B.shape[0], r, -1).sum(dim=-1)
            except Exception:
                return None
        else:
            return None

    try:
        return torch.matmul(Bf, A2)
    except Exception:
        return None


@torch.no_grad()
def _weight_score(m, domain: str, k_topk: int, score_mode: str, eps: float = 1e-12) -> float:
    delta = _delta_abs_matrix_proxy(m, domain)
    if delta is None:
        return _layer_param_abs_sum(m, domain)

    flat = delta.reshape(-1)

    if score_mode == "mean":
        return float(flat.mean().item())
    if score_mode == "median":
        return float(flat.median().item())
    if score_mode == "fro":
        return float(torch.linalg.norm(delta, ord="fro").item())
    if score_mode == "topk_ratio":
        total = float(flat.sum().item())
        if total <= eps:
            return 0.0
        if k_topk <= 0 or k_topk >= flat.numel():
            top_sum = total
        else:
            top_sum = float(torch.topk(flat, k=k_topk, largest=True).values.sum().item())
        return float(top_sum / (total + eps))

    # default: topk_sum
    if k_topk <= 0 or k_topk >= flat.numel():
        return float(flat.sum().item())
    return float(torch.topk(flat, k=k_topk, largest=True).values.sum().item())


@torch.no_grad()
def _compute_gamma_global(net, local_domain: str, global_domain: str, eps: float = 1e-12) -> float:
    sl, sg = 0.0, 0.0
    for _, m in iter_lora_named_modules(net):
        sl += _layer_param_abs_sum(m, local_domain)
        sg += _layer_param_abs_sum(m, global_domain)
    if sg <= eps:
        return 1.0
    return float(sl / (sg + eps))


@torch.no_grad()
def _compute_gamma_list_per_layer(mods: List[Tuple[str, Any]], local_domain: str, global_domain: str, eps: float = 1e-12, gamma_clip: float = 0.0) -> List[float]:
    out = []
    for _, m in mods:
        gl = _layer_param_abs_sum(m, local_domain)
        gg = _layer_param_abs_sum(m, global_domain)
        g = float(gl / (gg + eps)) if gg > eps else 1.0
        if gamma_clip and gamma_clip > 0:
            lo = 1.0 / float(gamma_clip)
            hi = float(gamma_clip)
            g = float(min(hi, max(lo, g)))
        out.append(g)
    return out


def _ramp_scaled(pos01: float, mode: str, alpha: float, beta: float) -> float:
    pos01 = float(max(0.0, min(1.0, pos01)))
    if mode == "linear":
        return float(alpha * pos01 + beta)
    if mode == "sigmoid":
        return float(alpha * _sigmoid(6.0 * (pos01 - 0.5)) + beta)
    return float(alpha * pos01 + beta)


def _none_tau(pos01: float, base: float, alpha: float, mode: str) -> float:
    pos01 = float(max(0.0, min(1.0, pos01)))
    if mode == "sigmoid":
        return float(base + alpha * _sigmoid(6.0 * (pos01 - 0.5)))
    return float(base + alpha * pos01)


@torch.no_grad()
def _build_kselect_plan(
    net,
    local_domain: str,
    global_domain: str,
    k_topk: int,
    score_mode: str,
    ramp_mode: str,
    k_alpha: float,
    k_beta: float,
    use_gamma: bool,
    gamma_mode: str,
    gamma_clip: float,
    eps: float,
    none_tau_base: float,
    none_tau_alpha: float,
    none_tau_mode: str,
):
    mods = collect_lora_modules(net)
    if not mods:
        raise RuntimeError("[kselect] found 0 LoRA modules. Injection failed?")

    if (not use_gamma) or (gamma_mode == "none"):
        gamma_global = 1.0
        gamma_list = [1.0] * len(mods)
    else:
        gamma_global = _compute_gamma_global(net, local_domain, global_domain, eps=eps)
        if gamma_mode == "per_layer":
            gamma_list = _compute_gamma_list_per_layer(mods, local_domain, global_domain, eps=eps, gamma_clip=gamma_clip)
        else:
            gamma_list = [gamma_global] * len(mods)

    picks: List[str] = []
    names: List[str] = []
    Sc_list: List[float] = []
    Ss_list: List[float] = []
    Ss_scaled_list: List[float] = []
    S_list: List[float] = []
    none_list: List[bool] = []

    n = len(mods)
    for i, (name, m) in enumerate(mods):
        pos = 0.0 if n <= 1 else float(i) / float(n - 1)
        S = _ramp_scaled(pos, ramp_mode, alpha=k_alpha, beta=k_beta)
        gamma_i = float(gamma_list[i])

        Sc = _weight_score(m, local_domain, k_topk=k_topk, score_mode=score_mode, eps=eps)
        Ss = _weight_score(m, global_domain, k_topk=k_topk, score_mode=score_mode, eps=eps)
        Ss_scaled = float(Ss * gamma_i * S)

        # optional none threshold
        tau_none = _none_tau(pos, base=none_tau_base, alpha=none_tau_alpha, mode=none_tau_mode)
        best = max(float(Sc), float(Ss_scaled))
        if best < tau_none:
            pick = "none"
            none_flag = True
        else:
            pick = local_domain if Sc >= Ss_scaled else global_domain
            none_flag = False

        picks.append(pick)
        names.append(name)
        Sc_list.append(float(Sc))
        Ss_list.append(float(Ss))
        Ss_scaled_list.append(float(Ss_scaled))
        S_list.append(float(S))
        none_list.append(bool(none_flag))

    return {
        "mods": mods,
        "names": names,
        "picks": picks,
        "none": none_list,
        "gamma_global": float(gamma_global),
        "gamma_list": gamma_list,
        "Sc_list": Sc_list,
        "Ss_list": Ss_list,
        "Ss_scaled_list": Ss_scaled_list,
        "S_list": S_list,
        "params": {
            "k_topk": int(k_topk),
            "score_mode": str(score_mode),
            "ramp_mode": str(ramp_mode),
            "k_alpha": float(k_alpha),
            "k_beta": float(k_beta),
            "use_gamma": bool(use_gamma),
            "gamma_mode": str(gamma_mode),
            "gamma_clip": float(gamma_clip),
            "eps": float(eps),
            "none_tau": float(none_tau_base),
            "none_tau_alpha": float(none_tau_alpha),
            "none_tau_mode": str(none_tau_mode),
        },
    }


def _apply_kselect_plan(net, plan: Dict[str, Any], local_domain: str, global_domain: str):
    for (_, m), pick in zip(plan["mods"], plan["picks"]):
        if pick == "none":
            m.set_domain_weights({local_domain: 0.0, global_domain: 0.0})
        elif pick == local_domain:
            m.set_domain_weights({local_domain: 1.0, global_domain: 0.0})
        else:
            m.set_domain_weights({local_domain: 0.0, global_domain: 1.0})


def _activation_score(t: torch.Tensor, mode: str, eps: float = 1e-12) -> float:
    if mode == "l2":
        return float(torch.linalg.norm(t.reshape(-1).float(), ord=2).item())
    if mode == "rms":
        return float(torch.sqrt((t.float() ** 2).mean() + eps).item())
    return float(t.float().abs().mean().item())


@torch.no_grad()
def _rms_tensor(t: Optional[torch.Tensor], eps: float = 1e-12) -> float:
    if t is None or (not torch.is_tensor(t)) or t.numel() == 0:
        return 0.0
    tt = t.float()
    return float(torch.sqrt((tt * tt).mean() + eps).item())


@torch.no_grad()
def _lora_delta_output(m, domain: str, x: torch.Tensor) -> torch.Tensor:
    up, down = _get_domain_up_down(m, domain)
    y = up(down(x))
    scale = _lora_scale(m)
    if abs(scale - 1.0) > 1e-12:
        y = y * scale
    return y


@torch.no_grad()
def _deltaW_fro_norm(m, domain: str, eps: float = 1e-12) -> float:
    """Debug helper: ||ΔW||_F where ΔW is the effective weight delta.

    - Linear:  ΔW = Up @ Down
    - Conv2d:  Down is 1x1 conv (r,in,1,1) in our injection;
              Up is kxk conv (out,r,k,k) sharing stride/pad with base.
              Effective ΔW is (out,in,k,k): ΔW[o,i,h,w] = Σ_r Up[o,r,h,w] * Down[r,i,0,0]

    We include the LoRA scale (alpha/r).

    This is ONLY for logging/debug (not used for decisions in DY-KSelect).
    """
    if (domain not in m.lora_up) or (domain not in m.lora_down):
        return 0.0
    up = m.lora_up[domain]
    down = m.lora_down[domain]
    scale = float(_lora_scale(m))

    try:
        # Linear
        if isinstance(up, torch.nn.Linear) and isinstance(down, torch.nn.Linear):
            dw = torch.matmul(up.weight.float(), down.weight.float()) * scale
            return float(torch.linalg.norm(dw, ord="fro").item())

        # Conv2d (our LoRAConv2d design: down=1x1, up=kxk)
        if isinstance(up, torch.nn.Conv2d) and isinstance(down, torch.nn.Conv2d):
            A = down.weight.float()  # [r,in,1,1]
            B = up.weight.float()    # [out,r,k,k]
            if A.ndim != 4 or B.ndim != 4:
                return 0.0
            if A.shape[2] != 1 or A.shape[3] != 1:
                # not 1x1: fall back to a cheap proxy
                return float((B.abs().mean() * A.abs().mean()).item())
            A2 = A[:, :, 0, 0]  # [r,in]
            # ΔW[o,i,h,w] = Σ_r B[o,r,h,w] * A2[r,i]
            dw = torch.einsum("orhw,ri->oihw", B, A2) * scale
            return float(torch.linalg.norm(dw.reshape(-1), ord=2).item())
    except Exception:
        return 0.0

    return 0.0


@torch.no_grad()
def _forward_with_activation_kselect(
    net,
    x: torch.Tensor,
    plan: Dict[str, Any],
    local_domain: str,
    global_domain: str,
    act_score_mode: str,
    act_every_n: int,
    eps: float,
    factor: int = 8,
):
    mods = plan["mods"]
    picks_static = plan["picks"]
    gamma_list = plan.get("gamma_list", [1.0] * len(mods))
    S_list = plan.get("S_list", [1.0] * len(mods))
    none_tau_base = float(plan["params"].get("none_tau", 0.0))
    none_tau_alpha = float(plan["params"].get("none_tau_alpha", 0.0))
    none_tau_mode = str(plan["params"].get("none_tau_mode", "linear"))

    mod2idx = {id(m): i for i, (_, m) in enumerate(mods)}

    chosen = ["" for _ in mods]
    used_act = [False] * len(mods)

    def _use_act(i: int) -> bool:
        if act_every_n <= 0:
            return False
        return (i % act_every_n) == 0

    handles = []

    def pre_hook(module, inputs):
        i = mod2idx.get(id(module), None)
        if i is None:
            return
        xin = inputs[0]
        pos = 0.0 if len(mods) <= 1 else float(i) / float(len(mods) - 1)
        gamma_i = float(gamma_list[i])
        S = float(S_list[i])

        if _use_act(i):
            dl = _lora_delta_output(module, local_domain, xin)
            dg = _lora_delta_output(module, global_domain, xin)
            Sc = _activation_score(dl, act_score_mode, eps=eps)
            Ss = _activation_score(dg, act_score_mode, eps=eps)
            Ss_scaled = float(Ss * gamma_i * S)

            tau_none = _none_tau(pos, base=none_tau_base, alpha=none_tau_alpha, mode=none_tau_mode)
            best = max(float(Sc), float(Ss_scaled))
            if best < tau_none:
                pick = "none"
            else:
                pick = local_domain if Sc >= Ss_scaled else global_domain

            used_act[i] = True
        else:
            pick = picks_static[i]

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
        y = forward_padded(net, x, factor=factor)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    # stats
    n_local = sum(1 for p in chosen if p == local_domain)
    n_global = sum(1 for p in chosen if p == global_domain)
    n_none = sum(1 for p in chosen if p == "none")
    n_act = sum(1 for u in used_act if u)
    info = {
        "chosen": None,  # omit heavy per-layer list by default
        "n_layers": int(len(chosen)),
        "n_local": int(n_local),
        "n_global": int(n_global),
        "n_none": int(n_none),
        "local_ratio": float(n_local / max(1, len(chosen))),
        "global_ratio": float(n_global / max(1, len(chosen))),
        "none_ratio": float(n_none / max(1, len(chosen))),
        "act_layers": int(n_act),
        "act_ratio": float(n_act / max(1, len(chosen))),
    }
    return y, info


@MIXERS.register("kselect_static")
def mix_kselect_static(ctx: Mixer, x: torch.Tensor, decision: Decision, route: RouteResult, args: Dict[str, Any]):
    ctx.disable_lora()
    if decision.mode == "single" or not decision.local or not decision.global_:
        return mix_single(ctx, x, decision, route, args)

    k_topk = int(args.get("k_topk", 0))
    score_mode = str(args.get("k_score_mode", "topk_sum"))
    ramp_mode = str(args.get("k_ramp_mode", "sigmoid"))
    k_alpha = float(args.get("k_alpha", 1.0))
    k_beta = float(args.get("k_beta", 1.0))
    use_gamma = bool(int(args.get("use_gamma", 1)))
    gamma_mode = str(args.get("gamma_mode", "global"))
    gamma_clip = float(args.get("gamma_clip", 0.0))
    eps = float(args.get("score_eps", 1e-12))

    none_tau_base = float(args.get("none_tau", 0.0))
    none_tau_alpha = float(args.get("none_tau_alpha", 0.0))
    none_tau_mode = str(args.get("none_tau_mode", "linear"))

    # sensible default
    if k_topk <= 0:
        # rank*rank is a good proxy for conv/linear small-rank
        r = int(getattr(next(iter(iter_lora_named_modules(ctx.net)))[1], "rank", 16))
        k_topk = r * r

    plan = _build_kselect_plan(
        ctx.net,
        local_domain=decision.local,
        global_domain=decision.global_,
        k_topk=k_topk,
        score_mode=score_mode,
        ramp_mode=ramp_mode,
        k_alpha=k_alpha,
        k_beta=k_beta,
        use_gamma=use_gamma,
        gamma_mode=gamma_mode,
        gamma_clip=gamma_clip,
        eps=eps,
        none_tau_base=none_tau_base,
        none_tau_alpha=none_tau_alpha,
        none_tau_mode=none_tau_mode,
    )

    _apply_kselect_plan(ctx.net, plan, decision.local, decision.global_)
    y = forward_padded(ctx.net, x, factor=ctx.factor)

    picks = plan["picks"]
    n_local = sum(1 for p in picks if p == decision.local)
    n_global = sum(1 for p in picks if p == decision.global_)
    n_none = sum(1 for p in picks if p == "none")

    info = {
        "mix_mode": "kselect_static",
        "local": decision.local,
        "global": decision.global_,
        "kselect": {
            "n_layers": int(len(picks)),
            "n_local": int(n_local),
            "n_global": int(n_global),
            "n_none": int(n_none),
            "local_ratio": float(n_local / max(1, len(picks))),
            "global_ratio": float(n_global / max(1, len(picks))),
            "none_ratio": float(n_none / max(1, len(picks))),
            "gamma": float(plan.get("gamma_global", 1.0)),
            **plan["params"],
        },
    }
    return y, info


@MIXERS.register("kselect_activation")
def mix_kselect_activation(ctx: Mixer, x: torch.Tensor, decision: Decision, route: RouteResult, args: Dict[str, Any]):
    """Activation-based selection for every layer."""
    ctx.disable_lora()
    if decision.mode == "single" or not decision.local or not decision.global_:
        return mix_single(ctx, x, decision, route, args)

    # build a static plan (provides ramp+gamma+none thresholds); then override every layer by activation
    plan = _build_kselect_plan(
        ctx.net,
        local_domain=decision.local,
        global_domain=decision.global_,
        k_topk=int(args.get("k_topk", 0)) or 16 * 16,
        score_mode=str(args.get("k_score_mode", "topk_sum")),
        ramp_mode=str(args.get("k_ramp_mode", "sigmoid")),
        k_alpha=float(args.get("k_alpha", 1.0)),
        k_beta=float(args.get("k_beta", 1.0)),
        use_gamma=bool(int(args.get("use_gamma", 1))),
        gamma_mode=str(args.get("gamma_mode", "global")),
        gamma_clip=float(args.get("gamma_clip", 0.0)),
        eps=float(args.get("score_eps", 1e-12)),
        none_tau_base=float(args.get("none_tau", 0.0)),
        none_tau_alpha=float(args.get("none_tau_alpha", 0.0)),
        none_tau_mode=str(args.get("none_tau_mode", "linear")),
    )

    y, act_info = _forward_with_activation_kselect(
        ctx.net,
        x,
        plan=plan,
        local_domain=decision.local,
        global_domain=decision.global_,
        act_score_mode=str(args.get("act_score_mode", "mean_abs")),
        act_every_n=1,
        eps=float(args.get("score_eps", 1e-12)),
        factor=ctx.factor,
    )

    info = {
        "mix_mode": "kselect_activation",
        "local": decision.local,
        "global": decision.global_,
        "kselect": {
            **plan["params"],
            "gamma": float(plan.get("gamma_global", 1.0)),
            **act_info,
            "act_every_n": 1,
            "act_score_mode": str(args.get("act_score_mode", "mean_abs")),
        },
    }
    return y, info


@MIXERS.register("kselect_hybrid")
def mix_kselect_hybrid(ctx: Mixer, x: torch.Tensor, decision: Decision, route: RouteResult, args: Dict[str, Any]):
    """Activation decision every N layers + static fallback."""
    ctx.disable_lora()
    if decision.mode == "single" or not decision.local or not decision.global_:
        return mix_single(ctx, x, decision, route, args)

    plan = _build_kselect_plan(
        ctx.net,
        local_domain=decision.local,
        global_domain=decision.global_,
        k_topk=int(args.get("k_topk", 0)) or 16 * 16,
        score_mode=str(args.get("k_score_mode", "topk_sum")),
        ramp_mode=str(args.get("k_ramp_mode", "sigmoid")),
        k_alpha=float(args.get("k_alpha", 1.0)),
        k_beta=float(args.get("k_beta", 1.0)),
        use_gamma=bool(int(args.get("use_gamma", 1))),
        gamma_mode=str(args.get("gamma_mode", "global")),
        gamma_clip=float(args.get("gamma_clip", 0.0)),
        eps=float(args.get("score_eps", 1e-12)),
        none_tau_base=float(args.get("none_tau", 0.0)),
        none_tau_alpha=float(args.get("none_tau_alpha", 0.0)),
        none_tau_mode=str(args.get("none_tau_mode", "linear")),
    )

    act_every_n = int(args.get("act_every_n", 4))
    y, act_info = _forward_with_activation_kselect(
        ctx.net,
        x,
        plan=plan,
        local_domain=decision.local,
        global_domain=decision.global_,
        act_score_mode=str(args.get("act_score_mode", "mean_abs")),
        act_every_n=act_every_n,
        eps=float(args.get("score_eps", 1e-12)),
        factor=ctx.factor,
    )

    info = {
        "mix_mode": "kselect_hybrid",
        "local": decision.local,
        "global": decision.global_,
        "kselect": {
            **plan["params"],
            "gamma": float(plan.get("gamma_global", 1.0)),
            **act_info,
            "act_every_n": act_every_n,
            "act_score_mode": str(args.get("act_score_mode", "mean_abs")),
        },
    }
    return y, info


# -----------------------------------------------------------------------------
# DY-KSelect (activation-aware delta-output scoring + TopK/Tau + optional BOTH)
# -----------------------------------------------------------------------------


@torch.no_grad()
def _dy_collect_layer_scores(
    net,
    x: torch.Tensor,
    d1: str,
    d2: str,
    score_mode: str = "rms",
    eps: float = 1e-12,
    factor: int = 8,
):
    """Run a base-only forward to collect per-layer DY scores.

    We register forward_pre_hook on every LoRA module. For each module we capture:
      - rms_x : RMS of input activation x
      - dy1   : score(Δy1) where Δy1 = ΔW1 x
      - dy2   : score(Δy2) where Δy2 = ΔW2 x

    Notes
    -----
    - We DO NOT apply any LoRA during this pass (all modules keep empty weights).
    - score_mode uses _activation_score semantics: mean_abs / rms / l2.
      (We still log fields as dy1/dy2 for readability.)
    """

    mods = collect_lora_modules(net)
    if not mods:
        return {
            "mods": [],
            "hit": 0,
            "total": 0,
            "x_rms": [],
            "dy1": [],
            "dy2": [],
        }

    # Disable LoRA strictly (also safe if someone injected single-domain LoRA).
    for _, m in mods:
        m.set_domain_weights({})

    mod2idx = {id(m): i for i, (_, m) in enumerate(mods)}
    x_rms = [0.0] * len(mods)
    dy1 = [0.0] * len(mods)
    dy2 = [0.0] * len(mods)
    hit = 0

    handles = []

    def pre_hook(module, inputs):
        nonlocal hit
        i = mod2idx.get(id(module), None)
        if i is None:
            return
        try:
            xin = inputs[0]
            # Some modules may pass non-tensor (rare). Just skip.
            if not torch.is_tensor(xin):
                return
            x_rms[i] = _rms_tensor(xin, eps=eps)

            dl = _lora_delta_output(module, d1, xin)
            dg = _lora_delta_output(module, d2, xin)

            dy1[i] = float(_activation_score(dl, score_mode, eps=eps))
            dy2[i] = float(_activation_score(dg, score_mode, eps=eps))
            hit += 1
        except Exception:
            # keep zeros
            return

    for _, m in mods:
        handles.append(m.register_forward_pre_hook(pre_hook))

    try:
        _ = forward_padded(net, x, factor=factor)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    return {
        "mods": mods,
        "hit": int(hit),
        "total": int(len(mods)),
        "x_rms": x_rms,
        "dy1": dy1,
        "dy2": dy2,
    }


@torch.no_grad()
def _dy_build_and_apply_plan(
    net,
    stats: Dict[str, Any],
    d1: str,
    d2: str,
    topk_layers: int,
    tau: float,
    enable_both: bool,
    both_tau: float,
    both_ratio: float,
    eps: float = 1e-12,
    verbose: int = 1,
    debug_topn: int = 30,
    store_ranked: bool = False,
    factor: int = 8,
    x: Optional[torch.Tensor] = None,
):
    """Select layers by DY score and apply per-layer domain weights.

    Returns:
      y (if x is provided) and a rich info dict for logging.
    """

    mods: List[Tuple[str, Any]] = stats.get("mods", [])
    if not mods:
        # no LoRA modules
        if x is not None:
            y = forward_padded(net, x, factor=factor)
        else:
            y = None
        info = {
            "total": 0,
            "selected": 0,
            "pick0": 0,
            "pick1": 0,
            "pick2": 0,
            "both": 0,
            "dw_missing": 0,
        }
        return y, info

    dy1 = list(stats.get("dy1", []))
    dy2 = list(stats.get("dy2", []))
    x_rms = list(stats.get("x_rms", []))

    total = len(mods)
    # Build per-layer records
    recs = []
    for i, ((name, m), a, b, rx) in enumerate(zip(mods, dy1, dy2, x_rms)):
        a = float(a)
        b = float(b)
        s = float(max(a, b))
        pick = 1 if a >= b else 2
        mn = float(min(a, b))
        mx = float(max(a, b))
        ratio = float(mn / (mx + eps)) if mx > 0 else 0.0
        recs.append(
            {
                "idx": int(i),
                "name": str(name),
                "score": s,
                "rms_x": float(rx),
                "dy1": a,
                "dy2": b,
                "pick": int(pick),
                "ratio": ratio,
                # debug weight norms (not used for selection)
                "n1": float(_deltaW_fro_norm(m, d1, eps=eps)),
                "n2": float(_deltaW_fro_norm(m, d2, eps=eps)),
            }
        )

    recs_sorted = sorted(recs, key=lambda r: r["score"], reverse=True)

    # select layers: TopK then tau filter
    if topk_layers <= 0 or topk_layers > total:
        topk_layers = total
    topk_set = set(r["idx"] for r in recs_sorted[:topk_layers])
    selected = [r for r in recs_sorted if (r["idx"] in topk_set and r["score"] >= tau)]
    selected_set = set(r["idx"] for r in selected)

    pick0 = pick1 = pick2 = both = 0
    dw_missing = 0

    # Apply weights
    for i, (_, m) in enumerate(mods):
        if i not in selected_set:
            m.set_domain_weights({})
            pick0 += 1
            continue

        r = recs[i]
        s = float(r["score"])
        a = float(r["dy1"])
        b = float(r["dy2"])
        ratio = float(r["ratio"])

        # BOTH condition (AND): score >= both_tau AND ratio >= both_ratio
        if enable_both and (s >= both_tau) and (ratio >= both_ratio):
            denom = a + b + eps
            w1 = float(a / denom)
            w2 = float(b / denom)
            m.set_domain_weights({d1: w1, d2: w2})
            both += 1
            r["mode"] = "both"
            r["w1"] = w1
            r["w2"] = w2
        else:
            if int(r["pick"]) == 1:
                m.set_domain_weights({d1: 1.0, d2: 0.0})
                pick1 += 1
                r["mode"] = "pick"
                r["w1"] = 1.0
                r["w2"] = 0.0
            else:
                m.set_domain_weights({d1: 0.0, d2: 1.0})
                pick2 += 1
                r["mode"] = "pick"
                r["w1"] = 0.0
                r["w2"] = 1.0

    selected_n = len(selected)

    # Print debug logs
    if verbose:
        # global stats
        x_rms_arr = np.asarray(x_rms, dtype=np.float32) if x_rms else np.asarray([0.0], dtype=np.float32)
        dy_rms_arr = np.asarray([max(float(a), float(b)) for a, b in zip(dy1, dy2)], dtype=np.float32) if dy1 else np.asarray([0.0], dtype=np.float32)
        print(
            f"[act_kselect_dy] total={total} topk={topk_layers} tau={tau} selected={selected_n} "
            f"pick0={pick0} pick1={pick1} pick2={pick2} both={both} dw_missing={dw_missing} "
            f"(both_tau={both_tau} both_ratio={both_ratio} enable_both={int(enable_both)})"
        )
        print(
            f"[act_kselect_dy][hook] hit={int(stats.get('hit', 0))}/{total} "
            f"x_rms(min/max/mean)={float(x_rms_arr.min()):.6g}/{float(x_rms_arr.max()):.6g}/{float(x_rms_arr.mean()):.6g} "
            f"dy_rms(min/max/mean)={float(dy_rms_arr.min()):.6g}/{float(dy_rms_arr.max()):.6g}/{float(dy_rms_arr.mean()):.6g}"
        )

        # per-layer ranked list (top debug_topn)
        topn = int(debug_topn) if debug_topn and debug_topn > 0 else 0
        if topn > 0:
            print(f"[act_kselect_dy] scored modules (sorted):")
            for j, r in enumerate(recs_sorted[:topn]):
                i = int(r["idx"])
                mode = "none" if i not in selected_set else r.get("mode", "pick")
                pick = int(r["pick"]) if i in selected_set else 0
                print(
                    f"{j:02d} {r['name']} score={r['score']:.6g} rms_x={r['rms_x']:.6g} "
                    f"dy1={r['dy1']:.6g} dy2={r['dy2']:.6g} pick={pick} mode={mode} "
                    f"n1={r['n1']:.6g} n2={r['n2']:.6g}"
                )

    # forward with applied weights
    y = None
    if x is not None:
        y = forward_padded(net, x, factor=factor)

    info = {
        "total": int(total),
        "topk_layers": int(topk_layers),
        "tau": float(tau),
        "selected": int(selected_n),
        "pick0": int(pick0),
        "pick1": int(pick1),
        "pick2": int(pick2),
        "both": int(both),
        "dw_missing": int(dw_missing),
        "both_tau": float(both_tau),
        "both_ratio": float(both_ratio),
        "enable_both": int(enable_both),
        "hook_hit": int(stats.get("hit", 0)),
        "hook_total": int(total),
        "score_mode": str(stats.get("score_mode", "")),
    }
    if store_ranked:
        # store only top debug_topn records to keep jsonl smaller
        topn = int(debug_topn) if debug_topn and debug_topn > 0 else min(30, total)
        info["ranked_top"] = recs_sorted[:topn]
        info["selected_ids"] = sorted(list(selected_set))

    return y, info


@MIXERS.register("act_kselect_dy")
def mix_act_kselect_dy(ctx: Mixer, x: torch.Tensor, decision: Decision, route: RouteResult, args: Dict[str, Any]):
    """DY-KSelect: two-LoRA mixing with per-layer DY scoring.

    This is the strategy you described:
      1) base-only forward + hooks to compute dy1/dy2 per LoRA module
      2) select TopK layers by score=max(dy1,dy2), then apply tau
      3) for selected layers: BOTH if score>=both_tau AND ratio>=both_ratio; else hard pick
      4) unselected layers keep none

    Required: decision.local and decision.global_ must be set.
    """
    ctx.disable_lora()
    if decision.mode == "single" or not decision.local or not decision.global_:
        return mix_single(ctx, x, decision, route, args)

    # params
    topk_layers = int(args.get("dy_topk_layers", 30))
    tau = float(args.get("dy_tau", 0.0))
    enable_both = bool(int(args.get("dy_enable_both", 1)))
    both_tau = float(args.get("dy_both_tau", 1.0))
    both_ratio = float(args.get("dy_both_ratio", 0.6))
    score_mode = str(args.get("dy_score_mode", "rms"))
    eps = float(args.get("score_eps", 1e-12))
    verbose = int(args.get("dy_verbose", 1))
    debug_topn = int(args.get("dy_debug_topn", 30))
    store_ranked = bool(int(args.get("dy_store_ranked", 0)))

    # 1) collect layer scores
    stats = _dy_collect_layer_scores(
        ctx.net,
        x,
        d1=decision.local,
        d2=decision.global_,
        score_mode=score_mode,
        eps=eps,
        factor=ctx.factor,
    )
    stats["score_mode"] = score_mode

    # 2) build plan, apply, and forward
    y, dy_info = _dy_build_and_apply_plan(
        ctx.net,
        stats,
        d1=decision.local,
        d2=decision.global_,
        topk_layers=topk_layers,
        tau=tau,
        enable_both=enable_both,
        both_tau=both_tau,
        both_ratio=both_ratio,
        eps=eps,
        verbose=verbose,
        debug_topn=debug_topn,
        store_ranked=store_ranked,
        factor=ctx.factor,
        x=x,
    )

    info = {
        "mix_mode": "act_kselect_dy",
        "local": decision.local,
        "global": decision.global_,
        "dy": dy_info,
    }
    return y, info


# Optional alias (shorter)
@MIXERS.register("kselect_dy")
def mix_kselect_dy_alias(ctx: Mixer, x: torch.Tensor, decision: Decision, route: RouteResult, args: Dict[str, Any]):
    return mix_act_kselect_dy(ctx, x, decision, route, args)

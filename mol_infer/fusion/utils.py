# mol_infer/fusion/utils.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mol_infer.lora.modules import (
    iter_lora_named_modules,
    get_up_down,
    get_scale,
)


# ---------------------------
# small helpers
# ---------------------------
def to_float(x) -> float:
    try:
        if isinstance(x, (float, int)):
            return float(x)
        if isinstance(x, np.ndarray):
            return float(x.reshape(-1)[0])
        if isinstance(x, torch.Tensor):
            return float(x.detach().reshape(-1)[0].cpu().item())
        if isinstance(x, (list, tuple)) and len(x) > 0:
            return to_float(x[0])
    except Exception:
        pass
    return float(x)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def ramp_value(pos01: float, mode: str, alpha: float, beta: float) -> float:
    """
    Used to scale global score, typical:
      Ss' = Ss * gamma * S(pos)

    mode:
      - linear:  S = beta + alpha * pos
      - sigmoid: S = beta + alpha * sigmoid(10*(pos-0.5))
    """
    mode = str(mode).lower()
    if mode == "linear":
        return float(beta + alpha * pos01)
    if mode == "sigmoid":
        return float(beta + alpha * sigmoid(10.0 * (pos01 - 0.5)))
    raise ValueError(f"Unknown ramp mode: {mode}")


def none_threshold_value(pos01: float, mode: str, alpha: float, beta: float) -> float:
    """
    Threshold schedule for 'none/base' in 3-way KSelect:
      compare Sc vs Ss' vs T_none(pos).

    mode:
      - const:    T = beta
      - linear:   T = beta + alpha * pos
      - sigmoid:  T = beta + alpha * sigmoid(10*(pos-0.5))
    """
    mode = str(mode).lower()
    if mode == "const":
        return float(beta)
    if mode == "linear":
        return float(beta + alpha * pos01)
    if mode == "sigmoid":
        return float(beta + alpha * sigmoid(10.0 * (pos01 - 0.5)))
    raise ValueError(f"Unknown none_mode: {mode}")


# ---------------------------
# weight-space scoring
# ---------------------------
@torch.no_grad()
def layer_param_abs_sum(m: Any, domain: str) -> float:
    """
    Very safe proxy: |up|_1 + |down|_1 (domain-specific)
    """
    up, down = get_up_down(m, domain)
    return float(up.weight.abs().sum().item() + down.weight.abs().sum().item())


@torch.no_grad()
def delta_abs_matrix_proxy(m: Any, domain: str) -> Optional[torch.Tensor]:
    """
    Build a 2D non-negative proxy of |ΔW| for this (module, domain).

    For LoRA: ΔW ≈ (up.weight @ down.weight) (after flattening).
    We approximate |ΔW| by: |B| @ |A|.

    Returns:
      delta_abs: [out, in_flat]  (torch.Tensor)  or None if incompatible.
    """
    up, down = get_up_down(m, domain)
    A = down.weight
    B = up.weight

    # A: [r, in] or [r, in, kH, kW] -> [r, in_flat]
    A2 = A.reshape(A.shape[0], -1).float().abs()

    # B: [out, r] or [out, r, kH, kW] -> [out, r_flat]
    Bf = B.reshape(B.shape[0], -1).float().abs()

    # align r dimension if B has extra kernel dims
    if Bf.shape[1] != A2.shape[0]:
        r = A2.shape[0]
        if Bf.numel() % r == 0:
            try:
                # [out, r, *] -> sum over * => [out, r]
                Bf = Bf.reshape(B.shape[0], r, -1).sum(dim=-1)
            except Exception:
                return None
        else:
            return None

    try:
        return torch.matmul(Bf, A2)  # [out, in_flat]
    except Exception:
        return None


@torch.no_grad()
def weight_score(
    m: Any,
    domain: str,
    k_topk: int,
    score_mode: str,
    eps: float = 1e-12,
) -> float:
    """
    score_mode:
      - topk_sum:    sum(topK(|ΔW|)) (legacy)
      - mean:        mean(|ΔW|)
      - median:      median(|ΔW|)
      - fro:         Frobenius norm of |ΔW|
      - topk_ratio:  sum(topK)/sum(all)   ("尖锐度/稀疏度" proxy)
    """
    score_mode = str(score_mode).lower()
    delta = delta_abs_matrix_proxy(m, domain)
    if delta is None:
        return layer_param_abs_sum(m, domain)

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


# ---------------------------
# gamma (scale balancing)
# ---------------------------
@torch.no_grad()
def compute_gamma_global_paramsum(net: Any, local_domain: str, global_domain: str, eps: float = 1e-12) -> float:
    """
    Global gamma computed from Σ(|up|_1+|down|_1).
    gamma > 1 means local params larger -> scale global up.
    """
    sum_local, sum_global = 0.0, 0.0
    for _, m in iter_lora_named_modules(net):
        sum_local += layer_param_abs_sum(m, local_domain)
        sum_global += layer_param_abs_sum(m, global_domain)
    if sum_global <= eps:
        return 1.0
    return float(sum_local / (sum_global + eps))


@torch.no_grad()
def compute_gamma_list_paramsum(
    mods: List[Tuple[str, Any]],
    local_domain: str,
    global_domain: str,
    eps: float = 1e-12,
    gamma_clip: float = 0.0,
) -> List[float]:
    out: List[float] = []
    for _, m in mods:
        gl = layer_param_abs_sum(m, local_domain)
        gg = layer_param_abs_sum(m, global_domain)
        g = float(gl / (gg + eps)) if gg > eps else 1.0
        if gamma_clip and gamma_clip > 0:
            lo = 1.0 / float(gamma_clip)
            hi = float(gamma_clip)
            g = float(min(hi, max(lo, g)))
        out.append(g)
    return out


# ---------------------------
# activation-space helpers
# ---------------------------
def activation_score(t: torch.Tensor, mode: str = "mean_abs", eps: float = 1e-12) -> float:
    if t is None:
        return 0.0
    mode = str(mode).lower()
    if mode == "l2":
        return float(torch.linalg.norm(t.reshape(-1).float(), ord=2).item())
    if mode == "rms":
        return float(torch.sqrt((t.float() ** 2).mean() + eps).item())
    return float(t.float().abs().mean().item())


@torch.no_grad()
def lora_delta_output(m: Any, domain: str, x: torch.Tensor) -> torch.Tensor:
    """
    Compute Δy = LoRA_domain(x) (WITHOUT base), best-effort.
    Works for both LoRALinear/LoRAConv2d because up/down are Modules.
    """
    up, down = get_up_down(m, domain)
    y = up(down(x))
    scale = get_scale(m, domain)
    if abs(scale - 1.0) > 1e-12:
        y = y * float(scale)
    return y


def pad_to_factor(x: torch.Tensor, factor: int = 8) -> Tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h != 0 or pad_w != 0:
        x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    else:
        x_in = x
    return x_in, h, w

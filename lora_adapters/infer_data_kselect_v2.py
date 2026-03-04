# lora_adapters/infer_data_kselect_v2.py
# -*- coding: utf-8 -*-
"""
KSelect v2 inference (Histoformer + multi-domain LoRA + routing + layerwise selection)

Changes vs infer_data_kselect.py:
  1) Static KSelect weight-score becomes configurable:
       - topk_sum (legacy)
       - mean / median / fro / topk_ratio
     + optional per-layer gamma (scale balancing)

  2) Input-adaptive KSelect (activation-domain):
       For each LoRA module (or every N-th module),
         compute Δy_local and Δy_global on current activation x_l,
         score by mean_abs / l2 / rms,
         compare with ramp+gamma scaling, then hard-pick local/global.

This file is designed to fit YOUR repo imports:
  - build_histoformer / inject_lora
  - CLIPEmbedder / DomainOrchestrator
  - tensor_psnr / tensor_ssim / get_embedder_tag / load_all_domain_loras / set_all_lora_domain_weights
"""

import os
import csv
import json
import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from lora_adapters.utils import build_histoformer
from lora_adapters.inject_lora import inject_lora
from lora_adapters.vis_utils import save_triplet

# ---- embedder ----
try:
    from lora_adapters.embedding_clip import CLIPEmbedder
except Exception as e:
    raise ImportError(f"Cannot import CLIPEmbedder from lora_adapters.embedding_clip: {e}")

# ---- orchestrator ----
from lora_adapters.domain_orchestrator import DomainOrchestrator

# ---- reuse utilities ----
try:
    from lora_adapters.infer_data import (
        tensor_psnr,
        tensor_ssim,
        get_embedder_tag,
        load_all_domain_loras,
        set_all_lora_domain_weights,
    )
except Exception as e:
    raise ImportError(f"Cannot import required utilities from lora_adapters.infer_data: {e}")

from lora_adapters.common.seed import set_seed

try:
    from lora_adapters.lora_linear import LoRALinear, LoRAConv2d
except Exception as e:
    raise ImportError(f"Cannot import LoRA modules (LoRALinear/LoRAConv2d): {e}")


# ---------------------------
# IO helpers
# ---------------------------
def parse_pair_list(path: str) -> List[Tuple[str, Optional[str]]]:
    pairs: List[Tuple[str, Optional[str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
            else:
                pairs.append((parts[0], None))
    return pairs


def list_images(folder: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
    out = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(exts):
            out.append(os.path.join(folder, fn))
    return out


# ---------------------------
# forward helper (pad to 8x)
# ---------------------------
def forward_padded(net, x: torch.Tensor, factor: int = 8) -> torch.Tensor:
    _, _, h, w = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h != 0 or pad_w != 0:
        x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    else:
        x_in = x
    y = net(x_in)[:, :, :h, :w]
    return y


# ---------------------------
# LoRA module helpers
# ---------------------------
def iter_lora_named_modules(net):
    """Yield (name, module) for LoRA modules in stable order."""
    for name, m in net.named_modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            yield name, m


def collect_lora_modules(net) -> List[Tuple[str, Any]]:
    return list(iter_lora_named_modules(net))


def _get_domain_up_down(m, domain: str):
    if not hasattr(m, "lora_up") or not hasattr(m, "lora_down"):
        raise RuntimeError("LoRA module does not have lora_up/lora_down dicts.")
    if domain not in m.lora_up or domain not in m.lora_down:
        raise KeyError(f"domain={domain} not found in this LoRA module.")
    return m.lora_up[domain], m.lora_down[domain]


def _get_lora_scale(m) -> float:
    # your LoRA modules store alpha & rank
    alpha = getattr(m, "alpha", 1.0)
    r = getattr(m, "rank", getattr(m, "r", 0))
    r = int(r) if r is not None else 0
    if r <= 0:
        return 1.0
    return float(alpha) / float(r)


# ---------------------------
# decision: single or mix
# ---------------------------
def decide_single_or_mix(
    picks,  # [(domain, weight)] already sorted desc
    single_tau: float,
    single_margin: float,
    mix_topk: int,
    local_domains,
    global_domains,
):
    top1_d, top1_w = picks[0][0], float(picks[0][1])
    top2_w = float(picks[1][1]) if len(picks) >= 2 else 0.0

    if top1_w >= single_tau:
        return {
            "mode": "single", "top1": top1_d, "top1_w": top1_w, "top2_w": top2_w,
            "local": None, "global": None,
            "reason": f"top1_w>=tau({top1_w:.4f}>={single_tau})",
        }
    if (top1_w - top2_w) >= single_margin:
        return {
            "mode": "single", "top1": top1_d, "top1_w": top1_w, "top2_w": top2_w,
            "local": None, "global": None,
            "reason": f"top1-top2>=margin({top1_w-top2_w:.4f}>={single_margin})",
        }

    cand = picks[:max(1, mix_topk)]
    best_local = None
    best_global = None
    for d, w in cand:
        w = float(w)
        if (d in local_domains) and (best_local is None):
            best_local = (d, w)
        if (d in global_domains) and (best_global is None):
            best_global = (d, w)
        if best_local is not None and best_global is not None:
            break

    if best_local is None or best_global is None:
        return {
            "mode": "single",
            "top1": top1_d, "top1_w": top1_w, "top2_w": top2_w,
            "local": None, "global": None,
            "reason": "mix not possible (no local/global in cand) -> fallback single",
        }

    local, local_w = best_local
    global_, global_w = best_global
    if local == global_:
        return {
            "mode": "single",
            "top1": top1_d, "top1_w": top1_w, "top2_w": top2_w,
            "local": None, "global": None,
            "reason": "mix not possible (local==global) -> fallback single",
        }

    return {
        "mode": "mix",
        "top1": top1_d, "top1_w": top1_w, "top2_w": top2_w,
        "local": local,
        "global": global_,
        "local_w": local_w,
        "global_w": global_w,
        "reason": f"mix(best_local={local}:{local_w:.4f}, best_global={global_}:{global_w:.4f})",
    }


# ---------------------------
# KSelect v2 helpers
# ---------------------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _ramp_value(pos01: float, mode: str, alpha: float, beta: float) -> float:
    pos01 = float(max(0.0, min(1.0, pos01)))
    if mode == "linear":
        return float(alpha * pos01 + beta)
    if mode == "sigmoid":
        return float(alpha * _sigmoid(6.0 * (pos01 - 0.5)) + beta)
    return float(alpha * pos01 + beta)

def _none_tau(pos01: float, base: float, alpha: float, mode: str) -> float:
    """
    Threshold for enabling LoRA at this layer.
    tau_none(pos) = base + alpha * f(pos)
    """
    pos01 = float(max(0.0, min(1.0, pos01)))
    if mode == "linear":
        return float(base + alpha * pos01)
    if mode == "sigmoid":
        return float(base + alpha * _sigmoid(6.0 * (pos01 - 0.5)))
    return float(base + alpha * pos01)


@torch.no_grad()
def _layer_param_abs_sum(m, domain: str) -> float:
    up, down = _get_domain_up_down(m, domain)
    return float(up.weight.abs().sum().item() + down.weight.abs().sum().item())


@torch.no_grad()
def _delta_abs_matrix_proxy(m, domain: str) -> Optional[torch.Tensor]:
    """
    Build a 2D non-negative proxy of |ΔW|:
      delta_abs ≈ |B| @ |A|
    where:
      A = down.weight, B = up.weight (flattened)
    For conv-up (out, r, k, k): collapse kernel dims into r by summing.
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
                Bf = Bf.reshape(B.shape[0], r, -1).sum(dim=-1)  # [out, r]
            except Exception:
                return None
        else:
            return None

    try:
        return torch.matmul(Bf, A2)  # [out, in_flat]
    except Exception:
        return None


@torch.no_grad()
def _weight_score(
    m,
    domain: str,
    k_topk: int,
    score_mode: str,
    eps: float = 1e-12,
) -> float:
    """
    Static (weight-space) scoring:
      - topk_sum: sum of top-K(|ΔW|)
      - mean: mean(|ΔW|)
      - median: median(|ΔW|)
      - fro: Frobenius norm of |ΔW|
      - topk_ratio: sum_topk(|ΔW|) / (sum(|ΔW|)+eps)
    """
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
    sum_local, sum_global = 0.0, 0.0
    for _, m in iter_lora_named_modules(net):
        sum_local += _layer_param_abs_sum(m, local_domain)
        sum_global += _layer_param_abs_sum(m, global_domain)
    if sum_global <= eps:
        return 1.0
    return float(sum_local / (sum_global + eps))


@torch.no_grad()
def _compute_gamma_list_per_layer(
    mods: List[Tuple[str, Any]],
    local_domain: str,
    global_domain: str,
    eps: float = 1e-12,
    gamma_clip: float = 0.0,
) -> List[float]:
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


@torch.no_grad()
def build_kselect_static_plan(
    net,
    local_domain: str,
    global_domain: str,
    k_topk: int,
    alpha: float,
    beta: float,
    ramp_mode: str,
    score_mode: str,
    use_gamma: bool,
    gamma_mode: str,
    gamma_clip: float = 0.0,
    eps: float = 1e-12,
    verbose: bool = False,
) -> Dict[str, Any]:
    mods = list(iter_lora_named_modules(net))
    if len(mods) == 0:
        raise RuntimeError("[kselect] found 0 LoRA modules. Injection failed?")

    if (not use_gamma) or (gamma_mode == "none"):
        gamma_global = 1.0
        gamma_list = [1.0 for _ in mods]
    else:
        gamma_global = _compute_gamma_global(net, local_domain, global_domain, eps=eps)
        if gamma_mode == "per_layer":
            gamma_list = _compute_gamma_list_per_layer(
                mods, local_domain, global_domain, eps=eps, gamma_clip=gamma_clip
            )
        else:
            gamma_list = [gamma_global for _ in mods]

    picks: List[str] = []
    sc_list: List[float] = []
    ss_list: List[float] = []
    ss_scaled_list: List[float] = []
    s_list: List[float] = []
    names: List[str] = []

    for i, (name, m) in enumerate(mods):
        pos = 0.0 if len(mods) == 1 else float(i) / float(len(mods) - 1)
        S = _ramp_value(pos, ramp_mode, alpha=alpha, beta=beta)
        gamma_i = gamma_list[i]

        Sc = _weight_score(m, local_domain, k_topk=k_topk, score_mode=score_mode, eps=eps)
        Ss = _weight_score(m, global_domain, k_topk=k_topk, score_mode=score_mode, eps=eps)
        Ss_scaled = float(Ss * gamma_i * S)

        pick = local_domain if Sc >= Ss_scaled else global_domain

        picks.append(pick)
        names.append(name)
        sc_list.append(float(Sc))
        ss_list.append(float(Ss))
        ss_scaled_list.append(float(Ss_scaled))
        s_list.append(float(S))

        if verbose and (i < 5 or i >= len(mods) - 2):
            print(f"[kselect-static][layer {i:03d}] pos={pos:.3f} S={S:.4f} gamma={gamma_i:.4f} "
                  f"Sc={Sc:.3e} Ss={Ss:.3e} Ss'={Ss_scaled:.3e} -> pick={pick} | {name}")

    n_local = sum(1 for p in picks if p == local_domain)
    n_global = len(picks) - n_local

    return {
        "_mods": mods,
        "names": names,
        "picks": picks,
        "gamma_global": float(gamma_global),
        "gamma_mode": str(gamma_mode),
        "gamma_list": gamma_list,
        "score_mode": str(score_mode),
        "k_topk": int(k_topk),
        "ramp_mode": str(ramp_mode),
        "alpha": float(alpha),
        "beta": float(beta),
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


@torch.no_grad()
def apply_kselect_plan(plan: Dict[str, Any], local_domain: str, global_domain: str):
    mods = plan["_mods"]
    picks = plan["picks"]
    for (_, m), pick in zip(mods, picks):
        if pick == local_domain:
            m.set_domain_weights({local_domain: 1.0, global_domain: 0.0})
        else:
            m.set_domain_weights({local_domain: 0.0, global_domain: 1.0})


def _activation_score(t: torch.Tensor, mode: str = "mean_abs", eps: float = 1e-12) -> float:
    if t is None:
        return 0.0
    if mode == "l2":
        return float(torch.linalg.norm(t.reshape(-1).float(), ord=2).item())
    if mode == "rms":
        return float(torch.sqrt((t.float() ** 2).mean() + eps).item())
    return float(t.float().abs().mean().item())


@torch.no_grad()
def _lora_delta_output(m, domain: str, x: torch.Tensor) -> torch.Tensor:
    up, down = _get_domain_up_down(m, domain)
    y = up(down(x))
    scale = _get_lora_scale(m)
    if abs(scale - 1.0) > 1e-12:
        y = y * scale
    return y


@torch.no_grad()
def forward_padded_with_activation_kselect(
    net,
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
    mods = plan["_mods"]
    picks_static = plan["picks"]
    gamma_list = plan.get("gamma_list", [1.0 for _ in mods])
    S_list = plan.get("S_list", [1.0 for _ in mods])

    _, _, h, w = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else x

    mod2idx = {id(m): i for i, (_, m) in enumerate(mods)}
    chosen = [None for _ in mods]
    sc_list = [0.0 for _ in mods]
    ss_list = [0.0 for _ in mods]
    ss_scaled_list = [0.0 for _ in mods]
    used_act = [False for _ in mods]

    def _should_act(i: int) -> bool:
        if act_every_n <= 0:
            return False
        return (i % act_every_n) == 0

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
                name = plan.get("names", [""] * len(mods))[i]
                print(f"[kselect-act][layer {i:03d}] S={S:.4f} gamma={gamma_i:.4f} "
                      f"Sc_act={Sc:.3e} Ss_act={Ss:.3e} Ss'={Ss_scaled:.3e} -> pick={pick} | {name}")
        else:
            pick = picks_static[i]
            sc_list[i] = float(plan.get("Sc_list", [0.0] * len(mods))[i])
            ss_list[i] = float(plan.get("Ss_list", [0.0] * len(mods))[i])
            ss_scaled_list[i] = float(plan.get("Ss_scaled_list", [0.0] * len(mods))[i])

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

    kstats = {
        "kselect_mode": f"activation(every_n={act_every_n})",
        "score_mode": f"act:{act_score_mode} + static:{plan.get('score_mode','')}",
        "gamma_mode": plan.get("gamma_mode", ""),
        "gamma": float(plan.get("gamma_global", 1.0)),
        "n_layers": int(len(chosen)),
        "n_local": int(n_local),
        "n_global": int(n_global),
        "local_ratio": float(n_local / max(1, len(chosen))),
        "global_ratio": float(n_global / max(1, len(chosen))),
        "act_layers": int(act_cnt),
        "act_ratio": float(act_cnt / max(1, len(chosen))),
        "sc_mean": float(np.mean(sc_list)) if sc_list else 0.0,
        "ss_mean": float(np.mean(ss_list)) if ss_list else 0.0,
        "ss_scaled_mean": float(np.mean(ss_scaled_list)) if ss_scaled_list else 0.0,
        "S_first": float(plan.get("S_first", 0.0)),
        "S_last": float(plan.get("S_last", 0.0)),
    }
    return y, kstats


# ---------------------------
# main
# ---------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser("infer_data_kselect_v2")

    # io
    ap.add_argument("--input", type=str, default=None, help="input folder (if no --pair_list)")
    ap.add_argument("--pair_list", type=str, default=None, help="txt: lq [gt]")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--gt_root", type=str, default=None)

    # model/lora
    ap.add_argument("--base_ckpt", type=str, required=True)
    ap.add_argument("--yaml", type=str, default=None)
    ap.add_argument("--loradb", type=str, required=True)
    ap.add_argument("--domains", type=str, required=True)
    ap.add_argument("--local_domains", type=str, required=True)
    ap.add_argument("--global_domains", type=str, required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)

    # retrieval
    ap.add_argument("--embedder", type=str, default="clip", choices=["clip"])
    ap.add_argument("--clip_model", type=str, default="ViT-B-16")
    ap.add_argument("--clip_pretrained", type=str, default="openai")
    ap.add_argument("--embedder_tag", type=str, default=None)
    ap.add_argument("--sim_metric", type=str, default="euclidean", choices=["euclidean", "cosine", "l2"])
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--mix_topk", type=int, default=5)

    # decision
    ap.add_argument("--single_tau", type=float, default=0.72)
    ap.add_argument("--single_margin", type=float, default=0.10)

    # kselect v2
    ap.add_argument("--kselect_mode", type=str, default="static",
                    choices=["static", "activation", "hybrid"],
                    help="static=weight-only plan; activation=input-adaptive every layer; hybrid=activation every N layers + static fallback")
    ap.add_argument("--k_topk", type=int, default=0, help="Top-K elements in |ΔW| scoring. 0=>use rank*rank")
    ap.add_argument("--k_score_mode", type=str, default="topk_sum",
                    choices=["topk_sum", "mean", "median", "fro", "topk_ratio"])
    ap.add_argument("--k_alpha", type=float, default=1.0)
    ap.add_argument("--k_beta", type=float, default=1.0)
    ap.add_argument("--k_ramp_mode", type=str, default="sigmoid", choices=["linear", "sigmoid"])

    ap.add_argument("--use_gamma", type=int, default=1)
    ap.add_argument("--gamma_mode", type=str, default="global", choices=["none", "global", "per_layer"])
    ap.add_argument("--gamma_clip", type=float, default=0.0, help="if >0, clip per-layer gamma to [1/gamma_clip, gamma_clip]")
    ap.add_argument("--score_eps", type=float, default=1e-12)
    ap.add_argument("--none_tau", type=float, default=0.0,
                    help="Base threshold for enabling ANY LoRA at a layer. If best_score < none_tau(pos), pick none.")
    ap.add_argument("--none_tau_alpha", type=float, default=0.0,
                    help="none threshold ramp alpha: none_tau(pos)=none_tau + none_tau_alpha*pos (pos in [0,1])")
    ap.add_argument("--none_tau_mode", type=str, default="linear", choices=["linear", "sigmoid"],
                    help="ramp mode for none threshold (same shape as k_ramp)")

    # activation kselect
    ap.add_argument("--act_score_mode", type=str, default="mean_abs", choices=["mean_abs", "l2", "rms"])
    ap.add_argument("--act_every_n", type=int, default=1, help="activation decision every N-th LoRA module (hybrid). N=1 means every module")

    # saving
    ap.add_argument("--save_images", action="store_true")
    ap.add_argument("--concat", action="store_true", help="save triplet concat (LQ|BASE|AUTO) into output_dir/triplets")
    ap.add_argument("--save_singles", action="store_true", help="also save base/auto single images into output_dir/images")
    ap.add_argument("--save_lq", action="store_true", help="if save_singles, also save lq")

    # logs
    ap.add_argument("--metrics_csv", type=str, default=None)
    ap.add_argument("--routing_jsonl", type=str, default=None)
    ap.add_argument("--summary_csv", type=str, default=None)
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--print_full_scores", action="store_true")

    # runtime
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_seed(args.seed, deterministic=args.deterministic)

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    args.embedder_tag = args.embedder_tag or get_embedder_tag(args)

    print(f"[INFO] device={device}")
    print(f"[INFO] run_name={args.run_name}")
    print(f"[INFO] embedder={args.embedder}, embedder_tag={args.embedder_tag}")
    print(f"[INFO] sim_metric={args.sim_metric}, temperature={args.temperature}")
    print(f"[INFO] kselect_mode={args.kselect_mode} score_mode={args.k_score_mode} gamma_mode={args.gamma_mode}")

    if args.pair_list:
        pairs = parse_pair_list(args.pair_list)
    else:
        assert args.input is not None, "Need --pair_list or --input"
        imgs = list_images(args.input) if os.path.isdir(args.input) else [args.input]
        pairs = [(p, None) for p in imgs]

    doms_all = [d.strip() for d in args.domains.split(",") if d.strip()]
    local_domains = [d.strip() for d in args.local_domains.split(",") if d.strip()]
    global_domains = [d.strip() for d in args.global_domains.split(",") if d.strip()]

    print(f"[INFO] total pairs={len(pairs)}")
    print(f"[INFO] domains={doms_all}")
    print(f"[INFO] local_domains={local_domains}")
    print(f"[INFO] global_domains={global_domains}")

    # build base model + inject LoRA slots
    print("[INFO] building base Histoformer...")
    net = build_histoformer(weights=args.base_ckpt, yaml_file=args.yaml).to(device)
    net = inject_lora(
        net,
        rank=args.rank,
        domain_list=doms_all,
        alpha=args.alpha,
        target_names=None,
        patterns=None,
    )
    net.eval().to(device)

    # embedder
    emb = CLIPEmbedder(device=device, model_name=args.clip_model, pretrained=args.clip_pretrained)

    # orchestrator
    print("[INFO] building orchestrator (all domains)...")
    orch = DomainOrchestrator(
        doms_all,
        lora_db_path=args.loradb,
        sim_metric=args.sim_metric,
        temperature=args.temperature,
        embedder_tag=args.embedder_tag,
    )

    # load all domain LoRAs into injected modules
    print("[INFO] loading all domain LoRA weights...")
    load_all_domain_loras(net, orch)
    net.to(device)

    # collect lora modules once
    lora_mods = collect_lora_modules(net)
    print(f"[INFO] detected LoRA modules = {len(lora_mods)}")

    if args.k_topk <= 0:
        args.k_topk = args.rank * args.rank

    if args.save_images:
        os.makedirs(os.path.join(args.output_dir, "triplets"), exist_ok=True)
        if args.save_singles:
            os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    if args.metrics_csv:
        os.makedirs(os.path.dirname(args.metrics_csv), exist_ok=True)
        if not os.path.isfile(args.metrics_csv):
            with open(args.metrics_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "run_name", "name",
                    "mode", "local", "global",
                    "top1_domain", "top1_w", "top2_w", "margin",
                    "psnr_base", "ssim_base", "psnr_auto", "ssim_auto",
                    "topk",
                    "reason",
                    "kselect_mode", "k_score_mode", "gamma_mode", "gamma_clip",
                    "k_topk", "k_alpha", "k_beta", "use_gamma", "k_ramp_mode",
                    "act_score_mode", "act_every_n",
                    "k_gamma", "k_layers", "k_local", "k_global", "k_local_ratio", "k_global_ratio",
                    "k_sc_mean", "k_ss_mean", "k_ss_scaled_mean", "k_S_first", "k_S_last",
                    "k_act_layers", "k_act_ratio",
                ])

    if args.routing_jsonl is None and args.save_images:
        args.routing_jsonl = os.path.join(args.output_dir, "routing.jsonl")
    if args.routing_jsonl:
        os.makedirs(os.path.dirname(args.routing_jsonl), exist_ok=True)

    tfm_to_tensor = T.ToTensor()

    base_psnr_list, base_ssim_list = [], []
    auto_psnr_list, auto_ssim_list = [], []

    def log_jsonl(obj: Dict[str, Any]):
        if not args.routing_jsonl:
            return
        with open(args.routing_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    for idx, (lq_path, gt_path_override) in enumerate(pairs):
        name = os.path.basename(lq_path)

        img = Image.open(lq_path).convert("RGB")
        x = tfm_to_tensor(img).unsqueeze(0).to(device)
        _, _, h, w = x.shape

        gt_path = gt_path_override
        if gt_path is None and args.gt_root is not None:
            gt_path = os.path.join(args.gt_root, name)

        gt = None
        if gt_path is not None and os.path.isfile(gt_path):
            gt_img = Image.open(gt_path).convert("RGB")
            gt = tfm_to_tensor(gt_img).unsqueeze(0).to(device)
            if gt.shape[-2:] != (h, w):
                gt = F.interpolate(gt, size=(h, w), mode="bilinear", align_corners=False)

        with torch.inference_mode():
            # base
            set_all_lora_domain_weights(net, None)
            y_base = forward_padded(net, x)

            # retrieval
            v = emb.embed_image(img)
            picks = orch.select_topk(v, top_k=max(args.topk, args.mix_topk))
            picks = [(d, float(wt)) for d, wt in picks]
            picks_short = picks[: args.topk]

            top1_dom, top1_w = picks_short[0]
            top2_w = picks_short[1][1] if len(picks_short) > 1 else 0.0
            margin = float(top1_w - top2_w)

            dec = decide_single_or_mix(
                picks=picks,
                single_tau=args.single_tau,
                single_margin=args.single_margin,
                mix_topk=args.mix_topk,
                local_domains=local_domains,
                global_domains=global_domains,
            )

            if args.verbose:
                print("=" * 90)
                print(f"[IMG {idx+1:04d}/{len(pairs):04d}] {name}")
                print(f"[route] topk(short={args.topk}) = {[(d, round(w,4)) for d,w in picks_short]}")
                if args.print_full_scores:
                    print(f"[route] topk(full={len(picks)}) = {[(d, round(w,6)) for d,w in picks]}")
                print(f"[route] top1={top1_dom}:{top1_w:.6f} top2={top2_w:.6f} margin={margin:.6f}")
                print(f"[route] mode={dec['mode']} reason={dec['reason']}")

            # auto
            kstats = {
                "kselect_mode": None, "score_mode": None, "gamma_mode": None, "gamma": None,
                "n_layers": None, "n_local": None, "n_global": None,
                "local_ratio": None, "global_ratio": None,
                "sc_mean": None, "ss_mean": None, "ss_scaled_mean": None,
                "S_first": None, "S_last": None,
                "act_layers": None, "act_ratio": None,
            }

            if dec["mode"] == "single":
                set_all_lora_domain_weights(net, {dec["top1"]: 1.0})
                y_auto = forward_padded(net, x)
            else:
                local = dec["local"]
                global_ = dec["global"]

                # build static plan (always)
                plan = build_kselect_static_plan(
                    net,
                    local_domain=local,
                    global_domain=global_,
                    k_topk=args.k_topk,
                    alpha=args.k_alpha,
                    beta=args.k_beta,
                    ramp_mode=args.k_ramp_mode,
                    score_mode=args.k_score_mode,
                    use_gamma=bool(args.use_gamma),
                    gamma_mode=args.gamma_mode,
                    gamma_clip=args.gamma_clip,
                    eps=args.score_eps,
                    verbose=args.verbose and (args.kselect_mode == "static"),
                )

                if args.kselect_mode == "static":
                    apply_kselect_plan(plan, local_domain=local, global_domain=global_)
                    y_auto = forward_padded(net, x)
                    kstats.update({
                        "kselect_mode": "static",
                        "score_mode": f"weight:{plan.get('score_mode','')}",
                        "gamma_mode": plan.get("gamma_mode", ""),
                        "gamma": float(plan.get("gamma_global", 1.0)),
                        "n_layers": int(plan.get("n_layers", 0)),
                        "n_local": int(plan.get("n_local", 0)),
                        "n_global": int(plan.get("n_global", 0)),
                        "local_ratio": float(plan.get("local_ratio", 0.0)),
                        "global_ratio": float(plan.get("global_ratio", 0.0)),
                        "sc_mean": float(plan.get("sc_mean", 0.0)),
                        "ss_mean": float(plan.get("ss_mean", 0.0)),
                        "ss_scaled_mean": float(plan.get("ss_scaled_mean", 0.0)),
                        "S_first": float(plan.get("S_first", 0.0)),
                        "S_last": float(plan.get("S_last", 0.0)),
                    })
                else:
                    # activation or hybrid: activation every N layers; N=1 equals full activation
                    act_every_n = 1 if args.kselect_mode == "activation" else max(1, args.act_every_n)

                    # reset: avoid any leftover mixing
                    set_all_lora_domain_weights(net, {d: 0.0 for d in doms_all})

                    y_auto, kstats2 = forward_padded_with_activation_kselect(
                        net,
                        x,
                        local_domain=local,
                        global_domain=global_,
                        plan=plan,
                        act_score_mode=args.act_score_mode,
                        act_every_n=act_every_n,
                        verbose=args.verbose,
                        factor=8,
                        eps=args.score_eps,
                    )
                    kstats.update(kstats2)

                if args.verbose:
                    print(f"[kselect] mode={kstats.get('kselect_mode')} gamma={kstats.get('gamma', 1.0):.6f} "
                          f"layers={kstats.get('n_layers')} local={kstats.get('n_local')} global={kstats.get('n_global')} "
                          f"local_ratio={kstats.get('local_ratio', 0.0):.3f} global_ratio={kstats.get('global_ratio', 0.0):.3f}")
                    if kstats.get("act_layers") is not None:
                        print(f"[kselect] act_layers={kstats.get('act_layers')} act_ratio={kstats.get('act_ratio', 0.0):.3f}")

        # metrics
        psnr_base = ssim_base = psnr_auto = ssim_auto = None
        if gt is not None:
            psnr_base = float(tensor_psnr(y_base, gt))
            ssim_base = float(tensor_ssim(y_base, gt))
            psnr_auto = float(tensor_psnr(y_auto, gt))
            ssim_auto = float(tensor_ssim(y_auto, gt))
            base_psnr_list.append(psnr_base)
            base_ssim_list.append(ssim_base)
            auto_psnr_list.append(psnr_auto)
            auto_ssim_list.append(ssim_auto)

            if args.verbose:
                print(f"[metric] PSNR base={psnr_base:.4f} auto={psnr_auto:.4f} | "
                      f"SSIM base={ssim_base:.4f} auto={ssim_auto:.4f}")

        # save images
        if args.save_images:
            trip_dir = os.path.join(args.output_dir, "triplets")
            os.makedirs(trip_dir, exist_ok=True)

            stem = os.path.splitext(name)[0]

            # 你 save_triplet 需要的是 tensor（一般 [1,C,H,W]）
            lq_t = x
            meta = {
                "name": name,
                "mode": dec["mode"],
                "top1": dec.get("top1"),
                "local": dec.get("local"),
                "global": dec.get("global"),
                "routing": [(d, float(w)) for d, w in picks_short],
                "kselect": kstats if dec["mode"] == "mix" else None,
            }

            # 三联对比：LQ | BASE | MIX/AUTO
            if args.concat:
                save_triplet(
                    out_dir=trip_dir,
                    stem=stem,
                    lq=lq_t,
                    base=y_base,
                    mix=y_auto,
                    save_concat=True,
                    save_singles=False,
                    save_lq=False,
                    annotate=getattr(args, "annotate", False),
                    meta=meta,
                )

        # logs (csv + jsonl)
        if args.metrics_csv:
            with open(args.metrics_csv, "a", newline="", encoding="utf-8") as f:
                wcsv = csv.writer(f)
                wcsv.writerow([
                    args.run_name, name,
                    dec["mode"], dec.get("local"), dec.get("global"),
                    top1_dom, top1_w, top2_w, margin,
                    psnr_base, ssim_base, psnr_auto, ssim_auto,
                    json.dumps([(d, float(w)) for d, w in picks_short], ensure_ascii=False),
                    dec["reason"],
                    args.kselect_mode, args.k_score_mode, args.gamma_mode, args.gamma_clip,
                    args.k_topk, args.k_alpha, args.k_beta, args.use_gamma, args.k_ramp_mode,
                    args.act_score_mode, args.act_every_n,
                    kstats.get("gamma"), kstats.get("n_layers"), kstats.get("n_local"), kstats.get("n_global"),
                    kstats.get("local_ratio"), kstats.get("global_ratio"),
                    kstats.get("sc_mean"), kstats.get("ss_mean"), kstats.get("ss_scaled_mean"),
                    kstats.get("S_first"), kstats.get("S_last"),
                    kstats.get("act_layers"), kstats.get("act_ratio"),
                ])

        log_jsonl({
            "run_name": args.run_name,
            "name": name,
            "lq_path": lq_path,
            "gt_path": gt_path,
            "topk_short": [(d, float(w)) for d, w in picks_short],
            "mode": dec["mode"],
            "decision": dec,
            "kselect_stats": kstats if dec["mode"] == "mix" else None,
            "metrics": {
                "psnr_base": psnr_base,
                "ssim_base": ssim_base,
                "psnr_auto": psnr_auto,
                "ssim_auto": ssim_auto,
            },
            "cfg": {
                "kselect_mode": args.kselect_mode,
                "k_score_mode": args.k_score_mode,
                "gamma_mode": args.gamma_mode,
                "gamma_clip": args.gamma_clip,
                "act_score_mode": args.act_score_mode,
                "act_every_n": args.act_every_n,
            } if dec["mode"] == "mix" else None,
        })

    # summary
    if args.summary_csv:
        os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
        mean_base_psnr = float(np.mean(base_psnr_list)) if base_psnr_list else None
        mean_base_ssim = float(np.mean(base_ssim_list)) if base_ssim_list else None
        mean_auto_psnr = float(np.mean(auto_psnr_list)) if auto_psnr_list else None
        mean_auto_ssim = float(np.mean(auto_ssim_list)) if auto_ssim_list else None

        file_exists = os.path.isfile(args.summary_csv)
        with open(args.summary_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow([
                    "run_name", "n",
                    "mean_psnr_base", "mean_ssim_base",
                    "mean_psnr_auto", "mean_ssim_auto",
                    "topk", "mix_topk", "single_tau", "single_margin",
                    "kselect_mode", "k_score_mode", "gamma_mode", "gamma_clip",
                    "k_topk", "k_alpha", "k_beta", "use_gamma", "k_ramp_mode",
                    "act_score_mode", "act_every_n",
                    "sim_metric", "temperature", "embedder", "embedder_tag",
                ])
            w.writerow([
                args.run_name, len(pairs),
                mean_base_psnr, mean_base_ssim,
                mean_auto_psnr, mean_auto_ssim,
                args.topk, args.mix_topk, args.single_tau, args.single_margin,
                args.kselect_mode, args.k_score_mode, args.gamma_mode, args.gamma_clip,
                args.k_topk, args.k_alpha, args.k_beta, args.use_gamma, args.k_ramp_mode,
                args.act_score_mode, args.act_every_n,
                args.sim_metric, args.temperature, args.embedder, args.embedder_tag,
            ])

    print("[DONE]")
    if base_psnr_list:
        print(f"[SUMMARY] base  PSNR={np.mean(base_psnr_list):.4f} SSIM={np.mean(base_ssim_list):.4f}")
        print(f"[SUMMARY] auto  PSNR={np.mean(auto_psnr_list):.4f} SSIM={np.mean(auto_ssim_list):.4f}")
    if args.routing_jsonl:
        print(f"[LOG] routing_jsonl: {args.routing_jsonl}")
    if args.metrics_csv:
        print(f"[LOG] metrics_csv: {args.metrics_csv}")


if __name__ == "__main__":
    main()

# lora_adapters/infer_cascade_data_v2.py
# -*- coding: utf-8 -*-
"""
Auto retrieval + cascade inference (v2)

Goal (v2 vs v1):
- Follow infer_data_ramp_new.py style: "decide_single_or_mix -> (if mix) cascade 2-stage"
- Keep the SAME CSV output structure / summary header as infer_cascade_data.py
- Default: residual DISABLED (stage2 uses stage1 output only)
- Support stage2 selection mode:
    --stage2_mode fixed   : stage2 domain decided from the initial (LQ) decision (no reroute)
    --stage2_mode reroute : reroute stage2 on stage1 output (y1) to pick stage2 domain

Notes:
- To preserve existing CSV schema, we keep fields for "stage2_res_*" but leave them empty/NaN.
- We encode stage2_source as "out_fixed" / "out_reroute" to avoid changing the summary header.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# -----------------------------
# basic utils
# -----------------------------
IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_domain_list(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def list_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    items: List[Path] = []
    for ext in IMG_EXTS:
        items.extend(sorted(input_path.rglob(f"*{ext}")))
    return items


def parse_pair_list(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            pairs.append((parts[0], parts[1]))
    return pairs


def match_gt_by_basename(lq_path: Path, gt_root: Path) -> Optional[Path]:
    """Try to match GT by basename/stem under gt_root."""
    cand = gt_root / lq_path.name
    if cand.exists():
        return cand

    stem = lq_path.stem
    for ext in IMG_EXTS:
        cand2 = gt_root / f"{stem}{ext}"
        if cand2.exists():
            return cand2

    # looser: scan once (costly, but ok for small sets)
    for p in gt_root.rglob("*"):
        if p.is_file() and p.stem == stem and p.suffix.lower() in IMG_EXTS:
            return p
    return None


def safe_mean(xs: List[float]) -> float:
    if len(xs) == 0:
        return float("nan")
    return float(np.mean(np.asarray(xs, dtype=np.float64)))


# -----------------------------
# metrics (fallback)
# -----------------------------
def tensor_psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-10) -> float:
    """x,y: [1,3,H,W] in [0,1]"""
    mse = torch.mean((x - y) ** 2).clamp_min(eps)
    psnr = -10.0 * torch.log10(mse)
    return float(psnr.item())


def _gaussian_window(window_size: int, sigma: float, channel: int, device: torch.device):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,ws,ws]
    w = w.repeat(channel, 1, 1, 1)  # [C,1,ws,ws]
    return w


def tensor_ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    """Simple SSIM (fallback). x,y: [1,3,H,W] in [0,1]"""
    assert x.shape == y.shape and x.dim() == 4 and x.size(0) == 1
    device = x.device
    channel = x.size(1)
    w = _gaussian_window(window_size, sigma, channel, device)

    mu_x = F.conv2d(x, w, padding=window_size // 2, groups=channel)
    mu_y = F.conv2d(y, w, padding=window_size // 2, groups=channel)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, w, padding=window_size // 2, groups=channel) - mu_x2
    sigma_y2 = F.conv2d(y * y, w, padding=window_size // 2, groups=channel) - mu_y2
    sigma_xy = F.conv2d(x * y, w, padding=window_size // 2, groups=channel) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12)
    return float(ssim_map.mean().item())


# -----------------------------
# LoRA helpers
# -----------------------------
def zero_single_lora(net: torch.nn.Module, domain_name: str = "_Single"):
    """Clear injected LoRA params for a given domain slot to avoid residue across stages."""
    for m in net.modules():
        if hasattr(m, "lora_down") and hasattr(m, "lora_up"):
            try:
                down = m.lora_down[domain_name]
                up = m.lora_up[domain_name]
                if hasattr(down, "weight") and down.weight is not None:
                    down.weight.data.zero_()
                if hasattr(up, "weight") and up.weight is not None:
                    up.weight.data.zero_()
            except Exception:
                continue


# -----------------------------
# routing utilities
# -----------------------------
@dataclass
class RouteInfo:
    metric: str
    topk: List[Tuple[str, float]]               # [(domain, weight)]
    scores: List[Tuple[str, float]]             # [(domain, raw_score)] for all domains (sorted desc)
    margin: float
    uncertain: bool
    reason: str


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / n


def _softmax(xs: np.ndarray, temperature: float) -> np.ndarray:
    t = max(1e-8, float(temperature))
    z = xs / t
    z = z - np.max(z)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)


def _proto_reduce(orch, sims: np.ndarray) -> float:
    """
    sims: [K] similarity of an embedding to K prototypes.
    Prefer orch.proto_reduce if exists, otherwise use max.
    """
    mode = getattr(orch, "proto_reduce", "max")
    if mode == "mean":
        return float(np.mean(sims))
    if mode == "sum":
        return float(np.sum(sims))
    # default "max"
    return float(np.max(sims))


def compute_scores(orch, emb: np.ndarray) -> List[Tuple[str, float]]:
    """
    Return raw similarity score per domain sorted desc.
    Domain object is expected to have `prototypes` field: (K,D) or (D,)
    """
    metric = getattr(orch, "sim_metric", "cosine")
    e = np.asarray(emb, dtype=np.float32).reshape(-1)
    scores: List[Tuple[str, float]] = []

    if metric == "cosine":
        e1 = _l2norm(e[None, :])[0]
        for name, dom in orch.domains.items():
            p = np.asarray(dom.prototypes, dtype=np.float32)
            if p.ndim == 1:
                p2 = _l2norm(p[None, :])[0]
                s = float(np.dot(e1, p2))
            elif p.ndim == 2:
                p2 = _l2norm(p)
                sims = p2 @ e1
                s = _proto_reduce(orch, sims)
            else:
                raise ValueError(f"Domain {name} prototypes shape unsupported: {p.shape}")
            scores.append((name, s))
    elif metric == "euclidean":
        for name, dom in orch.domains.items():
            p = np.asarray(dom.prototypes, dtype=np.float32)
            if p.ndim == 1:
                d = float(np.linalg.norm(e - p))
            elif p.ndim == 2:
                ds = np.linalg.norm(p - e[None, :], axis=1)
                # reduce distance then negate => "higher is better"
                d = float(np.min(ds))
            else:
                raise ValueError(f"Domain {name} prototypes shape unsupported: {p.shape}")
            scores.append((name, -d))
    else:
        raise ValueError(f"Unsupported sim_metric: {metric}")

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def compute_margin(metric: str, scores_sorted: List[Tuple[str, float]]) -> float:
    """top1 - top2 margin in score space."""
    if len(scores_sorted) < 2:
        return float("inf")
    return float(scores_sorted[0][1] - scores_sorted[1][1])


def route_with_margin(
    orch,
    emb: np.ndarray,
    top_k: int,
    temperature: float,
    min_margin: float,
    stage_name: str,
) -> RouteInfo:
    metric = getattr(orch, "sim_metric", "cosine")
    scores = compute_scores(orch, emb)
    margin = compute_margin(metric, scores)
    uncertain = margin < float(min_margin)

    raw = np.asarray([s for _, s in scores], dtype=np.float32)
    probs = _softmax(raw, temperature=float(temperature))
    top_k = max(1, int(top_k))
    idxs = np.argsort(-probs)[:top_k]
    dom_names = [scores[i][0] for i in idxs]
    dom_probs = [float(probs[i]) for i in idxs]
    topk = list(zip(dom_names, dom_probs))

    reason = f"{stage_name}: margin={margin:.4f} (thr={min_margin:.4f}) -> " + ("uncertain" if uncertain else "confident")
    return RouteInfo(metric=metric, topk=topk, scores=scores, margin=margin, uncertain=uncertain, reason=reason)


def decide_single_or_mix(
    picks: List[Tuple[str, float]],  # [(domain, weight)] (softmax weights)
    single_tau: float,
    single_margin: float,
    mix_topk: int,
    group1: List[str],
    group2: List[str],
) -> Dict:
    """
    Decide if we do:
      - single : use top1 domain only
      - mix    : do cascade with one domain from group1 and one domain from group2

    Return dict:
      mode: "single" | "mix"
      top1, top1_w, top2_w
      d1, d2 (if mix)
      reason
    """
    if not picks:
        return {"mode": "single", "top1": None, "top1_w": 0.0, "top2_w": 0.0,
                "d1": None, "d2": None, "reason": "empty picks -> single"}

    top1_d, top1_w = picks[0][0], float(picks[0][1])
    top2_w = float(picks[1][1]) if len(picks) > 1 else 0.0
    margin_w = top1_w - top2_w

    # single decision
    if (top1_w >= float(single_tau)) and (margin_w >= float(single_margin)):
        return {"mode": "single", "top1": top1_d, "top1_w": top1_w, "top2_w": top2_w,
                "d1": None, "d2": None,
                "reason": f"single (top1_w={top1_w:.3f}>=tau={single_tau}, w_margin={margin_w:.3f}>=m={single_margin})"}

    # mix: pick one from each group in top-mix_topk
    cand = picks[:max(1, int(mix_topk))]
    d1 = None
    d2 = None
    for d, _ in cand:
        if (d1 is None) and (d in group1):
            d1 = d
        if (d2 is None) and (d in group2):
            d2 = d

    if d1 is None or d2 is None or d1 == d2:
        return {"mode": "single", "top1": top1_d, "top1_w": top1_w, "top2_w": top2_w,
                "d1": None, "d2": None,
                "reason": "mix not possible (no group1/group2 in topk) -> fallback single"}

    return {"mode": "mix", "top1": top1_d, "top1_w": top1_w, "top2_w": top2_w,
            "d1": d1, "d2": d2, "reason": f"mix (d1={d1}, d2={d2})"}


# -----------------------------
# model forward (pad -> net -> crop)
# -----------------------------
@torch.no_grad()
def forward_restore(net: torch.nn.Module, x: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """x: [1,3,H,W] in [0,1]"""
    assert x.dim() == 4 and x.size(0) == 1
    _, _, h, w = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h or pad_w:
        x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    else:
        x_in = x
    y = net(x_in)
    y = y[:, :, :h, :w]
    return y.clamp(0, 1)


def tensor_to_pil01(x: torch.Tensor) -> Image.Image:
    """x: [1,3,H,W] in [0,1]"""
    x = x.detach().float().cpu().clamp(0, 1)[0]
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


# -----------------------------
# embedder tag align
# -----------------------------
def infer_embedder_tag(args) -> str:
    # keep consistent with your existing scripts; allow override
    if getattr(args, "embedder_tag", None):
        return args.embedder_tag

    if args.embedder == "clip":
        # normalize like: clip_vit-b-16
        name = str(args.clip_model).lower().replace("/", "-").replace("_", "-")
        return f"clip_{name}"
    if args.embedder == "dinov2":
        name = str(args.dino_model).lower().replace("/", "-").replace("_", "-")
        return f"dinov2_{name}"
    if args.embedder == "fft":
        return "fft_amp"
    if args.embedder == "fft_enhanced":
        return "fft_enh"
    return str(args.embedder)


def build_embedder(args, device: str):
    if args.embedder == "clip":
        from lora_adapters.embedding_clip import CLIPEmbedder  # type: ignore
        return CLIPEmbedder(model_name=args.clip_model, pretrained=args.clip_pretrained, device=device)
    if args.embedder == "dinov2":
        from lora_adapters.embedding_dinov2 import DINOv2Embedder  # type: ignore
        return DINOv2Embedder(model_name=args.dino_model, device=device)
    if args.embedder == "fft":
        from lora_adapters.embedding_fft import FFTEmbedder  # type: ignore
        return FFTEmbedder(mode="amp")
    if args.embedder == "fft_enhanced":
        from lora_adapters.embedding_fft import FFTEmbedder  # type: ignore
        return FFTEmbedder(mode="enh")
    raise ValueError(f"Unsupported embedder: {args.embedder}")


# -----------------------------
# CLI
# -----------------------------
def build_args():
    p = argparse.ArgumentParser()
    # input
    p.add_argument("--input", type=str, required=True, help="image file or folder")
    p.add_argument("--pair_list", type=str, default="", help="txt list: LQ_path GT_path")
    p.add_argument("--gt_root", type=str, default="", help="GT root dir (if no pair_list)")
    p.add_argument("--output_dir", type=str, required=True)

    # model / lora
    p.add_argument("--base_ckpt", type=str, required=True)
    p.add_argument("--yaml", type=str, required=True)
    p.add_argument("--loradb", type=str, required=True, help="weights/lora root")

    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=8.0)

    # domain groups
    p.add_argument("--local_domains", type=str, required=True)
    p.add_argument("--global_domains", type=str, required=True)
    p.add_argument("--order", type=str, default="local2global", choices=["local2global", "global2local"])

    # stage2 selection mode (NEW)
    p.add_argument("--stage2_mode", type=str, default="reroute", choices=["fixed", "reroute"],
                   help="fixed: stage2 domain from initial decision; reroute: re-route on y1 to pick stage2 domain")

    # routing / decision
    p.add_argument("--embedder", type=str, default="clip", choices=["clip", "dinov2", "fft", "fft_enhanced"])
    p.add_argument("--embedder_tag", type=str, default="", help="override embedder_tag (prototypes filename suffix)")
    p.add_argument("--sim_metric", type=str, default="cosine", choices=["cosine", "euclidean"])

    p.add_argument("--topk1", type=int, default=1)
    p.add_argument("--topk2", type=int, default=1)
    p.add_argument("--temperature1", type=float, default=0.05)
    p.add_argument("--temperature2", type=float, default=0.05)

    # decide_single_or_mix thresholds (from ramp style)
    p.add_argument("--single_tau", type=float, default=0.70, help="top1 weight >= tau => single (also need margin)")
    p.add_argument("--single_margin", type=float, default=0.15, help="top1_w - top2_w >= margin => single")
    p.add_argument("--mix_topk", type=int, default=5, help="search local/global candidates inside topk for mix decision")

    # margin skip thresholds (score-space margin, not prob margin); -1 disables
    p.add_argument("--stage1_min_margin", type=float, default=-1.0)
    p.add_argument("--stage2_min_margin", type=float, default=-1.0)

    # clip args
    p.add_argument("--clip_model", type=str, default="ViT-B-16")
    p.add_argument("--clip_pretrained", type=str, default="")

    # dino args
    p.add_argument("--dino_model", type=str, default="dinov2_vits14")

    # io / logging
    p.add_argument("--run_name", type=str, default="exp_cascade_v2")
    p.add_argument("--metrics_csv", type=str, default="")
    p.add_argument("--summary_csv", type=str, default="")

    p.add_argument("--save_base", action="store_true")
    p.add_argument("--save_stage_images", action="store_true")
    p.add_argument("--save_concat", action="store_true")

    # seed
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")

    p.add_argument("--device", type=str, default="cuda")
    return p


def set_seed(seed: int, deterministic: bool = False):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------------
# main
# -----------------------------
def main():
    args = build_args().parse_args()
    set_seed(args.seed, args.deterministic)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    dir_base = out_dir / "base"
    dir_s1 = out_dir / "stage1"
    dir_s2_out = out_dir / "stage2_out"
    dir_s2_res = out_dir / "stage2_res"       # kept for schema compatibility (unused)
    dir_concat = out_dir / "concat"
    for d in [dir_base, dir_s1, dir_s2_out, dir_s2_res, dir_concat]:
        ensure_dir(d)

    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else (out_dir / "metrics.csv")
    summary_csv = Path(args.summary_csv) if args.summary_csv else (out_dir / "summary.csv")

    # ---- build base model + inject LoRA(_Single) ----
    from lora_adapters.utils import build_histoformer  # type: ignore
    from lora_adapters.inject_lora import inject_lora  # type: ignore
    try:
        net = build_histoformer(args.yaml, args.base_ckpt, device=device)
    except TypeError:
        net = build_histoformer(weights=args.base_ckpt, yaml_file=args.yaml)

    inject_lora(net, rank=args.rank, alpha=args.alpha, domain_list=["_Single"])
    net = net.to(device)
    net.eval()
    print("[DEBUG] net param device:", next(net.parameters()).device)

    # ---- embedder ----
    embedder_tag = infer_embedder_tag(args)
    print(f"[INFO] embedder={args.embedder} | embedder_tag={embedder_tag}")
    emb = build_embedder(args, device=str(device))

    # ---- orchestrators ----
    from lora_adapters.domain_orchestrator import DomainOrchestrator  # type: ignore

    local_domains = parse_domain_list(args.local_domains)
    global_domains = parse_domain_list(args.global_domains)

    if args.order == "local2global":
        stage1_domains = local_domains
        stage2_domains = global_domains
        stage1_name = "local"
        stage2_name = "global"
    else:
        stage1_domains = global_domains
        stage2_domains = local_domains
        stage1_name = "global"
        stage2_name = "local"

    if not stage1_domains or not stage2_domains:
        raise ValueError("Empty stage1_domains or stage2_domains. Check --local_domains/--global_domains")

    # stage-specific orch
    orch1 = DomainOrchestrator(
        stage1_domains,
        lora_db_path=args.loradb,
        sim_metric=args.sim_metric,
        temperature=args.temperature1,
        embedder_tag=embedder_tag,
    )
    orch2 = DomainOrchestrator(
        stage2_domains,
        lora_db_path=args.loradb,
        sim_metric=args.sim_metric,
        temperature=args.temperature2,
        embedder_tag=embedder_tag,
    )
    # for decision on union
    all_domains = list(dict.fromkeys(stage1_domains + stage2_domains))
    orch_all = DomainOrchestrator(
        all_domains,
        lora_db_path=args.loradb,
        sim_metric=args.sim_metric,
        temperature=args.temperature1,
        embedder_tag=embedder_tag,
    )

    tfm_to_tensor = T.ToTensor()

    # ---- dataset items ----
    items: List[Tuple[str, Optional[str]]] = []
    if args.pair_list:
        pairs = parse_pair_list(Path(args.pair_list))
        for lq, gt in pairs:
            items.append((lq, gt))
    else:
        inp = Path(args.input)
        lqs = list_images(inp)
        if args.gt_root:
            gt_root = Path(args.gt_root)
            for lq in lqs:
                gt = match_gt_by_basename(lq, gt_root)
                items.append((str(lq), str(gt) if gt else None))
        else:
            for lq in lqs:
                items.append((str(lq), None))

    print(f"[INFO] total images: {len(items)}")
    run_name = args.run_name.strip() or f"cascade_v2_{args.order}_{args.stage2_mode}"
    print(f"[INFO] run_name={run_name}")

    # ---- CSV headers (keep same as v1) ----
    per_image_header = [
        "run_name",
        "lq_path", "gt_path", "stem",
        "order", "stage1_group", "stage2_group", "stage2_source",
        "embedder", "embedder_tag", "sim_metric",
        "topk1", "topk2", "temperature1", "temperature2",
        "stage1_min_margin", "stage2_min_margin",
        "stage1_topk", "stage1_scores", "stage1_margin", "stage1_uncertain", "stage1_reason",
        "stage2_out_topk", "stage2_out_scores", "stage2_out_margin", "stage2_out_uncertain", "stage2_out_reason",
        "stage2_res_topk", "stage2_res_scores", "stage2_res_margin", "stage2_res_uncertain", "stage2_res_reason",
        "psnr_lq", "ssim_lq",
        "psnr_base", "ssim_base",
        "psnr_stage1", "ssim_stage1",
        "psnr_stage2_out", "ssim_stage2_out",
        "psnr_stage2_res", "ssim_stage2_res",
        "out_base_path", "out_s1_path", "out_s2_out_path", "out_s2_res_path", "out_concat_path",
    ]

    summary_header = [
        "run_name",
        "n_images", "n_with_gt",
        "order", "stage1_group", "stage2_group", "stage2_source",
        "embedder", "embedder_tag", "sim_metric",
        "topk1", "topk2", "temperature1", "temperature2",
        "stage1_min_margin", "stage2_min_margin",
        "mean_psnr_lq", "mean_ssim_lq",
        "mean_psnr_base", "mean_ssim_base",
        "mean_psnr_stage1", "mean_ssim_stage1",
        "mean_psnr_stage2_out", "mean_ssim_stage2_out",
        "mean_psnr_stage2_res", "mean_ssim_stage2_res",
    ]

    # init csv writers
    need_write_header = (not metrics_csv.exists())
    mf = open(metrics_csv, "a", newline="", encoding="utf-8")
    mw = csv.DictWriter(mf, fieldnames=per_image_header)
    if need_write_header:
        mw.writeheader()
        mf.flush()

    # stats accumulators
    vals: Dict[str, List[float]] = {
        "psnr_lq": [], "ssim_lq": [],
        "psnr_base": [], "ssim_base": [],
        "psnr_s1": [], "ssim_s1": [],
        "psnr_s2_out": [], "ssim_s2_out": [],
        # keep for schema compatibility
        "psnr_s2_res": [], "ssim_s2_res": [],
    }
    n_with_gt = 0

    from lora_adapters.utils_merge import apply_weighted_lora, map_to_single_domain_keys  # type: ignore
    from lora_adapters.utils import save_image  # type: ignore

    stage2_source = f"out_{args.stage2_mode}"

    for idx, (lq_p, gt_p) in enumerate(items, 1):
        lq_path = Path(lq_p)
        gt_path = Path(gt_p) if gt_p else None
        stem = lq_path.stem

        # ---- load ----
        img_lq = Image.open(lq_path).convert("RGB")
        x = tfm_to_tensor(img_lq).unsqueeze(0).to(device)

        # ---- base ----
        zero_single_lora(net, "_Single")
        y_base = forward_restore(net, x)

        # ---- embed LQ ----
        emb_x = emb.embed_image(img_lq)

        # decision route on union
        r_all = route_with_margin(
            orch=orch_all, emb=emb_x,
            top_k=max(args.mix_topk, 2),
            temperature=args.temperature1,
            min_margin=-1e9,
            stage_name="decide",
        )
        dec = decide_single_or_mix(
            picks=r_all.topk,
            single_tau=args.single_tau,
            single_margin=args.single_margin,
            mix_topk=args.mix_topk,
            group1=stage1_domains,
            group2=stage2_domains,
        )

        # ---- Stage1 routing info (for log) ----
        thr1 = args.stage1_min_margin if args.stage1_min_margin >= 0 else -1e9
        r1 = route_with_margin(
            orch=orch1, emb=emb_x,
            top_k=args.topk1,
            temperature=args.temperature1,
            min_margin=thr1,
            stage_name="stage1",
        )

        # choose stage1 domain
        if dec["mode"] == "single":
            s1_dom = dec["top1"]
            s1_reason = f"SINGLE: {dec['reason']}"
            # overwrite stage1 topk for clarity
            r1_use = RouteInfo(
                metric=r1.metric,
                topk=[(s1_dom, 1.0)] if s1_dom else [],
                scores=r_all.scores,
                margin=r_all.margin,
                uncertain=False,
                reason=s1_reason,
            )
        else:
            s1_dom = dec["d1"]
            s1_reason = f"MIX: {dec['reason']}"
            r1_use = RouteInfo(
                metric=r1.metric,
                topk=[(s1_dom, 1.0)] if s1_dom else [],
                scores=r1.scores,
                margin=r1.margin,
                uncertain=r1.uncertain,
                reason=s1_reason + " | " + r1.reason,
            )

        # apply stage1
        y1 = y_base
        if s1_dom:
            merged1 = orch_all.build_weighted_lora([(s1_dom, 1.0)])
            merged1 = map_to_single_domain_keys(merged1, target_domain_name="_Single")
            zero_single_lora(net, "_Single")
            apply_weighted_lora(net, merged1)
            y1 = forward_restore(net, x)

        # ---- Stage2 (out only; residual disabled) ----
        y2_out = y1
        r2_out = RouteInfo(metric=r1.metric, topk=[], scores=[], margin=float("nan"),
                           uncertain=True, reason="stage2 not executed")
        stage2_skip = True

        if dec["mode"] == "mix":
            # initial stage2 domain from decision
            s2_dom_fixed = dec["d2"]

            # stage2 embedding source
            if args.stage2_mode == "fixed":
                emb_s2 = emb_x
            else:
                img_y1 = tensor_to_pil01(y1)
                emb_s2 = emb.embed_image(img_y1)

            thr2 = args.stage2_min_margin if args.stage2_min_margin >= 0 else -1e9
            r2_tmp = route_with_margin(
                orch=orch2, emb=emb_s2,
                top_k=args.topk2,
                temperature=args.temperature2,
                min_margin=thr2,
                stage_name="stage2",
            )

            # decide stage2 domain
            if args.stage2_mode == "fixed":
                s2_dom = s2_dom_fixed
                r2_out = RouteInfo(
                    metric=r2_tmp.metric,
                    topk=[(s2_dom, 1.0)] if s2_dom else [],
                    scores=r2_tmp.scores,
                    margin=r2_tmp.margin,
                    uncertain=r2_tmp.uncertain,
                    reason=f"fixed stage2={s2_dom} | {r2_tmp.reason}",
                )
            else:
                s2_dom = r2_tmp.topk[0][0] if r2_tmp.topk else None
                r2_out = RouteInfo(
                    metric=r2_tmp.metric,
                    topk=[(s2_dom, 1.0)] if s2_dom else [],
                    scores=r2_tmp.scores,
                    margin=r2_tmp.margin,
                    uncertain=r2_tmp.uncertain,
                    reason=f"reroute stage2={s2_dom} | {r2_tmp.reason}",
                )

            stage2_skip = (args.stage2_min_margin >= 0 and r2_tmp.uncertain) or (s2_dom is None)

            if not stage2_skip:
                merged2 = orch_all.build_weighted_lora([(s2_dom, 1.0)])
                merged2 = map_to_single_domain_keys(merged2, target_domain_name="_Single")
                zero_single_lora(net, "_Single")
                apply_weighted_lora(net, merged2)
                y2_out = forward_restore(net, y1)

        # ---- metrics (if gt available) ----
        psnr_lq = ssim_lq = psnr_base = ssim_base = psnr_s1 = ssim_s1 = psnr_s2o = ssim_s2o = float("nan")
        if gt_path and gt_path.exists():
            n_with_gt += 1
            img_gt = Image.open(gt_path).convert("RGB")
            gt = tfm_to_tensor(img_gt).unsqueeze(0).to(device)

            psnr_lq = tensor_psnr(x, gt)
            ssim_lq = tensor_ssim(x, gt)
            psnr_base = tensor_psnr(y_base, gt)
            ssim_base = tensor_ssim(y_base, gt)
            psnr_s1 = tensor_psnr(y1, gt)
            ssim_s1 = tensor_ssim(y1, gt)
            psnr_s2o = tensor_psnr(y2_out, gt)
            ssim_s2o = tensor_ssim(y2_out, gt)

            vals["psnr_lq"].append(psnr_lq)
            vals["ssim_lq"].append(ssim_lq)
            vals["psnr_base"].append(psnr_base)
            vals["ssim_base"].append(ssim_base)
            vals["psnr_s1"].append(psnr_s1)
            vals["ssim_s1"].append(ssim_s1)
            vals["psnr_s2_out"].append(psnr_s2o)
            vals["ssim_s2_out"].append(ssim_s2o)

        # stage2_res kept empty
        psnr_s2r = ssim_s2r = float("nan")

        # ---- save images ----
        out_base_path = out_s1_path = out_s2_out_path = out_s2_res_path = out_concat_path = ""
        if args.save_base:
            out_base_path = str((dir_base / f"{stem}_base.png").as_posix())
            save_image(y_base, out_base_path)

        if args.save_stage_images:
            s1_tag = r1_use.topk[0][0] if r1_use.topk else "none"
            out_s1_path = str((dir_s1 / f"{stem}_s1_{stage1_name}_{s1_tag}.png").as_posix())
            save_image(y1, out_s1_path)

            s2_tag = r2_out.topk[0][0] if r2_out.topk else "none"
            out_s2_out_path = str((dir_s2_out / f"{stem}_s2_{stage2_name}_{args.stage2_mode}_{s2_tag}.png").as_posix())
            save_image(y2_out, out_s2_out_path)

        if args.save_concat:
            parts = [x, y_base, y1, y2_out]
            concat = torch.cat(parts, dim=3)
            out_concat_path = str((dir_concat / f"{stem}_concat.png").as_posix())
            save_image(concat, out_concat_path)

        # ---- write per-image row ----
        row = {
            "run_name": run_name,
            "lq_path": str(lq_path),
            "gt_path": str(gt_path) if gt_path else "",
            "stem": stem,
            "order": args.order,
            "stage1_group": stage1_name,
            "stage2_group": stage2_name,
            "stage2_source": stage2_source,
            "embedder": args.embedder,
            "embedder_tag": embedder_tag,
            "sim_metric": args.sim_metric,
            "topk1": args.topk1,
            "topk2": args.topk2,
            "temperature1": args.temperature1,
            "temperature2": args.temperature2,
            "stage1_min_margin": args.stage1_min_margin,
            "stage2_min_margin": args.stage2_min_margin,
            "stage1_topk": json.dumps(r1_use.topk, ensure_ascii=False),
            "stage1_scores": json.dumps(r1_use.scores[:10], ensure_ascii=False),  # top10 for readability
            "stage1_margin": r1_use.margin,
            "stage1_uncertain": int(bool(r1_use.uncertain)),
            "stage1_reason": r1_use.reason,
            "stage2_out_topk": json.dumps(r2_out.topk, ensure_ascii=False),
            "stage2_out_scores": json.dumps(r2_out.scores[:10], ensure_ascii=False),
            "stage2_out_margin": r2_out.margin,
            "stage2_out_uncertain": int(bool(r2_out.uncertain)),
            "stage2_out_reason": r2_out.reason + (f" | skip={stage2_skip}" if dec["mode"] == "mix" else " | single"),
            # residual branch disabled
            "stage2_res_topk": "",
            "stage2_res_scores": "",
            "stage2_res_margin": "",
            "stage2_res_uncertain": "",
            "stage2_res_reason": "residual disabled",
            "psnr_lq": psnr_lq, "ssim_lq": ssim_lq,
            "psnr_base": psnr_base, "ssim_base": ssim_base,
            "psnr_stage1": psnr_s1, "ssim_stage1": ssim_s1,
            "psnr_stage2_out": psnr_s2o, "ssim_stage2_out": ssim_s2o,
            "psnr_stage2_res": psnr_s2r, "ssim_stage2_res": ssim_s2r,
            "out_base_path": out_base_path,
            "out_s1_path": out_s1_path,
            "out_s2_out_path": out_s2_out_path,
            "out_s2_res_path": out_s2_res_path,
            "out_concat_path": out_concat_path,
        }
        mw.writerow(row)
        mf.flush()

        # ---- console ----
        if gt_path and gt_path.exists():
            print(f"[{idx}/{len(items)}] {stem} | base {psnr_base:.2f}/{ssim_base:.4f} | "
                  f"s1 {psnr_s1:.2f}/{ssim_s1:.4f} | s2 {psnr_s2o:.2f}/{ssim_s2o:.4f} | "
                  f"mode={dec['mode']} | s1={r1_use.topk[0][0] if r1_use.topk else 'none'} | "
                  f"s2={r2_out.topk[0][0] if r2_out.topk else 'none'}")
        else:
            print(f"[{idx}/{len(items)}] {stem} | done | mode={dec['mode']}")

    mf.close()

    # ---- write summary ----
    need_write_sum_header = (not summary_csv.exists())
    sf = open(summary_csv, "a", newline="", encoding="utf-8")
    sw = csv.DictWriter(sf, fieldnames=summary_header)
    if need_write_sum_header:
        sw.writeheader()

    srow = {
        "run_name": run_name,
        "n_images": len(items),
        "n_with_gt": n_with_gt,
        "order": args.order,
        "stage1_group": stage1_name,
        "stage2_group": stage2_name,
        "stage2_source": stage2_source,
        "embedder": args.embedder,
        "embedder_tag": embedder_tag,
        "sim_metric": args.sim_metric,
        "topk1": args.topk1,
        "topk2": args.topk2,
        "temperature1": args.temperature1,
        "temperature2": args.temperature2,
        "stage1_min_margin": args.stage1_min_margin,
        "stage2_min_margin": args.stage2_min_margin,
        "mean_psnr_lq": safe_mean(vals["psnr_lq"]),
        "mean_ssim_lq": safe_mean(vals["ssim_lq"]),
        "mean_psnr_base": safe_mean(vals["psnr_base"]),
        "mean_ssim_base": safe_mean(vals["ssim_base"]),
        "mean_psnr_stage1": safe_mean(vals["psnr_s1"]),
        "mean_ssim_stage1": safe_mean(vals["ssim_s1"]),
        "mean_psnr_stage2_out": safe_mean(vals["psnr_s2_out"]),
        "mean_ssim_stage2_out": safe_mean(vals["ssim_s2_out"]),
        "mean_psnr_stage2_res": float("nan"),
        "mean_ssim_stage2_res": float("nan"),
    }
    sw.writerow(srow)
    sf.flush()
    sf.close()

    print("[DONE] metrics_csv:", metrics_csv)
    print("[DONE] summary_csv:", summary_csv)


if __name__ == "__main__":
    # allow both:
    #   python -m lora_adapters.infer_cascade_data_v2 ...
    #   python lora_adapters/infer_cascade_data_v2.py ...
    if __package__ is None or __package__ == "":
        # fix relative imports if launched as a script
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
    main()

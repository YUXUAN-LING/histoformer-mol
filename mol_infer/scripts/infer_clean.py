# mol_infer/scripts/infer_clean.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import csv
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

# ---- your existing repo deps ----
from lora_adapters.utils import build_histoformer
from lora_adapter.inject_lora import inject_lora
from lora_adapters.domain_orchestrator import DomainOrchestrator
from lora_adapters.embedding_clip import CLIPEmbedder
from lora_adapters.vis_utils import save_triplet

try:
    from lora_adapters.common.seed import set_seed
except Exception:
    def set_seed(seed: int, deterministic: bool = False):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

try:
    from lora_adapter.lora_linear import LoRALinear, LoRAConv2d
except Exception as e:
    raise ImportError(f"Cannot import LoRALinear/LoRAConv2d from lora_adapters.lora_linear: {e}")


# -------------------------
# basic metrics (fallback)
# -------------------------
@torch.inference_mode()
def tensor_psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-10) -> float:
    # x,y: [1,3,H,W] in [0,1]
    mse = torch.mean((x - y) ** 2).clamp_min(eps)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return float(psnr.item())


# -------------------------
# io helpers
# -------------------------
def read_pair_list(pair_list: str) -> List[Tuple[str, Optional[str]]]:
    pairs: List[Tuple[str, Optional[str]]] = []
    with open(pair_list, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) == 1:
                pairs.append((parts[0], None))
            else:
                pairs.append((parts[0], parts[1]))
    return pairs


def list_images_in_dir(root: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    out = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in exts:
            out.append(p)
    return out


def load_image_tensor(path: str, device: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = T.ToTensor()(img).unsqueeze(0).to(device)  # [1,3,H,W]
    return x


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def move_lora_modules_to_device(net: torch.nn.Module, device: str):
    """
    Ensure all LoRA submodules (lora_up/lora_down) are on the same device as input.
    This fixes: Input cuda FloatTensor vs weight cpu FloatTensor.
    """
    for m in net.modules():
        if hasattr(m, "lora_up") and hasattr(m, "lora_down"):
            # ModuleDict / dict both supported
            ups = getattr(m, "lora_up")
            downs = getattr(m, "lora_down")

            if isinstance(ups, dict):
                for _, sub in ups.items():
                    if hasattr(sub, "to"):
                        sub.to(device)
            else:
                # ModuleDict-like
                for _, sub in ups.items():
                    sub.to(device)

            if isinstance(downs, dict):
                for _, sub in downs.items():
                    if hasattr(sub, "to"):
                        sub.to(device)
            else:
                for _, sub in downs.items():
                    sub.to(device)

# -------------------------
# routing: decide single/mix
# -------------------------
@dataclass
class RoutingDecision:
    mode: str  # "single" or "mix"
    top1: str
    top1_w: float
    top2_w: float
    margin: float
    local: Optional[str] = None
    local_w: Optional[float] = None
    global_: Optional[str] = None
    global_w: Optional[float] = None
    reason: str = ""


def decide_single_or_mix(
    picks: List[Tuple[str, float]],
    single_tau: float,
    single_margin: float,
    mix_topk: int,
    local_domains: List[str],
    global_domains: List[str],
) -> RoutingDecision:
    if not picks:
        raise ValueError("[routing] empty picks")

    top1_d, top1_w = picks[0][0], float(picks[0][1])
    top2_w = float(picks[1][1]) if len(picks) >= 2 else 0.0
    margin = top1_w - top2_w

    if top1_w >= single_tau:
        return RoutingDecision(
            mode="single", top1=top1_d, top1_w=top1_w, top2_w=top2_w, margin=margin,
            reason=f"top1_w({top1_w:.3f})>=tau({single_tau:.3f})",
        )
    if margin >= single_margin:
        return RoutingDecision(
            mode="single", top1=top1_d, top1_w=top1_w, top2_w=top2_w, margin=margin,
            reason=f"margin({margin:.3f})>=single_margin({single_margin:.3f})",
        )

    top = picks[:max(1, min(mix_topk, len(picks)))]
    best_local: Optional[Tuple[str, float]] = None
    best_global: Optional[Tuple[str, float]] = None

    for d, w in top:
        if (d in local_domains) and best_local is None:
            best_local = (d, float(w))
        if (d in global_domains) and best_global is None:
            best_global = (d, float(w))
        if best_local and best_global:
            break

    if best_local is None:
        best_local = (top1_d, top1_w)
    if best_global is None:
        best_global = (top1_d, top1_w)

    return RoutingDecision(
        mode="mix",
        top1=top1_d, top1_w=top1_w, top2_w=top2_w, margin=margin,
        local=best_local[0], local_w=best_local[1],
        global_=best_global[0], global_w=best_global[1],
        reason=f"mix: local={best_local[0]}({best_local[1]:.3f}), global={best_global[0]}({best_global[1]:.3f})",
    )


# -------------------------
# LoRA helpers
# -------------------------
def iter_lora_modules(net: torch.nn.Module):
    for m in net.modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            yield m


def set_all_lora_domain_weights(net: torch.nn.Module, weights: Optional[Dict[str, float]]):
    w = weights or {}
    for m in iter_lora_modules(net):
        m.set_domain_weights(w)


def load_lora_sd(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "params" in ckpt and isinstance(ckpt["params"], dict):
            return ckpt["params"]
        return ckpt
    raise ValueError(f"Unknown LoRA checkpoint type: {type(ckpt)}")


def load_all_domain_loras_by_path(net: torch.nn.Module, orch: DomainOrchestrator):
    """
    不依赖你旧 infer_data.py 的 load_all_domain_loras，直接：
      for each domain: net.load_state_dict(lora_sd, strict=False)
    因为 domain 参数名已经注入进模型（lora_up.<domain> / lora_down.<domain>），不会冲突。
    """
    dom_map = getattr(orch, "domains", {})
    if not dom_map:
        raise ValueError("[lora] orchestrator has empty domains")

    for d, dom in dom_map.items():
        p = getattr(dom, "lora_path", None)
        if not p or not os.path.isfile(p):
            raise FileNotFoundError(f"[lora] missing lora file for domain={d}: {p}")
        sd = load_lora_sd(p)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        # 不 print 太多，只在需要时你可以打开 verbose


# -------------------------
# KSelect-None (local/global/none) layerwise
# -------------------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def ramp(pos: float, mode: str) -> float:
    mode = (mode or "sigmoid").lower()
    pos = float(np.clip(pos, 0.0, 1.0))
    if mode == "linear":
        return pos
    # sigmoid centered at 0.5
    return _sigmoid((pos - 0.5) * 10.0)


@torch.inference_mode()
def lora_delta_norm(m: torch.nn.Module, domain: str) -> float:
    """
    计算该层 domain LoRA 的“强度”：
      || (W_up @ W_down) ||_F * scale
    """
    up = getattr(getattr(m, "lora_up", None), domain, None)
    down = getattr(getattr(m, "lora_down", None), domain, None)
    if up is None or down is None:
        return 0.0

    Wu = up.weight
    Wd = down.weight

    # flatten conv1x1 to matrix
    Wu2 = Wu.view(Wu.shape[0], -1)
    Wd2 = Wd.view(Wd.shape[0], -1)
    # shapes:
    #   up: [out, r] or [out, r,1,1] -> [out, r]
    #   down: [r, in] or [r, in,1,1] -> [r, in]
    if Wd2.shape[0] != Wu2.shape[1]:
        # fallback: treat as independent norms
        base = torch.norm(Wu2, p="fro") * torch.norm(Wd2, p="fro")
    else:
        base = torch.norm(Wu2 @ Wd2, p="fro")

    scale = float(getattr(m, "scale", 1.0))
    return float((base * scale).item())


def apply_kselect_none_layerwise(
    net: torch.nn.Module,
    local_domain: str,
    global_domain: str,
    k_alpha: float,
    k_beta: float,
    k_ramp_mode: str,
    none_beta: float,
    none_alpha: float,
    none_mode: str,
) -> Dict[str, float]:
    """
    三选一规则：
      sL = k_alpha * norm(local) * ramp(pos)
      sG = k_beta  * norm(global)
      T_none(pos) = none_beta + none_alpha * f(pos)

    if max(sL,sG) < T_none(pos): 选择 none (base)
    else 选择 local/global 更大的那个
    """
    mods = list(iter_lora_modules(net))
    n = max(1, len(mods))
    cnt_local = 0
    cnt_global = 0
    cnt_none = 0

    for i, m in enumerate(mods):
        pos = 0.0 if n == 1 else (i / (n - 1))
        r = ramp(pos, k_ramp_mode)
        sL = (k_alpha * lora_delta_norm(m, local_domain) * r)
        sG = (k_beta * lora_delta_norm(m, global_domain))

        if (none_mode or "const").lower() == "ramp":
            thr = none_beta + none_alpha * r
        else:
            thr = none_beta + none_alpha * pos

        if max(sL, sG) < thr:
            # none/base
            m.set_domain_weights({})
            cnt_none += 1
        else:
            if sL >= sG:
                m.set_domain_weights({local_domain: 1.0})
                cnt_local += 1
            else:
                m.set_domain_weights({global_domain: 1.0})
                cnt_global += 1

    return {
        "k_layers": float(n),
        "k_local": float(cnt_local),
        "k_global": float(cnt_global),
        "k_none": float(cnt_none),
        "k_local_ratio": float(cnt_local / n),
        "k_global_ratio": float(cnt_global / n),
        "k_none_ratio": float(cnt_none / n),
    }


# -------------------------
# model forward (pad)
# -------------------------
@torch.inference_mode()
def forward_padded(net: torch.nn.Module, x: torch.Tensor, factor: int = 8) -> torch.Tensor:
    _, _, h, w = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h or pad_w:
        x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    else:
        x_in = x
    y = net(x_in).clamp(0, 1)
    return y[:, :, :h, :w]


# -------------------------
# main
# -------------------------
def build_parser():
    p = argparse.ArgumentParser("mol_infer clean: base/single/mix + kselect_none")

    p.add_argument("--input", required=True, help="folder or a single image path")
    p.add_argument("--pair_list", default=None, help="txt: lq [gt]")
    p.add_argument("--gt_root", default=None)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--base_ckpt", required=True)
    p.add_argument("--yaml", default=None, help="histoformer yaml config (can be used to build arch)")

    p.add_argument("--loradb_root", required=True)
    p.add_argument("--domains", required=True, help="comma-separated")
    p.add_argument("--local_domains", required=True, help="comma-separated")
    p.add_argument("--global_domains", required=True, help="comma-separated")

    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--enable_patch_lora", action="store_true")

    p.add_argument("--embedder", default="clip", choices=["clip"])
    p.add_argument("--clip_model", default="ViT-B-16")
    p.add_argument("--clip_pretrained", default="openai", help="openai or local bin path")
    p.add_argument("--embedder_tag", default="clip_vit-b-16", help="matches avg_embedding_<tag>.npy")

    p.add_argument("--sim_metric", default="cosine", choices=["cosine", "euclidean", "l2"])
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--mix_topk", type=int, default=5)
    p.add_argument("--single_tau", type=float, default=0.72)
    p.add_argument("--single_margin", type=float, default=0.10)
    p.add_argument("--norm_topk_domains", type=int, default=0,
                   help="if >0, only normalize retrieval scores over top-N domains (+ local/global)")

    p.add_argument("--enable_lora", action="store_true", help="enable single/mix output, else only base")
    p.add_argument("--enable_fusion", action="store_true", help="enable kselect_none for mix")
    p.add_argument("--fusion", default="none", choices=["none", "kselect_none"])

    # kselect-none params
    p.add_argument("--k_alpha", type=float, default=1.0)
    p.add_argument("--k_beta", type=float, default=1.0)
    p.add_argument("--k_ramp_mode", default="sigmoid", choices=["sigmoid", "linear"])
    p.add_argument("--none_mode", default="const", choices=["const", "ramp"])
    p.add_argument("--none_beta", type=float, default=0.15)
    p.add_argument("--none_alpha", type=float, default=0.0)

    # save/log
    p.add_argument("--save_images", action="store_true")
    p.add_argument("--concat", action="store_true", help="save triplet concat")
    p.add_argument("--save_singles", action="store_true")
    p.add_argument("--save_lq", action="store_true")
    p.add_argument("--annotate", action="store_true")

    p.add_argument("--metrics_csv", default=None)
    p.add_argument("--routing_jsonl", default=None)
    p.add_argument("--summary_csv", default=None)
    p.add_argument("--run_name", default="run")

    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true")

    p.add_argument("--print_full_scores", action="store_true")

    return p


def main():
    args = build_parser().parse_args()

    device = "cuda" if (str(args.device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
    set_seed(args.seed, deterministic=args.deterministic)

    ensure_dir(args.output_dir)
    trip_dir = os.path.join(args.output_dir, "triplets")
    ensure_dir(trip_dir)

    # ---------- build net ----------
    # 这里直接用 build_histoformer 构建结构；权重用 base_ckpt 加载
    net = build_histoformer(weights=args.base_ckpt, yaml_file=args.yaml)
    net = net.to(device).eval()

    # load base
    ckpt = torch.load(args.base_ckpt, map_location="cpu")
    if isinstance(ckpt, dict):
        if "params" in ckpt and isinstance(ckpt["params"], dict):
            sd = ckpt["params"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        raise ValueError(f"Unexpected base ckpt type: {type(ckpt)}")
    net.load_state_dict(sd, strict=False)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    local_domains = [d.strip() for d in args.local_domains.split(",") if d.strip()]
    global_domains = [d.strip() for d in args.global_domains.split(",") if d.strip()]

    # inject lora
    inject_lora(
        net,
        rank=args.rank,
        domain_list=domains,
        alpha=args.alpha,
        enable_patch_lora=args.enable_patch_lora,
    )

    # ---------- retrieval components ----------
    embedder = CLIPEmbedder(
        pretrained=args.clip_pretrained,
        device=device,
    )
    orch = DomainOrchestrator(
        domains=domains,
        lora_db_path=args.loradb_root,
        sim_metric=args.sim_metric,
        temperature=args.temperature,
        embedder_tag=args.embedder_tag,
    )
    # 关键：把 prototype reshape 成 1D，避免 (1,D) dot (D,) 的坑
    for d, dom in getattr(orch, "domains", {}).items():
        if hasattr(dom, "avg_embedding") and isinstance(dom.avg_embedding, np.ndarray):
            dom.avg_embedding = dom.avg_embedding.reshape(-1)
        if hasattr(dom, "prototypes") and isinstance(dom.prototypes, np.ndarray):
            dom.prototypes = dom.prototypes.reshape(dom.prototypes.shape[0], -1)

    # load all LoRA weights once
    load_all_domain_loras_by_path(net, orch)
    move_lora_modules_to_device(net, device)

    # ---------- dataset ----------
    pairs: List[Tuple[str, Optional[str]]] = []
    if args.pair_list:
        pairs = read_pair_list(args.pair_list)
        # resolve relative paths
        resolved = []
        for lq, gt in pairs:
            lq_p = lq if os.path.isabs(lq) else os.path.join(args.input, lq)
            if gt is None:
                gt_p = None
            else:
                root = args.gt_root if args.gt_root else args.input
                gt_p = gt if os.path.isabs(gt) else os.path.join(root, gt)
            resolved.append((lq_p, gt_p))
        pairs = resolved
    else:
        if os.path.isdir(args.input):
            imgs = list_images_in_dir(args.input)
            pairs = [(p, None) for p in imgs]
        else:
            pairs = [(args.input, None)]

    # ---------- loggers ----------
    metrics_f = None
    metrics_writer = None
    if args.metrics_csv:
        metrics_f = open(args.metrics_csv, "w", newline="", encoding="utf-8")
        metrics_writer = csv.writer(metrics_f)
        metrics_writer.writerow([
            "run_name", "name",
            "mode", "top1", "top1_w", "top2_w", "margin",
            "local", "local_w", "global", "global_w",
            "psnr_base", "psnr_auto",
            "fusion",
            "k_layers", "k_local", "k_global", "k_none",
            "k_local_ratio", "k_global_ratio", "k_none_ratio",
            "reason"
        ])

    jsonl_f = open(args.routing_jsonl, "w", encoding="utf-8") if args.routing_jsonl else None

    base_psnr_all: List[float] = []
    auto_psnr_all: List[float] = []

    # ---------- loop ----------
    for idx, (lq_path, gt_path) in enumerate(pairs):
        name = os.path.basename(lq_path)
        stem = os.path.splitext(name)[0]

        # load tensor for restoration
        lq = load_image_tensor(lq_path, device=device)

        # base
        set_all_lora_domain_weights(net, None)
        y_base = forward_padded(net, lq)

        # retrieval (use image path -> PIL -> embed)
        img_pil = Image.open(lq_path).convert("RGB")
        emb = embedder.embed_image(img_pil).reshape(-1)  # [D]
        picks = orch.select_topk(
            emb,
            top_k=args.topk,
            temperature=args.temperature,
            norm_topk_domains=int(args.norm_topk_domains),
            include_domains=local_domains + global_domains,
        )

        if args.print_full_scores:
            # raw scores
            dom_map = getattr(orch, "domains", {})
            raw = []
            for d, dom in dom_map.items():
                raw.append((d, float(orch.cosine(emb, dom.avg_embedding))))
            raw.sort(key=lambda x: -x[1])
            print("[debug raw_scores]", raw[:10])

        dec = decide_single_or_mix(
            picks=picks,
            single_tau=args.single_tau,
            single_margin=args.single_margin,
            mix_topk=args.mix_topk,
            local_domains=local_domains,
            global_domains=global_domains,
        )

        # auto output
        fusion_tag = "none"
        kstats = {
            "k_layers": 0.0, "k_local": 0.0, "k_global": 0.0, "k_none": 0.0,
            "k_local_ratio": 0.0, "k_global_ratio": 0.0, "k_none_ratio": 0.0,
        }

        if not args.enable_lora:
            y_auto = y_base
        else:
            if dec.mode == "single":
                set_all_lora_domain_weights(net, {dec.top1: 1.0})
                y_auto = forward_padded(net, lq)
            else:
                # mix
                assert dec.local is not None and dec.global_ is not None
                if args.enable_fusion and args.fusion == "kselect_none":
                    fusion_tag = "kselect_none"
                    kstats = apply_kselect_none_layerwise(
                        net,
                        local_domain=dec.local,
                        global_domain=dec.global_,
                        k_alpha=args.k_alpha,
                        k_beta=args.k_beta,
                        k_ramp_mode=args.k_ramp_mode,
                        none_beta=args.none_beta,
                        none_alpha=args.none_alpha,
                        none_mode=args.none_mode,
                    )
                    y_auto = forward_padded(net, lq)
                else:
                    fusion_tag = "mix_weighted"
                    set_all_lora_domain_weights(net, {dec.local: float(dec.local_w), dec.global_: float(dec.global_w)})
                    y_auto = forward_padded(net, lq)

        # metrics (if gt exists)
        psnr_b = 0.0
        psnr_a = 0.0
        if gt_path and os.path.isfile(gt_path):
            gt = load_image_tensor(gt_path, device=device)
            psnr_b = tensor_psnr(y_base, gt)
            psnr_a = tensor_psnr(y_auto, gt)
            base_psnr_all.append(psnr_b)
            auto_psnr_all.append(psnr_a)

        # save triplet (your current signature style)
        if args.save_images and args.concat:
            meta = {
                "mode": dec.mode,
                "top1": dec.top1,
                "top1_w": dec.top1_w,
                "margin": dec.margin,
                "local": dec.local,
                "global": dec.global_,
                "fusion": fusion_tag,
                **kstats,
            }
            save_triplet(
                out_dir=trip_dir,
                stem=stem,
                lq=lq,
                base=y_base,
                mix=y_auto,
                save_concat=True,
                save_singles=args.save_singles,
                save_lq=args.save_lq,
                annotate=args.annotate,
                meta=meta,
            )

        # write logs
        if metrics_writer is not None:
            metrics_writer.writerow([
                args.run_name, name,
                dec.mode, dec.top1, f"{dec.top1_w:.6f}", f"{dec.top2_w:.6f}", f"{dec.margin:.6f}",
                dec.local, f"{(dec.local_w or 0.0):.6f}", dec.global_, f"{(dec.global_w or 0.0):.6f}",
                f"{psnr_b:.4f}", f"{psnr_a:.4f}",
                fusion_tag,
                int(kstats["k_layers"]), int(kstats["k_local"]), int(kstats["k_global"]), int(kstats["k_none"]),
                f"{kstats['k_local_ratio']:.4f}", f"{kstats['k_global_ratio']:.4f}", f"{kstats['k_none_ratio']:.4f}",
                dec.reason
            ])

        if jsonl_f is not None:
            jsonl_f.write(json.dumps({
                "name": name,
                "picks": picks,
                "decision": dec.__dict__,
                "fusion": fusion_tag,
                "kstats": kstats,
            }, ensure_ascii=False) + "\n")

        if (idx + 1) % 20 == 0:
            print(f"[{idx+1}/{len(pairs)}] {name} mode={dec.mode} fusion={fusion_tag}")

    if metrics_f is not None:
        metrics_f.close()
    if jsonl_f is not None:
        jsonl_f.close()

    # summary
    n = len(base_psnr_all)
    if args.summary_csv:
        with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run_name", "n", "mean_psnr_base", "mean_psnr_auto"])
            if n > 0:
                w.writerow([args.run_name, n, np.mean(base_psnr_all), np.mean(auto_psnr_all)])
            else:
                w.writerow([args.run_name, 0, "", ""])

    print(f"[DONE] n={len(pairs)}  gt_pairs={n}")
    if n > 0:
        print(f"[SUMMARY] mean_psnr base={np.mean(base_psnr_all):.4f} auto={np.mean(auto_psnr_all):.4f}")


if __name__ == "__main__":
    main()

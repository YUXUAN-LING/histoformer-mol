# lora_adapters/infer_data_kselect.py
# -*- coding: utf-8 -*-
"""
KSelect unified inference (for Histoformer + LoRA library):

pipeline:
  retrieval (CLIP/DINO) -> decide(single/mix) -> KSelect layerwise select(local/global) -> restoration

Key behaviors (debug-friendly):
  - AUTO decide single vs mix (NO manual mode switch)
  - Default save: only triplet concat (LQ|BASE|AUTO) into {output_dir}/triplets
  - Optional save singles via --save_singles [--save_lq]
  - Per-image detailed logs (console + metrics_csv + routing_jsonl)

Recommended run:
  export PYTHONPATH=.
  python -m lora_adapters.infer_data_kselect \
    --input samples/haze_rain \
    --pair_list data/txt_lists/val_list_haze_rain.txt \
    --output_dir test_results/haze_rain_kselect \
    --loradb weights/lora \
    --domains rain,rain2,rain3,rainy,snow,snow1,haze,haze1,haze2,low,low1 \
    --local_domains rain,rain2,rain3,rainy,snow,snow1 \
    --global_domains haze,haze1,haze2,low,low1 \
    --embedder clip --clip_model ViT-B-16 --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
    --sim_metric euclidean --temperature 0.07 \
    --topk 5 --mix_topk 5 --single_tau 0.72 --single_margin 0.10 \
    --k_alpha 1.0 --k_beta 1.0 --use_gamma 1 --k_ramp_mode sigmoid \
    --rank 16 --alpha 16 \
    --gt_root samples/clear \
    --metrics_csv test_results/haze_rain_kselect/metrics.csv \
    --routing_jsonl test_results/haze_rain_kselect/routing.jsonl \
    --run_name haze_rain_kselect \
    --save_images --concat \
    --seed 42 --deterministic --verbose
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
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
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


def iter_lora_named_modules(net):
    """Yield (name, module) for LoRA modules in stable order."""
    for name, m in net.named_modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            yield name, m


def collect_lora_modules(net) -> List[Tuple[str, Any]]:
    return list(iter_lora_named_modules(net))


# ---------------------------
# decision: single or mix
# ---------------------------
def decide_single_or_mix(
    picks,  # [(domain, weight)] 已经按 weight 降序
    single_tau: float,
    single_margin: float,
    mix_topk: int,
    local_domains,
    global_domains,
):
    top1_d, top1_w = picks[0][0], float(picks[0][1])
    top2_w = float(picks[1][1]) if len(picks) >= 2 else 0.0

    # --- single 判定（保持你现在的规则） ---
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

    # --- mix: 在 local / global 内分别找最高的那个 ---
    # 注意：这里 cand 你可以用 top mix_topk，也可以直接用 picks 全部。
    cand = picks[:max(1, mix_topk)]

    best_local = None   # (d, w)
    best_global = None  # (d, w)

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

    # 可选：如果 local/global 恰好同名（一般不会），fallback
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
# KSelect (K-LoRA-like) helpers
# ---------------------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _ramp_value(pos01: float, mode: str, alpha: float, beta: float) -> float:
    """
    Paper: S = alpha * (tnow/tall) + beta, then apply to style side.
    Here we map LoRA-module order to pos01 in [0,1].
    mode:
      - linear: S = alpha*pos + beta
      - sigmoid: S = alpha*sigmoid(6*(pos-0.5)) + beta   (smooth)
    """
    pos01 = float(max(0.0, min(1.0, pos01)))
    if mode == "linear":
        return float(alpha * pos01 + beta)
    if mode == "sigmoid":
        # center at 0.5, slope ~6
        return float(alpha * _sigmoid(6.0 * (pos01 - 0.5)) + beta)
    # fallback
    return float(alpha * pos01 + beta)


def _get_domain_up_down(m, domain: str):
    """
    Compatible with your LoRALinear/LoRAConv2d in this repo:
      m.lora_up[domain], m.lora_down[domain]
    """
    if not hasattr(m, "lora_up") or not hasattr(m, "lora_down"):
        raise RuntimeError("LoRA module does not have lora_up/lora_down dicts.")
    if domain not in m.lora_up or domain not in m.lora_down:
        raise KeyError(f"domain={domain} not found in this LoRA module.")
    return m.lora_up[domain], m.lora_down[domain]


@torch.no_grad()
def _delta_abs_topk_sum(m, domain: str, k_topk: int) -> float:
    """
    Compute sum of top-K abs elements of deltaW = B@A (Linear) or equivalent flatten (Conv).
    This is the "importance" proxy used by KSelect.
    """
    up, down = _get_domain_up_down(m, domain)

    # weights
    A = down.weight  # usually [r, in] for linear; conv shapes possible
    B = up.weight    # usually [out, r]

    # flatten to 2D for a stable delta estimation
    A2 = A.reshape(A.shape[0], -1).float().abs()
    B2 = B.reshape(B.shape[0], -1).float().abs()

    # exact delta magnitude proxy:
    # |B@A| but using abs(A), abs(B) keeps a consistent "dominance" notion without sign cancel.
    # delta_abs approx = (|B| @ |A|).
    # This avoids sign issues and is cheaper / stable.
    try:
        delta = torch.matmul(B2, A2)  # [out, in_flat]
    except Exception:
        # fallback: outer-product-like score
        delta = (B2.mean(dim=1, keepdim=True) * A2.mean(dim=1, keepdim=True).T)

    flat = delta.reshape(-1)
    if k_topk <= 0 or k_topk >= flat.numel():
        return float(flat.sum().item())
    topv = torch.topk(flat, k=k_topk, largest=True).values
    return float(topv.sum().item())


@torch.no_grad()
def _compute_gamma(net, local_domain: str, global_domain: str) -> float:
    """
    Paper gamma balances magnitude differences between two LoRAs:
      gamma = sum_l sum_i |Wc_l,i| / sum_l sum_j |Ws_l,j|
    Here we approximate using sum of abs in (up,down) weights across all LoRA modules.
    """
    sum_local = 0.0
    sum_global = 0.0
    for _, m in iter_lora_named_modules(net):
        upL, downL = _get_domain_up_down(m, local_domain)
        upG, downG = _get_domain_up_down(m, global_domain)
        sum_local += float(upL.weight.abs().sum().item() + downL.weight.abs().sum().item())
        sum_global += float(upG.weight.abs().sum().item() + downG.weight.abs().sum().item())
    if sum_global <= 1e-12:
        return 1.0
    return float(sum_local / sum_global)


@torch.no_grad()
def apply_kselect_layerwise(
    net,
    local_domain: str,
    global_domain: str,
    k_topk: int,
    alpha: float,
    beta: float,
    use_gamma: bool,
    ramp_mode: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    For each LoRA module (ordered), compute:
      Sc = topK_sum(|deltaW_local|)
      Ss = topK_sum(|deltaW_global|)
      Ss' = Ss * (gamma * S(pos))
      choose local if Sc >= Ss' else global

    Return stats for logging.
    """
    mods = list(iter_lora_named_modules(net))
    if len(mods) == 0:
        raise RuntimeError("[kselect] found 0 LoRA modules. Injection failed?")

    gamma = _compute_gamma(net, local_domain, global_domain) if use_gamma else 1.0

    chosen = []
    sc_list = []
    ss_list = []
    ss_scaled_list = []
    s_list = []

    for i, (name, m) in enumerate(mods):
        pos = 0.0 if len(mods) == 1 else float(i) / float(len(mods) - 1)
        S = _ramp_value(pos, ramp_mode, alpha=alpha, beta=beta)  # increases with pos (later => more global/style)
        S_eff = float(gamma * S)

        Sc = _delta_abs_topk_sum(m, local_domain, k_topk=k_topk)
        Ss = _delta_abs_topk_sum(m, global_domain, k_topk=k_topk)
        Ss_scaled = float(Ss * S_eff)

        pick = local_domain if Sc >= Ss_scaled else global_domain
        chosen.append(pick)
        sc_list.append(Sc)
        ss_list.append(Ss)
        ss_scaled_list.append(Ss_scaled)
        s_list.append(S)

        # apply weights: hard select (1.0 / 0.0)
        if pick == local_domain:
            m.set_domain_weights({local_domain: 1.0, global_domain: 0.0})
        else:
            m.set_domain_weights({local_domain: 0.0, global_domain: 1.0})

        if verbose and (i < 5 or i >= len(mods) - 2):
            print(f"[kselect][layer {i:03d}] pos={pos:.3f} S={S:.4f} gamma={gamma:.4f} "
                  f"Sc(local)={Sc:.3e} Ss(global)={Ss:.3e} Ss'={Ss_scaled:.3e} -> pick={pick} | {name}")

    # stats
    n_local = sum([1 for p in chosen if p == local_domain])
    n_global = len(chosen) - n_local
    return {
        "gamma": gamma,
        "n_layers": len(chosen),
        "n_local": n_local,
        "n_global": n_global,
        "local_ratio": float(n_local / max(1, len(chosen))),
        "global_ratio": float(n_global / max(1, len(chosen))),
        # lightweight summaries (don’t dump huge arrays into CSV by default)
        "sc_mean": float(np.mean(sc_list)) if sc_list else 0.0,
        "ss_mean": float(np.mean(ss_list)) if ss_list else 0.0,
        "ss_scaled_mean": float(np.mean(ss_scaled_list)) if ss_scaled_list else 0.0,
        "S_first": float(s_list[0]) if s_list else 0.0,
        "S_last": float(s_list[-1]) if s_list else 0.0,
    }


# ---------------------------
# main
# ---------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser()

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
    ap.add_argument("--clip_pretrained", type=str, default=None)
    ap.add_argument("--embedder_tag", type=str, default=None)
    ap.add_argument("--sim_metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--mix_topk", type=int, default=5)

    # decision
    ap.add_argument("--single_tau", type=float, default=0.72)
    ap.add_argument("--single_margin", type=float, default=0.10)

    # kselect
    ap.add_argument("--k_topk", type=int, default=0, help="Top-K elements in deltaW to score each layer. 0=>rank*rank")
    ap.add_argument("--k_alpha", type=float, default=1.0)
    ap.add_argument("--k_beta", type=float, default=1.0)
    ap.add_argument("--use_gamma", type=int, default=1)
    ap.add_argument("--k_ramp_mode", type=str, default="sigmoid", choices=["linear", "sigmoid"])
    ap.add_argument("--mix_mode", type=str, default="kselect",
                choices=["kselect", "act_kselect_dy_ch"],
                help="mix strategy: kselect (layerwise) or act_kselect_dy_ch (channel gate by dy)")

    # dy-channel mode args
    ap.add_argument("--act_layer_topk", type=int, default=30)
    ap.add_argument("--act_layer_tau", type=float, default=0.05)
    ap.add_argument("--act_ch_topk", type=int, default=32)
    ap.add_argument("--act_enable_both", type=int, default=1)
    ap.add_argument("--act_both_tau", type=float, default=0.05)
    ap.add_argument("--act_both_ratio", type=float, default=0.6)
    ap.add_argument("--act_print_top", type=int, default=30)
    ap.add_argument("--act_recompute", type=int, default=0,
                    help="recompute gate on apply forward (slower); default uses probe gate")

    # saving
    ap.add_argument("--save_images", action="store_true")
    ap.add_argument("--concat", action="store_true", help="save triplet concat (LQ|BASE|AUTO)")
    ap.add_argument("--save_singles", action="store_true", help="also save base/auto single images")
    ap.add_argument("--save_lq", action="store_true", help="if save_singles, also save lq")

    # logs / csv
    ap.add_argument("--metrics_csv", type=str, default=None)
    ap.add_argument("--routing_jsonl", type=str, default=None)
    ap.add_argument("--summary_csv", type=str, default=None)
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--print_full_scores", action="store_true", help="print full similarity list per image (can be long)")

    # runtime
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # deterministic setup
    if args.deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_seed(args.seed, deterministic=args.deterministic)

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    # embedder_tag (must match prototype dims!)
    args.embedder_tag = args.embedder_tag or get_embedder_tag(args)

    print(f"[INFO] device={device}")
    print(f"[INFO] run_name={args.run_name}")
    print(f"[INFO] embedder={args.embedder}, embedder_tag={args.embedder_tag}")
    print(f"[INFO] sim_metric={args.sim_metric}, temperature={args.temperature}")

    # pairs
    if args.pair_list:
        pairs = parse_pair_list(args.pair_list)
    else:
        assert args.input is not None, "Need --pair_list or --input"
        imgs = list_images(args.input) if os.path.isdir(args.input) else [args.input]
        pairs = [(p, None) for p in imgs]

    # domains
    doms_all = [d.strip() for d in args.domains.split(",") if d.strip()]
    local_domains = [d.strip() for d in args.local_domains.split(",") if d.strip()]
    global_domains = [d.strip() for d in args.global_domains.split(",") if d.strip()]
    local_set = set(local_domains)
    global_set = set(global_domains)

    print(f"[INFO] domains={doms_all}")
    print(f"[INFO] local_domains={local_domains}")
    print(f"[INFO] global_domains={global_domains}")
    print(f"[INFO] total pairs={len(pairs)}")

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

    # load all domain LoRAs
    print("[INFO] loading all domain LoRA weights...")
    load_all_domain_loras(net, orch)
    # safety: ensure everything moved to device after loading
    net.to(device)

    # collect lora modules once
    lora_mods = collect_lora_modules(net)
    print(f"[INFO] detected LoRA modules = {len(lora_mods)}")

    # k_topk default
    if args.k_topk <= 0:
        args.k_topk = args.rank * args.rank
    print(f"[INFO] kselect: k_topk={args.k_topk}, k_alpha={args.k_alpha}, k_beta={args.k_beta}, "
          f"use_gamma={args.use_gamma}, k_ramp_mode={args.k_ramp_mode}")

    # prepare output folders
    if args.save_images:
        os.makedirs(os.path.join(args.output_dir, "triplets"), exist_ok=True)
        if args.save_singles:
            os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    # csv header
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
                    "k_topk", "k_alpha", "k_beta", "use_gamma", "k_ramp_mode",
                    "k_gamma", "k_layers", "k_local", "k_global", "k_local_ratio", "k_global_ratio",
                    "k_sc_mean", "k_ss_mean", "k_ss_scaled_mean", "k_S_first", "k_S_last",
                ])

    # routing jsonl
    if args.routing_jsonl is None and args.save_images:
        args.routing_jsonl = os.path.join(args.output_dir, "routing.jsonl")
    if args.routing_jsonl:
        os.makedirs(os.path.dirname(args.routing_jsonl), exist_ok=True)

    tfm_to_tensor = T.ToTensor()

    # stats
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

        # GT resolve
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
            # ---------------- base ----------------
            set_all_lora_domain_weights(net, None)
            y_base = forward_padded(net, x)

            # ---------------- retrieval ----------------
            v = emb.embed_image(img)
            # ask for more than needed so we can do mix selection from top mix_topk
            picks = orch.select_topk(v, top_k=max(args.topk, args.mix_topk))
            picks = [(d, float(wt)) for d, wt in picks]
            picks_short = picks[: args.topk]

            top1_dom, top1_w = picks_short[0]
            top2_w = picks_short[1][1] if len(picks_short) > 1 else 0.0
            margin = float(top1_w - top2_w)

            # decision (AUTO)
            dec = decide_single_or_mix(
                picks=picks,
                single_tau=args.single_tau,
                single_margin=args.single_margin,
                mix_topk=args.mix_topk,
                local_domains=local_domains,
                global_domains=global_domains,
            )

            # print per-image routing logs
            if args.verbose:
                print("=" * 90)
                print(f"[IMG {idx+1:04d}/{len(pairs):04d}] {name}")
                print(f"[route] topk(short={args.topk}) = {[(d, round(w,4)) for d,w in picks_short]}")
                if args.print_full_scores:
                    print(f"[route] topk(full={len(picks)}) = {[(d, round(w,6)) for d,w in picks]}")
                print(f"[route] top1={top1_dom}:{top1_w:.6f} top2={top2_w:.6f} margin={margin:.6f}")
                print(f"[route] mode={dec['mode']} reason={dec['reason']}")

            # ---------------- apply LoRA (AUTO) ----------------
            kstats = {
                "gamma": None, "n_layers": None, "n_local": None, "n_global": None,
                "local_ratio": None, "global_ratio": None,
                "sc_mean": None, "ss_mean": None, "ss_scaled_mean": None, "S_first": None, "S_last": None,
            }

            if dec["mode"] == "single":
                # hard single
                set_all_lora_domain_weights(net, {dec["top1"]: 1.0})
                if args.verbose:
                    print(f"[apply] single domain={dec['top1']} (weight=1.0)")

            else:
                local = dec["local"]
                global_ = dec["global"]

                if args.mix_mode == "act_kselect_dy_ch":
                    from lora_adapters.act_kselect_dy_channel import ActDyChConfig, ActKSelectDyChannelController

                    if args.verbose:
                        print(f"[apply] mix via ACT_DY_CH: dom1(local)={local}, dom2(global)={global_} "
                            f"layer_topk={args.act_layer_topk} layer_tau={args.act_layer_tau} ch_topk={args.act_ch_topk}")

                    # 关键：先把 LoRA scalar 权重全部关掉（base forward），由 hook 决定加多少
                    set_all_lora_domain_weights(net, None)

                    cfg = ActDyChConfig(
                        dom1=local, dom2=global_,
                        layer_topk=args.act_layer_topk,
                        layer_tau=args.act_layer_tau,
                        ch_topk=args.act_ch_topk,
                        enable_both=bool(args.act_enable_both),
                        both_tau=args.act_both_tau,
                        both_ratio=args.act_both_ratio,
                        print_top_modules=args.act_print_top,
                        verbose=args.verbose,
                        recompute_gate_on_apply=bool(args.act_recompute),
                    )

                    ctrl = ActKSelectDyChannelController(net, cfg)

                    # ---- probe pass：收集每层 dy/score/gate，不改变输出（仍是 base）
                    ctrl.set_mode("probe")
                    _ = forward_padded(net, x)

                    selected_names, items_sorted = ctrl.finalize_selection()

                    # 打印 summary + per-layer 排序日志
                    if args.verbose:
                        ctrl.print_summary(items_sorted)

                    # ---- apply pass：只在 selected layers 内做 channel gate 注入
                    ctrl.set_mode("apply")
                    y_auto = forward_padded(net, x)

                    ctrl.remove()

                else:
                    # 你原来的 KSelect（layerwise hard selection）
                    if args.verbose:
                        print(f"[apply] mix via KSelect: local={local} global={global_}")
                    kstats = apply_kselect_layerwise(
                        net,
                        local_domain=local,
                        global_domain=global_,
                        k_topk=args.k_topk,
                        alpha=args.k_alpha,
                        beta=args.k_beta,
                        use_gamma=bool(args.use_gamma),
                        ramp_mode=args.k_ramp_mode,
                        verbose=args.verbose,
                    )
                    if args.verbose:
                        print(f"[kselect] gamma={kstats['gamma']:.6f} "
                            f"layers={kstats['n_layers']} local={kstats['n_local']} global={kstats['n_global']} "
                            f"local_ratio={kstats['local_ratio']:.3f} global_ratio={kstats['global_ratio']:.3f}")

                    y_auto = forward_padded(net, x)

            # forward auto
            y_auto = forward_padded(net, x)

        # ---------------- metrics ----------------
        psnr_base = ssim_base = psnr_auto = ssim_auto = None
        if gt is not None:
            def to_float(x):
                return float(x.item()) if hasattr(x, "item") else float(x)

            psnr_base = to_float(tensor_psnr(y_base, gt))
            ssim_base = to_float(tensor_ssim(y_base, gt))
            psnr_auto = to_float(tensor_psnr(y_auto, gt))
            ssim_auto = to_float(tensor_ssim(y_auto, gt))

            base_psnr_list.append(psnr_base)
            base_ssim_list.append(ssim_base)
            auto_psnr_list.append(psnr_auto)
            auto_ssim_list.append(ssim_auto)

            if args.verbose:
                print(f"[metric] PSNR base={psnr_base:.4f} auto={psnr_auto:.4f} | "
                      f"SSIM base={ssim_base:.4f} auto={ssim_auto:.4f}")

        # ---------------- saving ----------------
        if args.save_images:
            # auto suffix indicates which branch it actually used (for clarity)
            auto_suffix = "single" if dec["mode"] == "single" else "mix"
            if args.concat:
                trip_dir = os.path.join(args.output_dir, "triplets")

                # name: "00002.png" -> stem: "00002"
                stem0 = os.path.splitext(name)[0]
                stem = f"{stem0}_auto_{auto_suffix}"

                # 这里 lq 必须是 tensor（你的 img 是 PIL），用你已有的 tfm_to_tensor / 或 x
                lq_t = x  # 如果 x 是 [1,3,H,W] 的 LQ tensor（你前面已经做了）
                # 或者：lq_t = tfm_to_tensor(img).unsqueeze(0).to(device)

                meta = {
                    "mode": dec["mode"],
                    "local": dec.get("local", ""),
                    "global": dec.get("global", ""),
                    "psnr_base": psnr_base,
                    "psnr_mix":  psnr_auto,   # 注意：auto 这支在 triplet 里对应 mix 槽位
                    "ssim_base": ssim_base,
                    "ssim_mix":  ssim_auto,
                }


                save_triplet(
                    out_dir=trip_dir,
                    stem=stem,
                    lq=lq_t,          # tensor
                    base=y_base,      # tensor
                    mix=y_auto,       # tensor
                    save_concat=True,
                    save_singles=False,
                    save_lq=False,
                    annotate=getattr(args, "annotate", False),
                    meta=meta,
                )

            if args.save_singles:
                # singles dir
                out_dir = os.path.join(args.output_dir, "images")
                os.makedirs(out_dir, exist_ok=True)
                if args.save_lq:
                    img.save(os.path.join(out_dir, name.replace(".", "_lq.")))
                T.ToPILImage()(y_base[0].detach().clamp(0, 1).cpu()).save(
                    os.path.join(out_dir, name.replace(".", "_base."))
                )
                T.ToPILImage()(y_auto[0].detach().clamp(0, 1).cpu()).save(
                    os.path.join(out_dir, name.replace(".", f"_auto_{auto_suffix}."))
                )

        # ---------------- write per-image CSV + JSONL ----------------
        topk_str = json.dumps([(d, float(w)) for d, w in picks_short], ensure_ascii=False)
        if args.metrics_csv:
            with open(args.metrics_csv, "a", newline="", encoding="utf-8") as f:
                wcsv = csv.writer(f)
                wcsv.writerow([
                    args.run_name, name,
                    dec["mode"], dec.get("local"), dec.get("global"),
                    top1_dom, top1_w, top2_w, margin,
                    psnr_base, ssim_base, psnr_auto, ssim_auto,
                    topk_str,
                    dec["reason"],
                    args.k_topk, args.k_alpha, args.k_beta, args.use_gamma, args.k_ramp_mode,
                    kstats["gamma"], kstats["n_layers"], kstats["n_local"], kstats["n_global"],
                    kstats["local_ratio"], kstats["global_ratio"],
                    kstats["sc_mean"], kstats["ss_mean"], kstats["ss_scaled_mean"], kstats["S_first"], kstats["S_last"],
                ])

        log_jsonl({
            "run_name": args.run_name,
            "name": name,
            "lq_path": lq_path,
            "gt_path": gt_path,
            "mode": dec["mode"],
            "topk_short": [(d, float(w)) for d, w in picks_short],
            "topk_full": [(d, float(w)) for d, w in picks] if args.print_full_scores else None,
            "top1": {"domain": top1_dom, "w": float(top1_w)},
            "top2": {"w": float(top2_w)},
            "margin": float(margin),
            "decision": dec,
            "kselect_stats": kstats if dec["mode"] == "mix" else None,
            "metrics": {
                "psnr_base": psnr_base,
                "ssim_base": ssim_base,
                "psnr_auto": psnr_auto,
                "ssim_auto": ssim_auto,
            }
        })

    # ---------------- summary ----------------
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
                    "k_topk", "k_alpha", "k_beta", "use_gamma", "k_ramp_mode",
                    "sim_metric", "temperature", "embedder", "embedder_tag",
                ])
            w.writerow([
                args.run_name, len(pairs),
                mean_base_psnr, mean_base_ssim,
                mean_auto_psnr, mean_auto_ssim,
                args.topk, args.mix_topk, args.single_tau, args.single_margin,
                args.k_topk, args.k_alpha, args.k_beta, args.use_gamma, args.k_ramp_mode,
                args.sim_metric, args.temperature, args.embedder, args.embedder_tag,
            ])

    print("[DONE]")
    if base_psnr_list:
        print(f"[SUMMARY] base  PSNR={np.mean(base_psnr_list):.4f} SSIM={np.mean(base_ssim_list):.4f}")
        print(f"[SUMMARY] auto  PSNR={np.mean(auto_psnr_list):.4f} SSIM={np.mean(auto_ssim_list):.4f}")
    if args.routing_jsonl:
        print(f"[LOG] routing_jsonl saved to: {args.routing_jsonl}")
    if args.metrics_csv:
        print(f"[LOG] metrics_csv saved to: {args.metrics_csv}")


if __name__ == "__main__":
    main()

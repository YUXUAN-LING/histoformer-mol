# lora_adapters/infer_data_ramp_new.py
# -*- coding: utf-8 -*-
"""
Ramp unified inference:
  retrieval -> decision(single/mix) -> ramp layerwise weights -> restoration

Refactor:
  - Visualization / saving logic moved to lora_adapters.vis_utils
  - Default in CONCAT mode: save only triplet concat (LQ|BASE|MIX).
    (If you also want single images, pass --save_singles [--save_lq].)
  - Existing inference / routing / CSV / summary functionality is preserved.
"""

import os
import csv
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from lora_adapters.utils import build_histoformer
from lora_adapters.inject_lora import inject_lora
from lora_adapters.vis_utils import save_triplet

# ---- embedder + orchestrator (support both module paths) ----
try:
    from lora_adapters.embedding_clip import CLIPEmbedder
except Exception as e:
    raise ImportError(f"Cannot import CLIPEmbedder from lora_adapters.embedding_clip: {e}")


from lora_adapters.domain_orchestrator import DomainOrchestrator  # fallback


# ---- reuse proven utilities from infer_data.py ----
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
# ramp helpers
# ---------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def iter_lora_named_modules(net):
    """Yield (name, module) for LoRA modules in stable order."""
    for name, m in net.named_modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            yield name, m


def set_lora_domain_weights_ramp(
    net,
    local_domain: str,
    global_domain: str,
    p0: float,
    k: float,
    eps: float,
):
    """
    For LoRA modules ordered by iter_lora_named_modules():
      pos in [0,1] -> p = sigmoid(k*(pos-p0))
      weights = {local: 1-p, global: p}
    """
    mods = list(iter_lora_named_modules(net))
    if len(mods) == 0:
        raise RuntimeError("[ramp] found 0 LoRA modules. Injection failed?")

    n = len(mods)
    for i, (_, m) in enumerate(mods):
        pos = 0.0 if n == 1 else float(i) / float(n - 1)
        p = _sigmoid(k * (pos - p0))
        p = float(eps + (1.0 - 2.0 * eps) * p)
        p = max(0.0, min(1.0, p))
        m.set_domain_weights({local_domain: 1.0 - p, global_domain: p})


def decide_single_or_mix(
    picks: List[Tuple[str, float]],  # [(domain, weight)]
    single_tau: float,
    single_margin: float,
    mix_topk: int,
    local_domains: List[str],
    global_domains: List[str],
) -> Dict:
    """
    Return dict:
      mode: "single" or "mix"
      top1, top1_w, top2_w
      local, global (if mix)
      reason
    """
    top1_d, top1_w = picks[0][0], float(picks[0][1])
    top2_w = float(picks[1][1]) if len(picks) >= 2 else 0.0

    # single criteria
    if top1_w >= single_tau:
        return {"mode": "single", "top1": top1_d, "top1_w": top1_w, "top2_w": top2_w,
                "reason": f"top1_w>=tau({top1_w:.3f}>={single_tau})",
                "local": None, "global": None}

    if (top1_w - top2_w) >= single_margin:
        return {"mode": "single", "top1": top1_d, "top1_w": top1_w, "top2_w": top2_w,
                "reason": f"top1-top2>=margin({top1_w-top2_w:.3f}>={single_margin})",
                "local": None, "global": None}

    # mix: choose one local + one global from top-mix_topk
    cand = picks[:max(1, mix_topk)]
    local = None
    global_ = None
    for d, _ in cand:
        if (local is None) and (d in local_domains):
            local = d
        if (global_ is None) and (d in global_domains):
            global_ = d

    if local is None or global_ is None or local == global_:
        # fallback
        return {"mode": "single", "top1": top1_d, "top1_w": top1_w, "top2_w": top2_w,
                "reason": "mix not possible (no local/global in topk) -> fallback single",
                "local": None, "global": None}

    return {"mode": "mix", "top1": top1_d, "top1_w": top1_w, "top2_w": top2_w,
            "reason": f"mix (local={local}, global={global_})",
            "local": local, "global": global_}


# ---------------------------
# io helpers
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
            elif len(parts) == 1:
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
# main
# ---------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument('--base_ckpt', required=True)
    ap.add_argument('--yaml', required=True)

    ap.add_argument('--input', required=True, help='LQ 图像目录或单张图片')
    ap.add_argument('--output', required=True, help='输出目录')

    ap.add_argument('--loradb', required=True, help='LoRA 权重所在根目录，例如 weights/lora')
    ap.add_argument('--domains', required=True, help='参与候选的域列表，如 "rain2,low,rain,..."')

    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=float, default=16.0)
    ap.add_argument('--device', default='cuda')

    # embedder / retrieval
    ap.add_argument('--embedder', choices=['clip'], default='clip')
    ap.add_argument('--clip_model', default='ViT-B-16')
    ap.add_argument('--clip_pretrained', type=str, required=True)
    ap.add_argument('--embedder_tag', type=str, default=None)

    ap.add_argument('--sim_metric', default='euclidean')
    ap.add_argument('--temperature', type=float, default=0.07)
    ap.add_argument('--topk', type=int, default=3)

    # decision
    ap.add_argument('--single_tau', type=float, default=0.60)
    ap.add_argument('--single_margin', type=float, default=0.20)
    ap.add_argument('--mix_topk', type=int, default=3)

    # ramp
    ap.add_argument('--ramp_p0', type=float, default=0.45)
    ap.add_argument('--ramp_k', type=float, default=6.0)
    ap.add_argument('--ramp_eps', type=float, default=0.0)

    # eval pairs / csv
    ap.add_argument(
        '--pair_list',
        default=None,
        help='可选：TXT 文件，每行 "LQ_path GT_path"；若提供则覆盖 --input 目录遍历'
    )
    ap.add_argument('--metrics_csv', type=str, default=None)
    ap.add_argument('--summary_csv', type=str, default=None)
    ap.add_argument('--run_name', type=str, default='run')

    # outputs
    ap.add_argument('--save_images', action='store_true', help='保存输出图像')
    ap.add_argument('--concat', action='store_true', help='保存三联对比图（LQ|BASE|MIX）')

    # new: optional singles
    ap.add_argument('--save_singles', action='store_true', help='额外保存 base/mix 单张图（默认不保存）')
    ap.add_argument('--save_lq', action='store_true', help='与 --save_singles 配合，额外保存 lq 单张图（默认不保存）')
    ap.add_argument('--annotate_concat', action='store_true', help='在 concat 图上叠加文字信息（若 PIL 可用）')

    # reproducibility
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--deterministic', action='store_true')

    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # seed
    set_seed(args.seed, deterministic=args.deterministic)

    device = 'cuda' if (args.device.startswith('cuda') and torch.cuda.is_available()) else 'cpu'

    # embedder_tag (same as infer_data.py)
    args.embedder_tag = args.embedder_tag or get_embedder_tag(args)

    print(f"[INFO] run_name = {args.run_name}")
    print(f"[INFO] embedder = {args.embedder}, embedder_tag = {args.embedder_tag}")

    # domains
    doms = [d.strip() for d in args.domains.split(',') if d.strip()]
    print(f"[INFO] domains = {doms}")

    # local/global split
    local_domains = [d for d in doms if (d.startswith("rain") or d.startswith("snow") or d == "rainy")]
    global_domains = [d for d in doms if (d.startswith("haze") or d.startswith("low"))]
    print(f"[INFO] local_domains = {local_domains}")
    print(f"[INFO] global_domains = {global_domains}")

    # build base model + inject LoRA slots
    print('[INFO] building base Histoformer...')
    net = build_histoformer(weights=args.base_ckpt, yaml_file=args.yaml).to(device)
    net = inject_lora(
        net,
        rank=args.rank,
        domain_list=doms,
        alpha=args.alpha,
        target_names=None,
        patterns=None
    )
    net.eval().to(device)

    # embedder
    emb = CLIPEmbedder(
        device=device,
        model_name=args.clip_model,
        pretrained=args.clip_pretrained
    )

    # orchestrator
    orch = DomainOrchestrator(
        doms,
        lora_db_path=args.loradb,
        sim_metric=args.sim_metric,
        temperature=args.temperature,
        embedder_tag=args.embedder_tag
    )

    # CRITICAL: load all domain LoRA weights
    load_all_domain_loras(net, orch)

    # build input list / pairs
    if args.pair_list:
        pairs = parse_pair_list(args.pair_list)
    else:
        imgs = list_images(args.input) if os.path.isdir(args.input) else [args.input]
        pairs = [(p, None) for p in imgs]

    print(f"[INFO] total images: {len(pairs)}")

    tfm_to_tensor = T.ToTensor()

    # metrics csv
    if args.metrics_csv:
        os.makedirs(os.path.dirname(args.metrics_csv), exist_ok=True)
        if not os.path.isfile(args.metrics_csv):
            with open(args.metrics_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "run_name", "name",
                    "mode", "local", "global",
                    "top1_domain", "top1_w", "top2_w",
                    "psnr_base", "ssim_base", "psnr_mix", "ssim_mix",
                    "topk", "reason",
                    "ramp_p0", "ramp_k", "ramp_eps",
                ])

    # run stats
    base_psnr_list, base_ssim_list = [], []
    mix_psnr_list, mix_ssim_list = [], []

    def run_one(lq_path: str, gt_path: Optional[str]):
        img = Image.open(lq_path).convert("RGB")
        x = tfm_to_tensor(img).unsqueeze(0).to(device)   # [1,3,H,W]
        _, _, h, w = x.shape

        # pad to 8x
        factor = 8
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h != 0 or pad_w != 0:
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_in = x

        with torch.inference_mode():
            # base output: disable LoRA
            set_all_lora_domain_weights(net, None)
            y_base = net(x_in)[:, :, :h, :w]

            # routing
            v = emb.embed_image(img)
            picks = orch.select_topk(v, top_k=args.topk)
            picks = [(d, float(wt)) for d, wt in picks]

            # decision
            dec = decide_single_or_mix(
                picks=picks,
                single_tau=args.single_tau,
                single_margin=args.single_margin,
                mix_topk=args.mix_topk,
                local_domains=local_domains,
                global_domains=global_domains
            )

            # apply weights
            if dec["mode"] == "single":
                set_all_lora_domain_weights(net, {dec["top1"]: 1.0})
            else:
                set_lora_domain_weights_ramp(
                    net,
                    local_domain=dec["local"],
                    global_domain=dec["global"],
                    p0=args.ramp_p0,
                    k=args.ramp_k,
                    eps=args.ramp_eps
                )

            y_mix = net(x_in)[:, :, :h, :w]

        metrics = {
            "name": os.path.basename(lq_path),
            "psnr_base": None, "ssim_base": None,
            "psnr_mix": None, "ssim_mix": None,
            "mode": dec["mode"],
            "local": dec["local"],
            "global": dec["global"],
            "top1_domain": dec["top1"],
            "top1_w": dec["top1_w"],
            "top2_w": dec["top2_w"],
            "reason": dec["reason"],
            "topk": picks,
        }

        if gt_path is not None and os.path.isfile(gt_path):
            gt_img = Image.open(gt_path).convert("RGB")
            gt = tfm_to_tensor(gt_img).unsqueeze(0).to(device)
            if gt.shape[-2:] != (h, w):
                gt = F.interpolate(gt, size=(h, w), mode="bilinear", align_corners=False)

            metrics["psnr_base"] = tensor_psnr(y_base, gt)
            metrics["ssim_base"] = tensor_ssim(y_base, gt)
            metrics["psnr_mix"] = tensor_psnr(y_mix, gt)
            metrics["ssim_mix"] = tensor_ssim(y_mix, gt)

        return x, y_base, y_mix, metrics

    for idx, (lq, gt_path) in enumerate(pairs):
        x, y0, y1, m = run_one(lq, gt_path)

        # stats
        if m["psnr_base"] is not None:
            base_psnr_list.append(m["psnr_base"])
            base_ssim_list.append(m["ssim_base"])
            mix_psnr_list.append(m["psnr_mix"])
            mix_ssim_list.append(m["ssim_mix"])

        # log
        print(f"[{idx:04d}] {m['name']}")
        print("[topk]", m["topk"])
        print("[dec ]", {
            "mode": m["mode"],
            "top1": m["top1_domain"],
            "local": m["local"],
            "global": m["global"],
            "reason": m["reason"],
        })

        # save images (refactored)
        if args.save_images:
            # Preserve original behavior when NOT using concat:
            #   - legacy: save base/mix singles
            # New behavior when concat=True:
            #   - default: only save concat triplet (LQ|BASE|MIX)
            #   - optional: --save_singles to also save base/mix singles
            meta = {
                "mode": m["mode"],
                "local": m["local"] or "",
                "global": m["global"] or "",
                "psnr_base": m["psnr_base"] if m["psnr_base"] is not None else "",
                "ssim_base": m["ssim_base"] if m["ssim_base"] is not None else "",
                "psnr_mix": m["psnr_mix"] if m["psnr_mix"] is not None else "",
                "ssim_mix": m["ssim_mix"] if m["ssim_mix"] is not None else "",
            }

            save_concat = bool(args.concat)
            # if concat not requested, keep legacy: save singles
            save_singles = bool(args.save_singles) if save_concat else True

            save_triplet(
                out_dir=args.output,
                stem=f"{idx:05d}",
                lq=x,
                base=y0,
                mix=y1,
                save_concat=save_concat,
                save_singles=save_singles,
                save_lq=bool(args.save_lq),
                annotate=bool(args.annotate_concat),
                meta=meta,
            )

        # CSV row
        if args.metrics_csv:
            with open(args.metrics_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    args.run_name,
                    m["name"],
                    m["mode"],
                    m["local"] if m["local"] else "",
                    m["global"] if m["global"] else "",
                    m["top1_domain"],
                    m["top1_w"],
                    m["top2_w"],
                    m["psnr_base"] if m["psnr_base"] is not None else "",
                    m["ssim_base"] if m["ssim_base"] is not None else "",
                    m["psnr_mix"] if m["psnr_mix"] is not None else "",
                    m["ssim_mix"] if m["ssim_mix"] is not None else "",
                    json.dumps(m["topk"], ensure_ascii=False),
                    m["reason"],
                    args.ramp_p0,
                    args.ramp_k,
                    args.ramp_eps,
                ])

    # summary
    if len(base_psnr_list) > 0:
        mean_base_psnr = float(np.mean(base_psnr_list))
        mean_base_ssim = float(np.mean(base_ssim_list))
        mean_mix_psnr = float(np.mean(mix_psnr_list))
        mean_mix_ssim = float(np.mean(mix_ssim_list))

        print("[SUMMARY]")
        print(f"  BASE: PSNR={mean_base_psnr:.4f}, SSIM={mean_base_ssim:.4f}")
        print(f"  MIX : PSNR={mean_mix_psnr:.4f}, SSIM={mean_mix_ssim:.4f}")

        if args.summary_csv:
            os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
            write_header = not os.path.isfile(args.summary_csv)
            with open(args.summary_csv, "a", newline="", encoding="utf-8") as f:
                fields = [
                    "run_name", "n_images",
                    "single_tau", "single_margin", "mix_topk",
                    "ramp_p0", "ramp_k", "ramp_eps",
                    "mean_psnr_base", "mean_ssim_base",
                    "mean_psnr_mix", "mean_ssim_mix",
                ]
                w = csv.DictWriter(f, fieldnames=fields)
                if write_header:
                    w.writeheader()
                w.writerow({
                    "run_name": args.run_name,
                    "n_images": len(pairs),
                    "single_tau": args.single_tau,
                    "single_margin": args.single_margin,
                    "mix_topk": args.mix_topk,
                    "ramp_p0": args.ramp_p0,
                    "ramp_k": args.ramp_k,
                    "ramp_eps": args.ramp_eps,
                    "mean_psnr_base": mean_base_psnr,
                    "mean_ssim_base": mean_base_ssim,
                    "mean_psnr_mix": mean_mix_psnr,
                    "mean_ssim_mix": mean_mix_ssim,
                })
            print(f"[CSV] summary appended to {args.summary_csv}")
    else:
        print("[SUMMARY] no GT provided, skip PSNR/SSIM summary.")


if __name__ == "__main__":
    main()

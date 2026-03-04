# -*- coding: utf-8 -*-
"""lora_adapters.pipeline.infer

Unified inference pipeline:
  retrieval -> decision -> mixing -> save/log/metrics

This file intentionally contains NO strategy-specific logic; all mixing strategies
are registered in lora_adapters.mixing.mixer.

Entry:
  from lora_adapters.pipeline.infer import run
  run(args)
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from lora_adapter.retrieval.embedder_clip import CLIPEmbedder
from lora_adapter.retrieval.prototype_bank import PrototypeBank
from lora_adapter.retrieval.router import Router
from lora_adapter.decision.decide import decide_single_or_mix
from lora_adapter.mixing.mixer import Mixer, load_all_domain_loras, MIXERS
from lora_adapter.pipeline.io import ensure_dir, read_image, save_tensor_image, save_triplet
from lora_adapter.pipeline.metrics import tensor_psnr, tensor_ssim
from lora_adapter.pipeline.logger import RunLogger


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _unwrap_ckpt(sd: Dict):
    if isinstance(sd, dict) and "params_ema" in sd:
        return sd["params_ema"]
    if isinstance(sd, dict) and "params" in sd:
        return sd["params"]
    if isinstance(sd, dict) and "state_dict" in sd:
        return sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        return sd["model"]
    return sd


def build_histoformer(weights: str, yaml_file: Optional[str] = None) -> torch.nn.Module:
    """Build Histoformer and load base checkpoint.

    Supports two common setups:
    - Basicsr-style YAML with `network_g` dict.
    - Direct instantiation of Histoformer with defaults.

    If yaml_file is None, we instantiate with default params.
    """
    # Try import from basicsr; fallback to local file name.
    
    from basicsr.models.archs.histoformer_arch import Histoformer


    net_args = {}
    if yaml_file:
        cfg = yaml.safe_load(open(yaml_file, "r", encoding="utf-8"))
        if isinstance(cfg, dict) and "network_g" in cfg:
            net_cfg = cfg["network_g"]
        else:
            net_cfg = cfg
        if isinstance(net_cfg, dict):
            # strip 'type'
            net_args = {k: v for k, v in net_cfg.items() if k not in ("type", "name")}

    net = Histoformer(**net_args)

    ckpt = torch.load(weights, map_location="cpu")
    ckpt = _unwrap_ckpt(ckpt)

    # sometimes keys are prefixed with 'net_g.'
    if isinstance(ckpt, dict):
        new_ckpt = {}
        for k, v in ckpt.items():
            if k.startswith("net_g."):
                new_ckpt[k[len("net_g."):]] = v
            else:
                new_ckpt[k] = v
        ckpt = new_ckpt

    missing, unexpected = net.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"[build_histoformer] missing keys: {len(missing)}")
    if unexpected:
        print(f"[build_histoformer] unexpected keys: {len(unexpected)}")

    return net


def parse_pairs(
    input_dir: Optional[str],
    pair_list: Optional[str],
    gt_root: Optional[str],
) -> List[Tuple[str, Optional[str]]]:
    """Return list of (lq_path, gt_path or None)."""
    pairs: List[Tuple[str, Optional[str]]] = []

    if pair_list:
        with open(pair_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    lq, gt = parts[0], parts[1]
                else:
                    lq = parts[0]
                    gt = parts[0] if gt_root else None
                if input_dir and not os.path.isabs(lq):
                    lq = str(Path(input_dir) / lq)
                if gt is not None:
                    # if user passes gt_root, interpret gt as relative path under gt_root
                    if gt_root and (not os.path.isabs(gt)):
                        gt = str(Path(gt_root) / gt)
                    # else: keep as-is (can be absolute or relative to cwd)
                pairs.append((lq, gt))
        return pairs

    if not input_dir:
        raise ValueError("Need either --pair_list or --input_dir")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    in_dir = Path(input_dir)
    for p in sorted(in_dir.rglob("*")):
        if p.suffix.lower() in exts:
            gt = str(Path(gt_root) / p.relative_to(in_dir)) if gt_root else None
            pairs.append((str(p), gt))
    return pairs


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def run(args):
    """Run inference pipeline.

    args: argparse Namespace (see cli/infer_mol.py)
    """
    set_seed(int(getattr(args, "seed", 42)))

    device = getattr(args, "device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    # logger
    rlog = RunLogger(out_dir, routes_jsonl=bool(args.routes_jsonl), metrics_csv=True, summary_csv=True)

    # build base net (CPU first; we move to GPU after injecting/loading LoRA)
    net = build_histoformer(args.base_ckpt, yaml_file=getattr(args, "yaml", None))

    # import here so `python -m cli.infer_mol -h` can work even if heavy deps
    # (basicsr/open_clip) are not available in the current environment.
    from lora_adapters.inject_lora import inject_lora

    # inject multi-domain LoRA slots
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    if not domains:
        raise ValueError("--domains is empty")

    inject_lora(
        net,
        rank=int(args.rank),
        alpha=float(args.alpha),
        domain_list=domains,
        enable_patch_lora=bool(int(getattr(args, "enable_patch_lora", 0))),
    )

    # retrieval components
    bank = PrototypeBank(domains=domains, loradb_root=args.loradb_root, embedder_tag=getattr(args, "embedder_tag", None))
    router = Router(bank, sim_metric=getattr(args, "sim_metric", "euclidean"), temperature=float(getattr(args, "temperature", 0.07)))

    embedder = CLIPEmbedder(
        device=getattr(args, "embed_device", "cpu"),
        model_name=getattr(args, "clip_model", "ViT-B-16"),
        pretrained=getattr(args, "clip_pretrained", "openai"),
    )

    # load all domain loras into injected slots (still on CPU)
    load_all_domain_loras(net, bank, strict=False)

    # move the whole net (base + all LoRA params) to device
    net = net.to(device).eval()

    mixer = Mixer(net, factor=int(getattr(args, "pad_factor", 8)))

    print(f"[Pipeline] mix modes available: {MIXERS.names()}")

    # prepare pairs
    pairs = parse_pairs(getattr(args, "input_dir", None), getattr(args, "pair_list", None), getattr(args, "gt_root", None))
    print(f"[Pipeline] total items: {len(pairs)}")

    img_dir = ensure_dir(out_dir / "images")
    trip_dir = ensure_dir(out_dir / "triplets")

    local_domains = [d.strip() for d in args.local_domains.split(",") if d.strip()]
    global_domains = [d.strip() for d in args.global_domains.split(",") if d.strip()]

    # mix args dictionary: only parameters used by mixer.py
    mix_args = {
        "normalize": bool(int(getattr(args, "mix_normalize", 1))),
        "ramp_mode": getattr(args, "ramp_mode", "sigmoid"),
        "use_retrieval_weight": bool(int(getattr(args, "use_retrieval_weight", 1))),

        # kselect params
        "k_topk": int(getattr(args, "k_topk", 0)),
        "k_score_mode": getattr(args, "k_score_mode", "topk_sum"),
        "k_ramp_mode": getattr(args, "k_ramp_mode", "sigmoid"),
        "k_alpha": float(getattr(args, "k_alpha", 1.0)),
        "k_beta": float(getattr(args, "k_beta", 1.0)),
        "use_gamma": int(getattr(args, "use_gamma", 1)),
        "gamma_mode": getattr(args, "gamma_mode", "global"),
        "gamma_clip": float(getattr(args, "gamma_clip", 0.0)),
        "score_eps": float(getattr(args, "score_eps", 1e-12)),
        "none_tau": float(getattr(args, "none_tau", 0.0)),
        "none_tau_alpha": float(getattr(args, "none_tau_alpha", 0.0)),
        "none_tau_mode": getattr(args, "none_tau_mode", "linear"),
        "act_score_mode": getattr(args, "act_score_mode", "mean_abs"),
        "act_every_n": int(getattr(args, "act_every_n", 4)),

        # dy-kselect params (used by mix_mode=act_kselect_dy / kselect_dy)
        "dy_topk_layers": int(getattr(args, "dy_topk_layers", 30)),
        "dy_tau": float(getattr(args, "dy_tau", 0.0)),
        "dy_score_mode": getattr(args, "dy_score_mode", "rms"),
        "dy_enable_both": int(getattr(args, "dy_enable_both", 1)),
        "dy_both_tau": float(getattr(args, "dy_both_tau", 1.0)),
        "dy_both_ratio": float(getattr(args, "dy_both_ratio", 0.6)),
        "dy_verbose": int(getattr(args, "dy_verbose", 1)),
        "dy_debug_topn": int(getattr(args, "dy_debug_topn", 30)),
        "dy_store_ranked": int(getattr(args, "dy_store_ranked", 0)),
    }

    # iterate
    for idx, (lq_path, gt_path) in enumerate(pairs):
        name = Path(lq_path).stem

        lq = read_image(lq_path, device=device)
        gt = read_image(gt_path, device=device) if gt_path else None

        with torch.no_grad():
            base = mixer.forward_base(lq)

            # retrieval
            emb = embedder.embed_image(lq_path)
            rr = router.route(emb, topk=max(int(args.topk), int(args.mix_topk)))

            # decision
            dec = decide_single_or_mix(
                rr,
                single_tau=float(args.single_tau),
                single_margin=float(args.single_margin),
                mix_topk=int(args.mix_topk),
                local_domains=local_domains,
                global_domains=global_domains,
            )

            # mixing (+ forward)
            out, mix_info = mixer.apply(args.mix_mode, lq, decision=dec, route=rr, **mix_args)
            out = out.clamp(0, 1)

        # metrics
        psnr = ssim = None
        if gt is not None:
            psnr = tensor_psnr(out, gt)
            ssim = tensor_ssim(out, gt)

        # save
        if bool(int(getattr(args, "save_out", 1))):
            save_tensor_image(out, img_dir / f"{name}_out.png")
        if bool(int(getattr(args, "save_base", 0))):
            save_tensor_image(base, img_dir / f"{name}_base.png")

        if bool(int(getattr(args, "save_triplet", 1))):
            save_triplet(lq, base, out, trip_dir / f"{name}.png")

        # log payload
        payload = {
            "route": rr.to_dict(),
            "decision": dec.to_dict(),
            "mix": mix_info,
        }
        if psnr is not None and ssim is not None:
            payload["metrics"] = {"psnr": float(psnr), "ssim": float(ssim)}

        rlog.log_item(name=name, psnr=psnr, ssim=ssim, payload=payload)

        if (idx + 1) % int(getattr(args, "log_every", 10)) == 0:
            msg = f"[{idx+1}/{len(pairs)}] {name}"
            if psnr is not None:
                msg += f" | PSNR={psnr:.2f} SSIM={ssim:.4f}"
            msg += f" | mode={dec.mode}:{args.mix_mode}"
            print(msg)

    rlog.finalize()
    print(f"[Done] outputs saved to: {out_dir}")

# -*- coding: utf-8 -*-
"""cli.infer_mol

Ultra-thin CLI:
  parse args -> lora_adapters.pipeline.infer.run(args)

Example:
  python -m cli.infer_mol \
    --input_dir data/haze_snow/val/lq \
    --pair_list data/txt_lists/val_list_haze_snow.txt \
    --gt_root data/haze_snow/val/gt \
    --output_dir results/kselect_demo \
    --base_ckpt pretrained_models/histoformer_base.pth \
    --yaml options/test/Histoformer.yml \
    --loradb_root weights/lora \
    --domains rain,rain2,rain3,rainy,snow,snow1,haze,haze1,haze2,low,low1 \
    --local_domains rain,rain2,rain3,rainy,snow,snow1 \
    --global_domains haze,haze1,haze2,low,low1 \
    --mix_mode kselect_hybrid \
    --topk 3 --mix_topk 10 --temperature 0.07
"""

from __future__ import annotations

import argparse

from lora_adapter.pipeline.infer import run
from lora_adapter.mixing.mixer import MIXERS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--input_dir", type=str, default=None, help="LQ folder (used if pair_list items are relative)")
    p.add_argument("--pair_list", type=str, default=None, help="txt list: 'lq gt' or 'name' per line")
    p.add_argument("--gt_root", type=str, default=None, help="GT root (used when pair_list has 1 column)")
    p.add_argument("--output_dir", type=str, required=True)

    # model
    p.add_argument("--base_ckpt", type=str, required=True)
    p.add_argument("--yaml", type=str, default=None, help="(optional) Basicsr yaml with network_g")

    # LoRA
    p.add_argument("--loradb_root", type=str, required=True)
    p.add_argument("--domains", type=str, required=True)
    p.add_argument("--local_domains", type=str, default="")
    p.add_argument("--global_domains", type=str, default="")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--enable_patch_lora", type=int, default=0)

    # retrieval
    p.add_argument("--embedder_tag", type=str, default=None, help="proto file suffix, e.g. clip_vitb16_openai")
    p.add_argument("--clip_model", type=str, default="ViT-B-16")
    p.add_argument("--clip_pretrained", type=str, default="openai")
    p.add_argument("--sim_metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--topk", type=int, default=3, help="how many domains to keep in RouteResult")

    # decision
    p.add_argument("--mix_topk", type=int, default=10, help="consider top-k candidates to pick local/global")
    p.add_argument("--single_tau", type=float, default=0.7)
    p.add_argument("--single_margin", type=float, default=0.2)

    # mixing
    p.add_argument("--mix_mode", type=str, default="kselect_hybrid", choices=MIXERS.names())
    p.add_argument("--mix_normalize", type=int, default=1, help="normalize weights when needed")
    p.add_argument("--use_retrieval_weight", type=int, default=1, help="use retrieval weights inside ramp/linear")
    p.add_argument("--ramp_mode", type=str, default="sigmoid", choices=["sigmoid", "linear"])

    # kselect
    p.add_argument("--k_topk", type=int, default=0, help="top-k entries in |ΔW| proxy. 0 -> auto")
    p.add_argument("--k_score_mode", type=str, default="topk_sum", choices=["topk_sum", "topk_ratio", "mean", "median", "fro"])
    p.add_argument("--k_ramp_mode", type=str, default="sigmoid", choices=["sigmoid", "linear", "none"])
    p.add_argument("--k_alpha", type=float, default=1.0)
    p.add_argument("--k_beta", type=float, default=1.0)
    p.add_argument("--use_gamma", type=int, default=1)
    p.add_argument("--gamma_mode", type=str, default="global", choices=["none", "global", "per_layer"])
    p.add_argument("--gamma_clip", type=float, default=0.0, help="clip gamma to [1/c, c] if >0")
    p.add_argument("--score_eps", type=float, default=1e-12)

    # optional 'none'
    p.add_argument("--none_tau", type=float, default=0.0)
    p.add_argument("--none_tau_alpha", type=float, default=0.0)
    p.add_argument("--none_tau_mode", type=str, default="linear", choices=["linear", "sigmoid"])

    # activation kselect
    p.add_argument("--act_score_mode", type=str, default="mean_abs", choices=["mean_abs", "rms", "l2"])
    p.add_argument("--act_every_n", type=int, default=4)

    # DY-KSelect (act_kselect_dy)
    p.add_argument("--dy_topk_layers", type=int, default=30, help="select top-K LoRA modules by DY score")
    p.add_argument("--dy_tau", type=float, default=0.0, help="DY score threshold (after TopK)")
    p.add_argument("--dy_score_mode", type=str, default="rms", choices=["mean_abs", "rms", "l2"], help="how to reduce Δy to a scalar")
    p.add_argument("--dy_enable_both", type=int, default=1, help="enable BOTH mode when two LoRAs are strong and close")
    p.add_argument("--dy_both_tau", type=float, default=1.0, help="BOTH requires score>=this")
    p.add_argument("--dy_both_ratio", type=float, default=0.6, help="BOTH requires min/max ratio>=this")
    p.add_argument("--dy_verbose", type=int, default=1, help="print per-image DY-KSelect logs")
    p.add_argument("--dy_debug_topn", type=int, default=30, help="print top-N scored modules")
    p.add_argument("--dy_store_ranked", type=int, default=0, help="store top-N ranked list into routes.jsonl")

    # runtime
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--embed_device", type=str, default="cpu", help="CLIP embedding device")
    p.add_argument("--pad_factor", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    # outputs
    p.add_argument("--save_out", type=int, default=1)
    p.add_argument("--save_base", type=int, default=0)
    p.add_argument("--save_triplet", type=int, default=1)
    p.add_argument("--routes_jsonl", type=int, default=1)
    p.add_argument("--log_every", type=int, default=10)

    return p


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()

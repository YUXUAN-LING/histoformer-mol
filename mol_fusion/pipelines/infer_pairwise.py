from __future__ import annotations

import argparse

import torch

from mol_fusion.executors.pairwise_executor import PairwiseExecutor
from mol_fusion.fusion.registry import build_fusion_policy
from mol_fusion.pipelines.common import build_model, forward_padded, load_lora_bank, maybe_metric
from mol_fusion.utils.io_utils import list_images, parse_pair_list, write_csv, write_json
from mol_fusion.utils.image_utils import load_image_tensor


def main():
    ap = argparse.ArgumentParser("pairwise-study")
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--yaml", default=None)
    ap.add_argument("--lora_root", required=True)
    ap.add_argument("--domains", required=True)
    ap.add_argument("--val_data_dir", required=True)
    ap.add_argument("--val_list", default=None)
    ap.add_argument("--dom1", required=True)
    ap.add_argument("--dom2", required=True)
    ap.add_argument("--fusion_policy", default="topk_softmix")
    ap.add_argument("--score_type", default="delta")
    ap.add_argument("--shared_metric", default="min")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--nonkey_mode", default="half")
    ap.add_argument("--weight_rule", default="softmax")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    domains = [d for d in args.domains.split(",") if d]
    net = build_model(args.base_ckpt, args.yaml, args.rank, args.alpha, domains, args.device)
    load_lora_bank(net, domains, args.lora_root)

    policy = build_fusion_policy(
        args.fusion_policy,
        score_type=args.score_type,
        shared_metric=args.shared_metric,
        topk=args.topk,
        nonkey_mode=args.nonkey_mode,
        weight_rule=args.weight_rule,
        temp=args.temp,
    )
    executor = PairwiseExecutor(net, all_domains=domains)

    items = parse_pair_list(args.val_list) if args.val_list else [(p, None) for p in list_images(args.val_data_dir)]
    metrics = []
    first_weights = None
    with torch.inference_mode():
        for img, gt in items:
            x = load_image_tensor(f"{args.val_data_dir}/{img}" if args.val_list else img).to(args.device)
            lw = policy.build_layer_weights(model=net, dom1=args.dom1, dom2=args.dom2, context={})
            executor.apply_layer_weights(args.dom1, args.dom2, lw)
            y = forward_padded(net, x)
            psnr = maybe_metric(y, f"{args.val_data_dir}/{gt}" if (gt and args.val_list) else None)
            metrics.append({"image": img, "psnr": psnr if psnr is not None else ""})
            if first_weights is None:
                first_weights = lw

    write_csv(f"{args.out_dir}/metrics.csv", metrics)
    write_json(f"{args.out_dir}/layer_weights_example.json", first_weights or {})


if __name__ == "__main__":
    main()

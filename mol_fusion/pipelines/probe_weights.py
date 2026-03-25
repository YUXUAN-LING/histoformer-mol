from __future__ import annotations

import argparse

from mol_fusion.fusion.registry import build_fusion_policy
from mol_fusion.pipelines.common import build_model, load_lora_bank
from mol_fusion.utils.io_utils import write_csv


def main():
    ap = argparse.ArgumentParser("probe-weights (reuses real policy)")
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--yaml", default=None)
    ap.add_argument("--lora_root", required=True)
    ap.add_argument("--domains", required=True)
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
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    domains = [d for d in args.domains.split(",") if d]
    net = build_model(args.base_ckpt, args.yaml, args.rank, args.alpha, domains, args.device)
    load_lora_bank(net, domains, args.lora_root)

    policy = build_fusion_policy(args.fusion_policy, score_type=args.score_type, shared_metric=args.shared_metric, topk=args.topk, nonkey_mode=args.nonkey_mode, weight_rule=args.weight_rule, temp=args.temp)
    lw = policy.build_layer_weights(model=net, dom1=args.dom1, dom2=args.dom2, context={})
    rows = [{"layer": n, args.dom1: v[args.dom1], args.dom2: v[args.dom2]} for n, v in lw.items()]
    write_csv(args.out_csv, rows)


if __name__ == "__main__":
    main()

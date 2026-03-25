from __future__ import annotations

import argparse

from mol_fusion.fusion.registry import build_score
from mol_fusion.pipelines.common import build_model, load_lora_bank
from mol_fusion.utils.io_utils import write_csv


def main():
    ap = argparse.ArgumentParser("probe-scores")
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--yaml", default=None)
    ap.add_argument("--lora_root", required=True)
    ap.add_argument("--domains", required=True)
    ap.add_argument("--dom1", required=True)
    ap.add_argument("--dom2", required=True)
    ap.add_argument("--score_type", default="delta")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    domains = [d for d in args.domains.split(",") if d]
    net = build_model(args.base_ckpt, args.yaml, args.rank, args.alpha, domains, args.device)
    load_lora_bank(net, domains, args.lora_root)
    score_fn = build_score(args.score_type)
    sc = score_fn.compute(net, args.dom1, args.dom2)
    rows = [{"layer": n, args.dom1: v[args.dom1], args.dom2: v[args.dom2]} for n, v in sc.items()]
    write_csv(args.out_csv, rows)


if __name__ == "__main__":
    main()

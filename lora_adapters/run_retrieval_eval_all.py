# lora_adapters/run_retrieval_eval_all.py
# -*- coding: utf-8 -*-
import csv
import json
import argparse
from pathlib import Path
from typing import List

from lora_adapters.eval_retrieval import (
    set_seed,
    derive_embedder_tag,
    RetrievalConfig,
    SetSpec,
    evaluate_one_set,
)


def load_eval_sets(json_path: Path) -> List[SetSpec]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("eval_sets.json 必须是一个 list")

    out = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"eval_sets.json[{i}] 不是 dict")

        name = item.get("name") or item.get("true_domain") or f"set_{i:02d}"
        spec = SetSpec(
            name=name,
            true_domain=item.get("true_domain", ""),
            true_domains=item.get("true_domains", None),   # <- 新增
            input_dir=item.get("input_dir", None),
            pair_list=item.get("pair_list", None),
            data_root=item.get("data_root", "."),
        )
        spec.validate()
        out.append(spec)
    return out


def main():
    ap = argparse.ArgumentParser("run_retrieval_eval_all")

    ap.add_argument("--eval_sets", type=str, required=True, help="eval_sets.json")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--cache_emb_dir", type=str, default=None)

    ap.add_argument("--loradb_root", type=str, required=True)
    ap.add_argument("--domains", type=str, required=True)

    ap.add_argument("--embedder", choices=["clip", "dino_v2", "fft", "fft_enhanced"], default="clip")
    ap.add_argument("--embedder_tag", type=str, default=None)
    ap.add_argument("--sim_metric", choices=["cosine", "euclidean"], default="cosine")
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--topk", type=int, default=3)

    ap.add_argument("--proto_reduce", choices=["sum", "mean", "max", "topm_sum"], default="sum")
    ap.add_argument("--proto_topm", type=int, default=2)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--euclidean_mode", choices=["inv", "neg"], default="inv")
    ap.add_argument("--force_avg", action="store_true")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")

    ap.add_argument("--clip_model", type=str, default="ViT-B-16")
    ap.add_argument("--clip_pretrained", type=str, default="openai")
    ap.add_argument("--dino_ckpt", type=str, default=None)
    ap.add_argument("--fft_resize", type=int, default=256)
    ap.add_argument("--fft_center_crop", type=int, default=0)
    ap.add_argument("--fft_out_size", type=int, default=32)
    ap.add_argument("--fft_clean_proto", type=str, default=None)

    args = ap.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    embedder_tag = derive_embedder_tag(args.embedder, args.clip_model, args.embedder_tag)

    cfg = RetrievalConfig(
        loradb_root=args.loradb_root,
        domains=domains,
        embedder=args.embedder,
        embedder_tag=embedder_tag,
        sim_metric=args.sim_metric,
        temperature=args.temperature,
        topk=args.topk,
        proto_reduce=args.proto_reduce,
        proto_topm=args.proto_topm,
        normalize=bool(args.normalize),
        euclidean_mode=args.euclidean_mode,
        force_avg=bool(args.force_avg),
        seed=args.seed,
        deterministic=bool(args.deterministic),
        device=args.device,

        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        dino_ckpt=args.dino_ckpt,
        fft_resize=args.fft_resize,
        fft_center_crop=args.fft_center_crop,
        fft_out_size=args.fft_out_size,
        fft_clean_proto=args.fft_clean_proto,
    )

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_emb_dir) if args.cache_emb_dir else None

    sets = load_eval_sets(Path(args.eval_sets))

    summary_all = []
    for spec in sets:
        print(f"\n=== [RUN SET] {spec.name} | true_domain={spec.true_domain} | true_domains={spec.true_domains} ===")
        set_out = out_root / spec.name
        s = evaluate_one_set(cfg, spec, out_dir=set_out, cache_emb_dir=cache_dir)
        summary_all.append(s)

    if summary_all:
        fp = out_root / "summary_all.csv"
        keys = list(summary_all[0].keys())
        with fp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in summary_all:
                w.writerow(r)
        print(f"\n[SAVE] {fp} | rows={len(summary_all)}")

    print("\n[DONE] all sets evaluated.")


if __name__ == "__main__":
    main()

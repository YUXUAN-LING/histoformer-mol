from __future__ import annotations

import argparse
from PIL import Image

from mol_fusion.retrieval.embedders.clip_embedder import ClipImageEmbedder
from mol_fusion.retrieval.prototype_store import PrototypeStore
from mol_fusion.retrieval.retrieve import Retriever
from mol_fusion.routing.decision import route_decision
from mol_fusion.utils.io_utils import list_images, write_csv, write_jsonl


def main():
    ap = argparse.ArgumentParser("retrieval-only")
    ap.add_argument("--val_data_dir", required=True)
    ap.add_argument("--domains", required=True)
    ap.add_argument("--proto_root", required=True)
    ap.add_argument("--proto_tag", default=None)
    ap.add_argument("--embedder", default="clip")
    ap.add_argument("--clip_model", default="ViT-B-16")
    ap.add_argument("--clip_pretrained", default=None)
    ap.add_argument("--sim_metric", default="cosine")
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--single_tau", type=float, default=0.72)
    ap.add_argument("--margin_tau", type=float, default=0.10)
    ap.add_argument("--mix_pair_policy", default="top2")
    ap.add_argument("--local_domains", default="")
    ap.add_argument("--global_domains", default="")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    domains = [s.strip() for s in args.domains.split(",") if s.strip()]
    embedder = ClipImageEmbedder(model_name=args.clip_model, pretrained=args.clip_pretrained)
    store = PrototypeStore(args.proto_root, domains=domains, proto_tag=args.proto_tag)
    retriever = Retriever(embedder=embedder, store=store, sim_metric=args.sim_metric, temperature=args.temperature)

    rows, jsonl = [], []
    for p in list_images(args.val_data_dir):
        out = retriever.retrieve_image(Image.open(p).convert("RGB"))
        route = route_decision(
            out.weights,
            single_tau=args.single_tau,
            margin_tau=args.margin_tau,
            pair_policy=args.mix_pair_policy,
            local_domains=[d for d in args.local_domains.split(",") if d],
            global_domains=[d for d in args.global_domains.split(",") if d],
        )
        top = out.topk(args.topk)
        rows.append({"image": p, "top1": top[0][0], "top1_w": top[0][1], "route": route["route"], "dom1": route["dom1"], "dom2": route["dom2"]})
        jsonl.append({"image": p, "scores": out.scores, "weights": out.weights, "topk": top, "route": route})

    write_csv(f"{args.out_dir}/retrieval_summary.csv", rows)
    write_jsonl(f"{args.out_dir}/retrieval.jsonl", jsonl)


if __name__ == "__main__":
    main()

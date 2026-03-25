from __future__ import annotations

import argparse
from PIL import Image
import torch

from mol_fusion.executors.pairwise_executor import PairwiseExecutor
from mol_fusion.fusion.registry import build_fusion_policy
from mol_fusion.pipelines.common import build_model, forward_padded, load_lora_bank
from mol_fusion.retrieval.embedders.clip_embedder import ClipImageEmbedder
from mol_fusion.retrieval.prototype_store import PrototypeStore
from mol_fusion.retrieval.retrieve import Retriever
from mol_fusion.routing.decision import route_decision
from mol_fusion.utils.image_utils import load_image_tensor
from mol_fusion.utils.io_utils import list_images, write_jsonl


def main():
    ap = argparse.ArgumentParser("full-pipeline")
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--yaml", default=None)
    ap.add_argument("--lora_root", required=True)
    ap.add_argument("--val_data_dir", required=True)
    ap.add_argument("--domains", required=True)
    ap.add_argument("--proto_root", required=True)
    ap.add_argument("--proto_tag", default=None)
    ap.add_argument("--mix_pair_policy", default="local_global")
    ap.add_argument("--single_tau", type=float, default=0.72)
    ap.add_argument("--margin_tau", type=float, default=0.10)
    ap.add_argument("--local_domains", default="")
    ap.add_argument("--global_domains", default="")
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
    executor = PairwiseExecutor(net, all_domains=domains)

    embedder = ClipImageEmbedder()
    store = PrototypeStore(args.proto_root, domains, proto_tag=args.proto_tag)
    retriever = Retriever(embedder, store)
    policy = build_fusion_policy(args.fusion_policy, score_type=args.score_type, shared_metric=args.shared_metric, topk=args.topk, nonkey_mode=args.nonkey_mode, weight_rule=args.weight_rule, temp=args.temp)

    rows = []
    with torch.inference_mode():
        for p in list_images(args.val_data_dir):
            ret = retriever.retrieve_image(Image.open(p).convert("RGB"))
            route = route_decision(ret.weights, args.single_tau, args.margin_tau, pair_policy=args.mix_pair_policy, local_domains=args.local_domains.split(","), global_domains=args.global_domains.split(","))
            if route["route"] == "single":
                executor.apply_single(route["dom1"])
                layer_weights = None
            else:
                layer_weights = policy.build_layer_weights(model=net, dom1=route["dom1"], dom2=route["dom2"], context={})
                executor.apply_layer_weights(route["dom1"], route["dom2"], layer_weights)
            y = forward_padded(net, load_image_tensor(p).to(args.device))
            rows.append({"image": p, "route": route, "topk": ret.topk(5), "layer_weights": layer_weights, "shape": list(y.shape)})

    write_jsonl(f"{args.out_dir}/route.jsonl", rows)


if __name__ == "__main__":
    main()

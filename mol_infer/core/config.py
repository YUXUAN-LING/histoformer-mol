# mol_infer/core/config.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from mol_infer.core.types import (
    RunConfig,
    IOConfig,
    ModelConfig,
    RetrievalConfig,
    RoutingConfig,
    FusionConfig,
    RuntimeConfig,
)


def _get(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


class RunConfigFactory:
    @staticmethod
    def from_args(args: Any) -> RunConfig:
        """
        Build RunConfig from argparse args.

        IMPORTANT DESIGN RULE:
          - IOConfig: only I/O + saving + logging paths
          - RuntimeConfig: run_name/device/seed/deterministic/enable flags/debug flags
          - ModelConfig: base/yaml/loradb/domains/rank/alpha/patch
          - RetrievalConfig: embedder/clip/sim/topk/temp/tag
          - RoutingConfig: local/global/mix_topk/single_tau/single_margin
          - FusionConfig: fusion name + kselect/none params
        """

        # ---------------- IO ----------------
        io = IOConfig(
            input=_get(args, "input"),
            pair_list=_get(args, "pair_list", None),
            gt_root=_get(args, "gt_root", None),
            output_dir=_get(args, "output_dir"),
            save_images=bool(_get(args, "save_images", True)),
            concat=bool(_get(args, "concat", True)),
            save_singles=bool(_get(args, "save_singles", False)),
            save_lq=bool(_get(args, "save_lq", False)),
            annotate=bool(_get(args, "annotate", False)),
            metrics_csv=_get(args, "metrics_csv", None),
            routing_jsonl=_get(args, "routing_jsonl", None),
            summary_csv=_get(args, "summary_csv", None),
        )

        # ---------------- Model / LoRA ----------------
        model = ModelConfig(
            base_ckpt=_get(args, "base_ckpt"),
            yaml=_get(args, "yaml", None),
            loradb_root=_get(args, "loradb_root"),
            domains=_get(args, "domains"),
            rank=int(_get(args, "rank", 16)),
            alpha=float(_get(args, "alpha", 16)),
            enable_patch_lora=bool(_get(args, "enable_patch_lora", False)),
        )

        # ---------------- Retrieval ----------------
        retrieval = RetrievalConfig(
            embedder=_get(args, "embedder", "clip"),
            clip_model=_get(args, "clip_model", "ViT-B-16"),
            clip_pretrained=_get(args, "clip_pretrained", None),
            embedder_tag=_get(args, "embedder_tag", None),
            sim_metric=_get(args, "sim_metric", "cosine"),
            temperature=float(_get(args, "temperature", 0.07)),
            topk=int(_get(args, "topk", 5)),
            loradb_root=_get(args, "loradb_root"),
            domains=_get(args, "domains"),
        )

        # ---------------- Routing ----------------
        routing = RoutingConfig(
            local_domains=_get(args, "local_domains"),
            global_domains=_get(args, "global_domains"),
            mix_topk=int(_get(args, "mix_topk", 5)),
            single_tau=float(_get(args, "single_tau", 0.72)),
            single_margin=float(_get(args, "single_margin", 0.10)),
            norm_topk_domains=int(_get(args, "norm_topk_domains", 0)),
        )

        # ---------------- Fusion ----------------
        fusion = FusionConfig(
            name=_get(args, "fusion", "none"),
            k_topk=int(_get(args, "k_topk", 0)),
            k_score_mode=_get(args, "k_score_mode", "topk_ratio"),
            k_alpha=float(_get(args, "k_alpha", 1.0)),
            k_beta=float(_get(args, "k_beta", 1.0)),
            k_ramp_mode=_get(args, "k_ramp_mode", "sigmoid"),
            use_gamma=bool(_get(args, "use_gamma", False)),
            gamma_mode=_get(args, "gamma_mode", "per_layer"),
            gamma_clip=float(_get(args, "gamma_clip", 0.0)),
            none_mode=_get(args, "none_mode", "const"),
            none_alpha=float(_get(args, "none_alpha", 0.0)),
            none_beta=float(_get(args, "none_beta", 0.15)),
            none_metric=_get(args, "none_metric", "abs"),
        )

        # ---------------- Runtime ----------------
        runtime = RuntimeConfig(
            device=_get(args, "device", "cuda"),
            seed=int(_get(args, "seed", 0)),
            deterministic=bool(_get(args, "deterministic", False)),
            run_name=_get(args, "run_name", ""),
            enable_lora=bool(_get(args, "enable_lora", False)),
            enable_fusion=bool(_get(args, "enable_fusion", False)),
            print_full_scores=bool(_get(args, "print_full_scores", False)),
            debug=bool(_get(args, "debug", False)),
        )

        return RunConfig(
            io=io,
            model=model,
            retrieval=retrieval,
            routing=routing,
            fusion=fusion,
            runtime=runtime,
        )

    @staticmethod
    def to_dict(cfg: RunConfig) -> dict:
        return asdict(cfg)

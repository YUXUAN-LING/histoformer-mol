# mol_infer/scripts/infer.py
from __future__ import annotations

import argparse

from mol_infer.core.config import RunConfigFactory
from mol_infer.core.runner import Runner as InferenceRunner


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("mol_infer: base + retrieval + routing + (optional) fusion")

    # ----------------- io -----------------
    ap.add_argument("--input", type=str, required=True, help="folder of images, or a single image path")
    ap.add_argument("--pair_list", type=str, default=None, help="txt: lq [gt]")
    ap.add_argument("--gt_root", type=str, default=None)
    ap.add_argument("--output_dir", type=str, required=True)

    # save / logging
    ap.add_argument("--save_images", action="store_true")
    ap.add_argument("--concat", action="store_true")
    ap.add_argument("--save_singles", action="store_true")
    ap.add_argument("--save_lq", action="store_true")
    ap.add_argument("--metrics_csv", type=str, default=None)
    ap.add_argument("--routing_jsonl", type=str, default=None)
    ap.add_argument("--summary_csv", type=str, default=None)

    # ----------------- model/lora -----------------
    ap.add_argument("--base_ckpt", type=str, required=True)
    ap.add_argument("--yaml", type=str, default=None)
    ap.add_argument("--loradb_root", type=str, required=True)
    ap.add_argument("--domains", type=str, required=True, help="comma-separated domain list (order matters)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--enable_patch_lora", action="store_true")

    # switches
    ap.add_argument("--enable_lora", action="store_true")
    ap.add_argument("--enable_fusion", action="store_true")

    # ----------------- retrieval -----------------
    ap.add_argument("--embedder", type=str, default="clip", choices=["clip"])
    ap.add_argument("--clip_model", type=str, default="ViT-B-16")
    ap.add_argument("--clip_pretrained", type=str, default="openai",
                    help="can be 'openai' or local path to open_clip_pytorch_model.bin")
    ap.add_argument("--embedder_tag", type=str, default=None, help="must match avg_embedding*.npy if you use tags")
    ap.add_argument("--sim_metric", type=str, default="cosine", choices=["cosine", "euclidean", "l2"])
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--print_full_scores", action="store_true")

    # ----------------- routing -----------------
    ap.add_argument("--local_domains", type=str, required=True)
    ap.add_argument("--global_domains", type=str, required=True)
    ap.add_argument("--mix_topk", type=int, default=5)
    ap.add_argument("--single_tau", type=float, default=0.72)
    ap.add_argument("--single_margin", type=float, default=0.10)

    # ----------------- runtime -----------------
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--run_name", type=str, default="")

    # ================= fusion args (NEW) =================
    ap.add_argument("--fusion", type=str, default="none",
                    choices=["none", "kselect_none", "kselect_static", "kselect_activation"])

    # common kselect knobs
    ap.add_argument("--kselect_mode", type=str, default="static", choices=["static", "activation", "hybrid"])
    ap.add_argument("--k_score_mode", type=str, default="topk_ratio",
                    choices=["topk_sum", "mean", "median", "fro", "topk_ratio"])
    ap.add_argument("--k_topk", type=int, default=0)
    ap.add_argument("--k_alpha", type=float, default=1.0)
    ap.add_argument("--k_beta", type=float, default=1.0)
    ap.add_argument("--k_ramp_mode", type=str, default="sigmoid", choices=["linear", "sigmoid"])
    ap.add_argument("--use_gamma", action="store_true")
    ap.add_argument("--gamma_mode", type=str, default="per_layer", choices=["none", "global", "per_layer"])
    ap.add_argument("--gamma_clip", type=float, default=0.0)
    ap.add_argument("--score_eps", type=float, default=1e-12)

    # activation knobs
    ap.add_argument("--act_score_mode", type=str, default="mean_abs", choices=["mean_abs", "l2", "rms"])
    ap.add_argument("--act_every_n", type=int, default=1)

    # none-threshold knobs (for kselect_none)
    ap.add_argument("--none_mode", type=str, default="const", choices=["const", "linear", "sigmoid"])
    ap.add_argument("--none_beta", type=float, default=0.15)
    ap.add_argument("--none_alpha", type=float, default=0.0)
    ap.add_argument("--none_metric", type=str, default="abs", choices=["abs", "ratio"])
    ap.add_argument("--none_ratio_eps", type=float, default=1e-12)

    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    cfg = RunConfigFactory.from_args(args)
    runner = InferenceRunner(cfg)
    records, summary = runner.run()

    # print brief summary
    print(f"[DONE] n={summary.get('n', 0)} "
          f"base_psnr={summary.get('mean_psnr_base')} auto_psnr={summary.get('mean_psnr_auto')} "
          f"fusion={summary.get('fusion')}")


if __name__ == "__main__":
    main()

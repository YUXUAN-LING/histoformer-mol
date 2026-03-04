# scripts/plot_retrieval_results.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, argparse
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _maybe_json_list(s):
    if pd.isna(s) or s == "":
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _safe_float(x, default=np.nan):
    try:
        if x == "" or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="包含 per_image.csv/heatmap_*.csv 的目录")
    ap.add_argument("--out_dir", default=None, help="输出 plots 目录，默认 <in_dir>/plots")
    ap.add_argument("--title", default=None, help="图标题前缀（可选）")
    ap.add_argument("--topn_domains", type=int, default=20, help="freq/weight 图显示前 N 个域")
    ap.add_argument("--rank_cap", type=int, default=10, help="rank 直方图截断到前 N（更直观）")
    args = ap.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir or os.path.join(in_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    title_prefix = (args.title + " | ") if args.title else ""

    per_csv = os.path.join(in_dir, "per_image.csv")
    df = _read_csv(per_csv)

    # numeric columns (some may be "")
    for c in ["hit_top1", "hit_topk", "rank_true", "weight_true_in_topk",
              "hit_any", "hit_all", "rank_best_gt", "sum_weight_gt", "max_weight_gt",
              "entropy_topk", "margin_weight_12", "margin_score_12"]:
        if c in df.columns:
            df[c] = df[c].apply(_safe_float)

    # ---------- 1) accuracy bar ----------
    metrics = {}
    if "hit_top1" in df: metrics["top1_acc"] = np.nanmean(df["hit_top1"])
    if "hit_topk" in df: metrics["topk_acc"] = np.nanmean(df["hit_topk"])
    if "hit_any" in df and not np.all(np.isnan(df["hit_any"])): metrics["hit_any_acc"] = np.nanmean(df["hit_any"])
    if "hit_all" in df and not np.all(np.isnan(df["hit_all"])): metrics["hit_all_acc"] = np.nanmean(df["hit_all"])

    plt.figure(figsize=(7, 4))
    names = list(metrics.keys())
    vals = [metrics[k] for k in names]
    plt.bar(names, vals)
    plt.ylim(0, 1.0)
    plt.title(f"{title_prefix}Accuracy")
    plt.ylabel("rate")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_bar.png"), dpi=200)
    plt.close()

    # ---------- 2) rank hist ----------
    plt.figure(figsize=(7, 4))
    if "rank_true" in df and not np.all(np.isnan(df["rank_true"])):
        r = df["rank_true"].copy()
        r = r[np.isfinite(r)]
        r = np.minimum(r, args.rank_cap)
        plt.hist(r, bins=np.arange(1, args.rank_cap + 2), alpha=0.6, label="rank_true")
    if "rank_best_gt" in df and not np.all(np.isnan(df["rank_best_gt"])):
        r2 = df["rank_best_gt"].copy()
        r2 = r2[np.isfinite(r2)]
        r2 = np.minimum(r2, args.rank_cap)
        plt.hist(r2, bins=np.arange(1, args.rank_cap + 2), alpha=0.6, label="rank_best_gt")
    plt.title(f"{title_prefix}Rank histogram (cap={args.rank_cap})")
    plt.xlabel("rank (capped)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rank_hist.png"), dpi=200)
    plt.close()

    # ---------- 3) entropy hist ----------
    if "entropy_topk" in df and not np.all(np.isnan(df["entropy_topk"])):
        plt.figure(figsize=(7, 4))
        e = df["entropy_topk"]
        e = e[np.isfinite(e)]
        plt.hist(e, bins=30)
        plt.title(f"{title_prefix}Entropy(topk) histogram")
        plt.xlabel("entropy_topk")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "entropy_hist.png"), dpi=200)
        plt.close()

    # ---------- 4) sum_weight_gt vs entropy scatter (split by hit_all) ----------
    if ("sum_weight_gt" in df and "entropy_topk" in df and "hit_all" in df
        and not np.all(np.isnan(df["sum_weight_gt"])) and not np.all(np.isnan(df["entropy_topk"]))):
        ok = df[np.isfinite(df["sum_weight_gt"]) & np.isfinite(df["entropy_topk"])]
        plt.figure(figsize=(7, 5))
        hit = ok[ok["hit_all"] == 1]
        miss = ok[ok["hit_all"] == 0]
        if len(miss) > 0:
            plt.scatter(miss["entropy_topk"], miss["sum_weight_gt"], s=12, label="hit_all=0")
        if len(hit) > 0:
            plt.scatter(hit["entropy_topk"], hit["sum_weight_gt"], s=12, label="hit_all=1")
        plt.title(f"{title_prefix}sum_weight_gt vs entropy_topk")
        plt.xlabel("entropy_topk (higher = more uncertain)")
        plt.ylabel("sum_weight_gt (higher = more confident on GT)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "sumw_scatter.png"), dpi=200)
        plt.close()

    # ---------- 5) domain freq / mean weight bars ----------
    freq_csv = os.path.join(in_dir, "heatmap_freq_in_topk.csv")
    w_csv = os.path.join(in_dir, "heatmap_mean_weight.csv")

    if os.path.isfile(freq_csv):
        dff = _read_csv(freq_csv).sort_values("freq_in_topk", ascending=False).head(args.topn_domains)
        plt.figure(figsize=(10, 4))
        plt.bar(dff["domain"], dff["freq_in_topk"])
        plt.title(f"{title_prefix}freq_in_topk (Top {args.topn_domains})")
        plt.ylabel("freq_in_topk")
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "freq_in_topk.png"), dpi=200)
        plt.close()

    if os.path.isfile(w_csv):
        dfw = _read_csv(w_csv).sort_values("mean_weight", ascending=False).head(args.topn_domains)
        plt.figure(figsize=(10, 4))
        plt.bar(dfw["domain"], dfw["mean_weight"])
        plt.title(f"{title_prefix}mean_weight (Top {args.topn_domains})")
        plt.ylabel("mean_weight")
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mean_weight.png"), dpi=200)
        plt.close()

    # dump numeric summary for convenience
    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, ensure_ascii=False, indent=2)

    print(f"[OK] plots saved to: {out_dir}")


if __name__ == "__main__":
    main()

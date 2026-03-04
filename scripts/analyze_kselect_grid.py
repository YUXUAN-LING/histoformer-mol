# tools/analyze_kselect_grid.py
# -*- coding: utf-8 -*-
"""
Aggregate & analyze kselect grid results from many *_metrics.csv files.

Expected metrics_csv columns (from your current infer_data_kselect logging):
run_name, name,
mode, local, global,
top1_dom, top1_w, top2_w, margin,
psnr_base, ssim_base, psnr_auto, ssim_auto,
topk_str, reason,
k_topk, k_alpha, k_beta, use_gamma, k_ramp_mode,
k_gamma, n_layers, n_local, n_global, local_ratio, global_ratio,
sc_mean, ss_mean, ss_scaled_mean, S_first, S_last

If some columns are missing, the script will degrade gracefully.
"""

import os
import glob
import argparse
from collections import Counter, defaultdict

import pandas as pd


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def load_one_metrics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize common column names (if any small differences appear)
    rename_map = {
        "psnr_mix": "psnr_auto",
        "ssim_mix": "ssim_auto",
        "mix": "auto",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # If run_name missing, infer from filename
    if "run_name" not in df.columns:
        run = os.path.basename(path).replace("_metrics.csv", "")
        df["run_name"] = run

    df["__src"] = path
    return df


def summarize_run(df: pd.DataFrame) -> dict:
    # Basic
    run = str(df["run_name"].iloc[0]) if "run_name" in df.columns else "unknown"

    # Metrics (mean over images that have GT)
    def mean_col(col):
        if col not in df.columns:
            return None
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            return None
        return float(s.mean())

    psnr_b = mean_col("psnr_base")
    ssim_b = mean_col("ssim_base")
    psnr_a = mean_col("psnr_auto")
    ssim_a = mean_col("ssim_auto")

    dpsnr = (psnr_a - psnr_b) if (psnr_a is not None and psnr_b is not None) else None
    dssim = (ssim_a - ssim_b) if (ssim_a is not None and ssim_b is not None) else None

    # Router stats
    mode_counts = None
    mix_ratio = None
    single_ratio = None
    if "mode" in df.columns:
        mode_counts = df["mode"].astype(str).value_counts().to_dict()
        total = sum(mode_counts.values()) if mode_counts else 0
        if total > 0:
            mix_ratio = mode_counts.get("mix", 0) / total
            single_ratio = mode_counts.get("single", 0) / total

    # Mix choices
    local_counter = Counter()
    global_counter = Counter()
    if "mode" in df.columns and "local" in df.columns and "global" in df.columns:
        mix_df = df[df["mode"].astype(str) == "mix"]
        for x in mix_df["local"].astype(str).tolist():
            if x and x != "nan":
                local_counter[x] += 1
        for x in mix_df["global"].astype(str).tolist():
            if x and x != "nan":
                global_counter[x] += 1

    # KSelect hyperparams (assume constant within run)
    def first_val(col):
        if col not in df.columns:
            return None
        v = df[col].iloc[0]
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    row = {
        "run_name": run,
        "n_images": int(len(df)),
        "psnr_base_mean": psnr_b,
        "ssim_base_mean": ssim_b,
        "psnr_auto_mean": psnr_a,
        "ssim_auto_mean": ssim_a,
        "dpsnr_mean": dpsnr,
        "dssim_mean": dssim,
        "single_ratio": single_ratio,
        "mix_ratio": mix_ratio,
        "mode_counts": str(mode_counts) if mode_counts is not None else None,
        "top_local_in_mix": str(local_counter.most_common(5)) if local_counter else None,
        "top_global_in_mix": str(global_counter.most_common(5)) if global_counter else None,
        "k_topk": first_val("k_topk"),
        "k_alpha": first_val("k_alpha"),
        "k_beta": first_val("k_beta"),
        "use_gamma": first_val("use_gamma"),
        "k_ramp_mode": first_val("k_ramp_mode"),
        "src_metrics_csv": first_val("__src"),
    }

    # Optional layerwise stats if present
    for col in ["k_gamma", "n_layers", "n_local", "n_global", "local_ratio", "global_ratio",
                "sc_mean", "ss_mean", "ss_scaled_mean", "S_first", "S_last"]:
        if col in df.columns:
            row[col] = safe_float(df[col].iloc[0]) if df[col].notna().any() else None

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Directory containing many *_metrics.csv from grid runs")
    ap.add_argument("--pattern", type=str, default="*_metrics.csv",
                    help="Glob pattern under root (default: *_metrics.csv)")
    ap.add_argument("--out_csv", type=str, default=None,
                    help="Output aggregated runs csv (default: <root>/analysis_runs.csv)")
    ap.add_argument("--top_csv", type=str, default=None,
                    help="Output top-k csv (default: <root>/analysis_top.csv)")
    ap.add_argument("--topk", type=int, default=15, help="Top K runs to print/save")
    ap.add_argument("--sort_by", type=str, default="dpsnr_mean",
                    choices=["dpsnr_mean", "psnr_auto_mean", "dssim_mean", "ssim_auto_mean", "mix_ratio"],
                    help="Sort key for ranking runs")
    args = ap.parse_args()

    root = args.root
    paths = sorted(glob.glob(os.path.join(root, args.pattern)))
    if not paths:
        raise FileNotFoundError(f"No csv matched: {os.path.join(root, args.pattern)}")

    dfs = []
    for p in paths:
        try:
            dfs.append(load_one_metrics(p))
        except Exception as e:
            print(f"[warn] failed to read {p}: {e}")

    if not dfs:
        raise RuntimeError("No readable metrics csv files.")

    # summarize per run
    runs = []
    for df in dfs:
        # in case a file contains multiple run_names, split
        if "run_name" in df.columns and df["run_name"].nunique() > 1:
            for rn, sub in df.groupby("run_name"):
                runs.append(summarize_run(sub.copy()))
        else:
            runs.append(summarize_run(df))

    out = pd.DataFrame(runs)

    # Sorting
    out_sorted = out.sort_values(by=args.sort_by, ascending=False, na_position="last").reset_index(drop=True)

    out_csv = args.out_csv or os.path.join(root, "analysis_runs.csv")
    top_csv = args.top_csv or os.path.join(root, "analysis_top.csv")

    out_sorted.to_csv(out_csv, index=False, encoding="utf-8-sig")

    top = out_sorted.head(args.topk).copy()
    top.to_csv(top_csv, index=False, encoding="utf-8-sig")

    # Print a compact top-k table
    show_cols = [
        "run_name",
        "psnr_base_mean", "psnr_auto_mean", "dpsnr_mean",
        "ssim_base_mean", "ssim_auto_mean", "dssim_mean",
        "mix_ratio",
        "k_topk", "k_ramp_mode", "k_alpha", "k_beta", "use_gamma",
        "top_local_in_mix", "top_global_in_mix",
    ]
    show_cols = [c for c in show_cols if c in top.columns]

    print("====================================================")
    print(f"[DONE] aggregated {len(out_sorted)} runs from {len(paths)} files")
    print(f"[SAVE] {out_csv}")
    print(f"[SAVE] {top_csv}")
    print("----------------------------------------------------")
    print(top[show_cols].to_string(index=False))
    print("====================================================")


if __name__ == "__main__":
    main()

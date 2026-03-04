# analyze_mol_results.py
# -*- coding: utf-8 -*-
"""
对 MoL 实验的 summary.csv 和 metrics.csv 进行系统对比分析。

功能：
1. 读取 summary_csv（每个 run 一行平均指标），分析：
   - 不同 embedder / sim_metric / temperature / topk 的平均 PSNR/SSIM
   - 各 run 的提升幅度（psnr_mix - psnr_base, ssim_mix - ssim_base）
   - PSNR/SSIM 最好的若干组合
   - 绘制：温度-性能曲线、embedder×temperature 热力图

2. 可选：读取 metrics_glob（多份 per-image metrics.csv），分析：
   - 每个 run 的提升分布（直方图）
   - 找出「混合 LoRA 反而变差」的样本数

使用示例：

python analyze_mol_results.py \
  --summary_csv results/haze_summary.csv \
  --metrics_glob "results/haze_*/metrics.csv" \
  --out_dir results/haze_analysis
"""

import os
import glob
import argparse
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ====================== 1. 分析 summary.csv ======================

def analyze_summary(summary_csv: str, out_dir: Optional[str] = None):
    print(f"[INFO] Loading summary_csv: {summary_csv}")
    df = pd.read_csv(summary_csv)
    if df.empty:
        print("[WARN] summary_csv is empty")
        return

    # 自动添加「提升」指标：mix - base
    df["d_psnr"] = df["psnr_mix"] - df["psnr_base"]
    df["d_ssim"] = df["ssim_mix"] - df["ssim_base"]

    print("\n==== 总体概览（所有实验） ====")
    print(df[["psnr_lq", "psnr_base", "psnr_mix",
              "ssim_lq", "ssim_base", "ssim_mix",
              "d_psnr", "d_ssim"]].describe())

    # 1) 按 embedder 聚合
    if "embedder" in df.columns:
        print("\n==== 按 embedder 聚合 ====")
        print(df.groupby("embedder")[["psnr_mix", "ssim_mix", "d_psnr", "d_ssim"]].mean())

    # 2) 按 sim_metric 聚合
    if "sim_metric" in df.columns:
        print("\n==== 按 sim_metric 聚合 ====")
        print(df.groupby("sim_metric")[["psnr_mix", "ssim_mix", "d_psnr", "d_ssim"]].mean())

    # 3) 按 temperature 聚合
    if "temperature" in df.columns:
        print("\n==== 按 temperature 聚合 ====")
        print(df.groupby("temperature")[["psnr_mix", "ssim_mix", "d_psnr", "d_ssim"]].mean())

    # 4) 按 topk 聚合
    if "topk" in df.columns:
        print("\n==== 按 topk 聚合 ====")
        print(df.groupby("topk")[["psnr_mix", "ssim_mix", "d_psnr", "d_ssim"]].mean())

    # 5) 找出性能最好的几个组合
    print("\n==== PSNR_mix 最佳的前 10 个组合 ====")
    print(df.sort_values("psnr_mix", ascending=False).head(10)[
        ["run_name", "embedder", "sim_metric", "temperature", "topk",
         "psnr_mix", "ssim_mix", "d_psnr", "d_ssim", "num_images"]
    ])

    print("\n==== d_psnr（mix-base 提升）最大的前 10 个组合 ====")
    print(df.sort_values("d_psnr", ascending=False).head(10)[
        ["run_name", "embedder", "sim_metric", "temperature", "topk",
         "psnr_mix", "ssim_mix", "d_psnr", "d_ssim", "num_images"]
    ])

    # 6) 可视化：温度-性能 曲线
    if out_dir is not None:
        ensure_dir(out_dir)

        if "temperature" in df.columns:
            # 若只有一个 embedder，可以直接画一条线；否则每个 embedder 一条线
            if "embedder" in df.columns:
                for emb, sub in df.groupby("embedder"):
                    sub_group = sub.groupby("temperature")["psnr_mix"].mean()
                    plt.figure()
                    sub_group.sort_index().plot(marker="o")
                    plt.xlabel("temperature")
                    plt.ylabel("PSNR_mix")
                    plt.title(f"PSNR_mix vs temperature (embedder={emb})")
                    plt.grid(True)
                    plt.tight_layout()
                    save_path = os.path.join(out_dir, f"psnr_vs_temp_{emb}.png")
                    plt.savefig(save_path, dpi=200)
                    plt.close()
                    print(f"[PLOT] saved {save_path}")

            else:
                temp_group = df.groupby("temperature")["psnr_mix"].mean()
                plt.figure()
                temp_group.sort_index().plot(marker="o")
                plt.xlabel("temperature")
                plt.ylabel("PSNR_mix")
                plt.title("PSNR_mix vs temperature")
                plt.grid(True)
                plt.tight_layout()
                save_path = os.path.join(out_dir, "psnr_vs_temp.png")
                plt.savefig(save_path, dpi=200)
                plt.close()
                print(f"[PLOT] saved {save_path}")

        # 7) 可视化：embedder × temperature 热力图（如果 seaborn 可用）
        if _HAS_SEABORN and "embedder" in df.columns and "temperature" in df.columns:
            pivot = df.pivot_table(
                values="psnr_mix",
                index="embedder",
                columns="temperature",
                aggfunc="mean"
            )
            plt.figure(figsize=(8, 4))
            sns.heatmap(pivot, annot=True, fmt=".2f")
            plt.title("PSNR_mix vs temperature vs embedder")
            plt.tight_layout()
            save_path = os.path.join(out_dir, "heatmap_psnr_embedder_temp.png")
            plt.savefig(save_path, dpi=200)
            plt.close()
            print(f"[PLOT] saved {save_path}")

    return df


# ====================== 2. 分析 metrics.csv（逐图） ======================

def analyze_metrics_glob(metrics_glob: str, out_dir: Optional[str] = None):
    """
    metrics_glob 例如: "results/haze_*/metrics.csv"
    会把所有匹配的 CSV 读进来，并在里面已经有 run_name / embedder 等信息。
    """
    paths = sorted(glob.glob(metrics_glob))
    if not paths:
        print(f"[WARN] No metrics csv matched glob: {metrics_glob}")
        return None

    print(f"[INFO] Found {len(paths)} metrics csv files.")
    dfs: List[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        if df.empty:
            continue
        # 过滤掉 name="__mean__" 的行（那是整体平均）
        df = df[df["name"] != "__mean__"]
        # 添加一个列: metrics_csv_path，便于追踪
        df["metrics_path"] = p
        dfs.append(df)

    if not dfs:
        print("[WARN] All metrics csv are empty.")
        return None

    all_df = pd.concat(dfs, ignore_index=True)

    # 提升
    all_df["d_psnr"] = all_df["psnr_mix"] - all_df["psnr_base"]
    all_df["d_ssim"] = all_df["ssim_mix"] - all_df["ssim_base"]

    print("\n==== Per-image 全局概览 ====")
    print(all_df[["psnr_lq","psnr_base","psnr_mix",
                  "ssim_lq","ssim_base","ssim_mix",
                  "d_psnr","d_ssim"]].describe())

    # 看看有多少样本出现「混合反而变差」
    worse = all_df[all_df["d_psnr"] < 0]
    print(f"\n[STAT] 混合后 PSNR 下降的样本数: {len(worse)} / {len(all_df)} "
          f"({len(worse)/len(all_df)*100:.2f}%)")

    # 哪几个 run 问题最多
    print("\n==== 每个 run_name 中，PSNR 下降样本数 Top-10 ====")
    tmp = worse.groupby("run_name")["name"].count().sort_values(ascending=False).head(10)
    print(tmp)

    # 可视化：d_psnr 直方图
    if out_dir is not None:
        ensure_dir(out_dir)
        plt.figure()
        all_df["d_psnr"].hist(bins=50)
        plt.xlabel("d_psnr = psnr_mix - psnr_base")
        plt.ylabel("count")
        plt.title("Histogram of PSNR improvement (all runs)")
        plt.tight_layout()
        save_path = os.path.join(out_dir, "hist_d_psnr_all.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[PLOT] saved {save_path}")

    return all_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_csv", type=str, required=True,
        help="由 infer_retrieval.py 生成的 summary_csv，用于跨实验对比"
    )
    parser.add_argument(
        "--metrics_glob", type=str, default=None,
        help="可选，用通配符匹配多个 metrics.csv，例如 'results/haze_*/metrics.csv'"
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="可选：分析图表输出目录，例如 results/haze_analysis"
    )
    args = parser.parse_args()

    if args.out_dir is not None:
        ensure_dir(args.out_dir)

    # 1) 分析 summary
    df_summary = analyze_summary(args.summary_csv, out_dir=args.out_dir)

    # 2) 分析所有 metrics（逐图）
    if args.metrics_glob is not None:
        df_all_metrics = analyze_metrics_glob(args.metrics_glob, out_dir=args.out_dir)
    else:
        df_all_metrics = None

    print("\n[INFO] Done. You can now open CSV & PNG under:", args.out_dir or ".")


if __name__ == "__main__":
    main()

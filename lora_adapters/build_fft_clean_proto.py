# tools/build_fft_clean_proto_from_trainlists.py
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List
import argparse
import numpy as np
import torch

from lora_adapters.embedding_fft import FFTEnhancedEmbedder


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def collect_gt_from_trainlists(
    txt_root: Path,
    domains: List[str],
    data_root: Path,
    gt_keywords: List[str],
    max_images: int
) -> List[Path]:
    """
    从多个 train_list_<domain>.txt 中自动解析 GT 图像路径。
    解析规则：
      1. 每行所有图片 token 全拿出来 img_tokens
      2. 优先选择文件名中包含 gt_keywords 的作为 GT
      3. 若无关键词且有 >=2 张图，则默认最后一张为 GT（常见 LQ,GT 格式）
      4. 否则 fallback 用第一张
    """
    all_paths: List[Path] = []

    for d in domains:
        txt_path = txt_root / f"train_list_{d}.txt"
        if not txt_path.is_file():
            print(f"[WARN] train list not found: {txt_path}, skip domain {d}")
            continue

        print(f"[INFO] parsing GT from {txt_path}")
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                tokens = line.split()
                img_tokens = [t for t in tokens if t.lower().endswith(IMG_EXTS)]
                if not img_tokens:
                    continue

                # 优先使用包含关键字的作为 GT（如 gt/hq/clean/norain/clear）
                lower_tokens = [t.lower() for t in img_tokens]
                cand_idx = None
                for i, t in enumerate(lower_tokens):
                    if any(kw in t for kw in gt_keywords):
                        cand_idx = i
                        break

                if cand_idx is not None:
                    chosen = img_tokens[cand_idx]
                else:
                    # 若没有关键词，若有两张图，默认最后一张是 GT（LQ,GT）
                    if len(img_tokens) >= 2:
                        chosen = img_tokens[-1]
                    else:
                        chosen = img_tokens[0]

                p = Path(chosen)
                if not p.is_absolute():
                    p = data_root / p

                if p.is_file():
                    all_paths.append(p)
                else:
                    print(f"[WARN] GT file not found: {p}")

                if len(all_paths) >= max_images:
                    print(f"[INFO] reached max_images={max_images}, stop collecting.")
                    return all_paths

    print(f"[INFO] collected {len(all_paths)} GT images from trainlists.")
    return all_paths


def main():
    ap = argparse.ArgumentParser(
        description="Build FFT clean prototype (mu_clean) from train_list_<domain>.txt (using GT images)."
    )
    ap.add_argument("--txt_root", type=str, required=True,
                    help="train_list_*.txt 所在目录，例如 data/txt_lists")
    ap.add_argument("--domains", type=str, required=True,
                    help="逗号分隔的域名，例如 rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2")
    ap.add_argument("--data_root", type=str, required=True,
                    help="数据根目录，用于补全相对路径，例如 data")
    ap.add_argument("--out", type=str, required=True,
                    help="输出 clean prototype 路径，例如 weights/fft_clean_proto.npy")
    ap.add_argument("--max_images", type=int, default=2000,
                    help="最多使用多少张 GT 图像来计算 clean prototype")

    # FFTEnhancedEmbedder 参数（要和你后面使用时保持一致）
    ap.add_argument("--fft_resize", type=int, default=256)
    ap.add_argument("--radial_bins", type=int, default=32)
    ap.add_argument("--angle_bins", type=int, default=16)
    ap.add_argument("--patch_size", type=int, default=32)

    ap.add_argument(
        "--gt_keywords", type=str,
        default="gt,hq,clean,norain,clear",
        help="用逗号分隔的关键词，文件名中包含任一关键词就认为是 GT"
    )

    args = ap.parse_args()

    txt_root = Path(args.txt_root)
    data_root = Path(args.data_root)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    gt_keywords = [k.strip().lower() for k in args.gt_keywords.split(",") if k.strip()]

    # 1) 从所有 train_list_<domain>.txt 里收集 GT 路径
    gt_paths = collect_gt_from_trainlists(
        txt_root=txt_root,
        domains=domains,
        data_root=data_root,
        gt_keywords=gt_keywords,
        max_images=args.max_images,
    )

    if not gt_paths:
        print("[ERROR] No GT images collected, abort.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2) 构建 FFTEnhancedEmbedder（此处不做 residual）
    emb = FFTEnhancedEmbedder(
        device=device,
        resize=args.fft_resize,
        radial_bins=args.radial_bins,
        angle_bins=args.angle_bins,
        patch_size=args.patch_size,
        clean_proto_path=None,
        use_residual=False,   # 这里是用所有 GT 来算 mu_clean，所以不能减自己
    )

    vecs = []
    for i, p in enumerate(gt_paths):
        try:
            v = emb.embed_image(p)  # np.ndarray [C]
            vecs.append(v)
        except Exception as e:
            print(f"[ERROR] fail on {p}: {e}")
        if (i + 1) % 50 == 0:
            print(f"[INFO] processed {i+1}/{len(gt_paths)} GT images")

    if not vecs:
        print("[ERROR] No valid embeddings computed, abort.")
        return

    arr = np.stack(vecs, axis=0).astype(np.float32)  # [N,C]
    mu_clean = arr.mean(axis=0)
    print(f"[INFO] mu_clean shape = {mu_clean.shape}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mu_clean)
    print(f"[DONE] clean prototype saved to {out_path}")


if __name__ == "__main__":
    main()

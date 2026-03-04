# lora_adapters/build_prototypes.py
import os, argparse
from pathlib import Path
from typing import Union, List
import torch
import numpy as np

from sklearn.cluster import KMeans  # ⭐ 新增：K-means 聚类，用于每域多原型
from lora_adapters.embedding_dinov2 import DINOv2Embedder   # DINOv2
# CLIP / FFT 类在用到时再 import，避免你没装对应依赖也报错


# ----------------------------------------
# 自动递归收集全部图片
# ----------------------------------------
def collect_all_images(root: Union[str, Path],
                       exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    root = Path(root)
    files = []
    for e in exts:
        files += list(root.rglob(f"*{e}"))
    return files


# ----------------------------------------
# 从 txt 自动解析 LQ 图片路径
# ----------------------------------------
def load_images_from_txt(
    txt_path: Union[str, Path],
    data_root: Union[str, Path] = "",
    exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
) -> List[Path]:

    txt_path = Path(txt_path)
    data_root = Path(data_root)
    files: List[Path] = []

    if not txt_path.is_file():
        print(f"[txt] warning: {txt_path} 不存在")
        return files

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            tokens = line.split()
            img_tokens = [t for t in tokens if t.lower().endswith(exts)]
            if not img_tokens:
                continue

            # 优先拿 'lq'
            lq_candidates = [t for t in img_tokens if "lq" in t.lower()]
            chosen = lq_candidates[0] if lq_candidates else img_tokens[0]

            p = Path(chosen)
            if not p.is_absolute() and data_root != Path(""):
                p = data_root / p

            files.append(p)

    return files


def get_embedder_tag(args) -> str:
    """
    根据 embedder 配置生成一个在磁盘上区分用的 tag，
    用来命名 avg_embedding_<tag>.npy / prototypes_<tag>.npy / <tag>.pt
    """
    if args.embedder == 'dino_v2':
        ckpt = str(getattr(args, "dino_ckpt", ""))
        if 'vit_base_patch14' in ckpt:
            return 'dinov2_vitb14'
        else:
            return 'dinov2'
    elif args.embedder == 'clip':
        model = getattr(args, "clip_model", "ViT-B-16")
        return f"clip_{model.replace('/', '_').lower()}"
    elif args.embedder == 'fft':
        return 'fft_amp'
    elif args.embedder == 'fft_enhanced':
        return 'fft_enh'
    else:
        return args.embedder


# ----------------------------------------
# main
# ----------------------------------------
@torch.no_grad()
def main():
    DEFAULT_OUT = "weights/prototypes/embeddings.pt"

    parser = argparse.ArgumentParser(
        description="Build prototypes for weather domains (DINOv2 / CLIP / FFT / FFTEnhanced)"
    )

    parser.add_argument(
        "--domains", type=str, default="rain,snow,fog,blur",
        help="逗号分隔的域名列表"
    )
    parser.add_argument(
        '--embedder',
        type=str,
        default='dino_v2',
        # ⭐ 这里已经加入 fft_enhanced ⭐
        choices=['dino_v2', 'clip', 'fft', 'fft_enhanced'],
        help='which embedder to use: dino_v2 / clip / fft / fft_enhanced'
    )

    # ⭐ 新增：每个域的 K-means 原型个数（只对 DINO/CLIP 真正启用）
    parser.add_argument(
        '--k_proto', type=int, default=1,
        help="每个域使用多少个 prototype（K-means 聚类中心）；1 表示只用单均值原型"
    )

    # FFT 参数（基础版 + 增强版都会用到 resize / out_size）
    parser.add_argument('--fft_resize', type=int, default=256,
                        help='FFT embedding: resize image to this size before FFT')
    parser.add_argument('--fft_center_crop', type=int, default=128,
                        help='FFT embedding: center crop for simple FFTAmplitudeEmbedder')
    parser.add_argument('--fft_out_size', type=int, default=32,
                        help='FFT embedding: out_size or patch_size (for fft_enhanced)')

    # FFTEnhanced 专用：clean prototype 路径
    parser.add_argument('--fft_clean_proto', type=str, default=None,
                        help='可选：FFTEnhancedEmbedder 使用的 clean prototype .npy 路径')

    # CLIP 参数
    parser.add_argument(
        "--clip_model", type=str, default="ViT-B-16",
        help="CLIP 模型名，例如 ViT-B-16 / ViT-L-14"
    )
    parser.add_argument(
        "--clip_pretrained", type=str, default="openai",
        help="CLIP 预训练权重名，例如 openai / laion2b_s32b_b82k 等"
    )

    parser.add_argument(
        "--loradb_root", type=str, default="weights/lora",
        help="为每个域保存 avg_embedding_<tag>.npy / prototypes_<tag>.npy 的根目录"
    )

    parser.add_argument(
        "--data_root", type=str, required=True,
        help="数据根目录，例如 data"
    )

    parser.add_argument(
        "--subpath", type=str, default="train/lq",
        help="若未使用 txt，则自动扫描 data_root/domain/subpath"
    )

    parser.add_argument(
        "--txt_root", type=str, default=None,
        help="可选：训练集 txt 所在目录；使用 train_list_<domain>.txt 命名"
    )

    parser.add_argument(
        "--out", type=str, default=DEFAULT_OUT,
        help="保存 prototype 的输出路径（字典，域名 -> 向量/矩阵）"
    )

    parser.add_argument(
        "--max_images", type=int, default=300,
        help="每个域最多使用多少张图计算 embedding"
    )

    parser.add_argument(
        "--dino_ckpt", type=str,
        default="weights/dinov2/vit_base_patch14_dinov2.lvd142m-384.pth",
        help="本地 DINOv2 权重（仅当 embedder=dino_v2 时使用）"
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 根据 embedder 得到 tag，比如 dinov2_vitb14 / clip_vit-b-16 / fft_amp / fft_enh
    embedder_tag = get_embedder_tag(args)
    print(f"[INFO] using embedder_tag = {embedder_tag}")

    # 如果 out 还在默认值，就自动根据 tag 生成一个更合理的名字
    if args.out == DEFAULT_OUT:
        out_dir = Path("weights/prototypes")
        out_dir.mkdir(parents=True, exist_ok=True)
        args.out = str(out_dir / f"{embedder_tag}.pt")
        print(f"[INFO] auto set --out to {args.out}")

    # 1) 选择 embedder
    if args.embedder == "dino_v2":
        embedder = DINOv2Embedder(device=device, ckpt_path=args.dino_ckpt)
        print(f"[embedder] 使用 DINOv2, ckpt = {args.dino_ckpt}")

    elif args.embedder == "clip":
        from lora_adapters.embedding_clip import CLIPEmbedder
        embedder = CLIPEmbedder(
            device=device,
            model_name=args.clip_model,
            pretrained=args.clip_pretrained
        )
        print(f"[embedder] 使用 CLIP: {args.clip_model} ({args.clip_pretrained})")

    elif args.embedder == 'fft':
        from lora_adapters.embedding_fft import FFTAmplitudeEmbedder
        embedder = FFTAmplitudeEmbedder(
            device=device,
            resize=args.fft_resize,
            center_crop=args.fft_center_crop,
            out_size=args.fft_out_size,
        )
        print(f'[embedder] 使用 FFTAmplitudeEmbedder (resize={args.fft_resize}, '
              f'center_crop={args.fft_center_crop}, out_size={args.fft_out_size}) on {device}')

    elif args.embedder == 'fft_enhanced':
        # ⭐ 使用升级版 FFTEnhancedEmbedder ⭐
        from lora_adapters.embedding_fft import FFTEnhancedEmbedder
        embedder = FFTEnhancedEmbedder(
            device=device,
            resize=args.fft_resize,
            radial_bins=32,
            angle_bins=16,
            patch_size=args.fft_out_size,
            clean_proto_path=args.fft_clean_proto,
            use_residual=True,
        )
        print(f'[embedder] 使用 FFTEnhancedEmbedder (resize={args.fft_resize}, '
              f'radial_bins=32, angle_bins=16, patch_size={args.fft_out_size}, '
              f'clean_proto={args.fft_clean_proto}) on {device}')

    else:
        raise ValueError(f"unknown embedder: {args.embedder}")

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    prototypes = {}

    # -------- 2) 处理每个 domain --------
    for d in domains:
        print(f"\n========================\n[domain] {d}")

        files: List[Path] | None = None

        # 优先使用 txt：train_list_<domain>.txt
        if args.txt_root is not None:
            txt_file = Path(args.txt_root) / f"train_list_{d}.txt"
            if txt_file.is_file():
                print(f"[txt] 使用 TXT: {txt_file}")
                files = load_images_from_txt(
                    txt_path=txt_file,
                    data_root=args.data_root
                )
            else:
                print(f"[txt] 未找到 {txt_file}，fallback 自动扫描")

        # fallback: 自动递归扫描目录 data_root/domain/subpath
        if files is None:
            domain_root = Path(args.data_root) / d / args.subpath
            print(f"[auto] 递归搜索: {domain_root}")
            files = collect_all_images(domain_root)

        if not files:
            print(f"[warn] 域 {d} 找不到有效图像，跳过")
            continue

        # 限制 max_images
        files = files[:args.max_images]
        print(f"[proto] 使用 {len(files)} 张图像构建 prototype")

        # -------- 提取 embedding --------
        vecs = []
        for img_path in files:
            try:
                v = embedder.embed_image(img_path)  # np.ndarray [C]
                vecs.append(v)
            except Exception as e:
                print(f"[error] 处理 {img_path} 失败: {e}")

        if len(vecs) == 0:
            print(f"[warn] 域 {d} 有效特征数为 0，跳过")
            continue

        vecs = np.stack(vecs, axis=0).astype(np.float32)   # [N, C]
        N, C = vecs.shape
        print(f"[proto] 域 {d} embedding 数量 N={N}, 维度 C={C}")

        # ⭐ 新增：按需使用 K-means 生成多原型（仅 DINO/CLIP 时启用）
        use_kmeans = (args.k_proto > 1) and (args.embedder in ("dino_v2", "clip"))
        if use_kmeans:
            K = min(args.k_proto, N)
            print(f"[proto] 使用 K-means 为域 {d} 生成 {K} 个原型 (embedder={args.embedder})")
            kmeans = KMeans(
                n_clusters=K,
                random_state=0,   # 固定随机种子，保证复现
                n_init=10,
            )
            kmeans.fit(vecs)
            centers = kmeans.cluster_centers_.astype(np.float32)  # [K, C]
        else:
            # 退回单均值原型（兼容旧行为）
            centers = vecs.mean(axis=0, keepdims=True).astype(np.float32)  # [1, C]
            if args.k_proto > 1 and args.embedder not in ("dino_v2", "clip"):
                print(f"[proto] 注意: embedder={args.embedder} 暂不启用 K-means，多原型设置被忽略，退回单均值原型。")
            else:
                print(f"[proto] 域 {d} 使用单均值原型。")

        # 保存多原型：prototypes_<tag>.npy（shape [K, C]，K=1 时即单原型）
        dom_dir = Path(args.loradb_root) / d
        dom_dir.mkdir(parents=True, exist_ok=True)
        proto_path = dom_dir / f"prototypes_{embedder_tag}.npy"
        np.save(proto_path, centers)
        print(f"[proto] 域 {d} prototypes_{embedder_tag}.npy 已保存到: {proto_path} | shape={centers.shape}")

        # 仍然保存 avg_embedding_<tag>.npy（用多原型的均值，兼容旧逻辑）
        proto_mean = centers.mean(axis=0)  # [C]
        avg_path = dom_dir / f"avg_embedding_{embedder_tag}.npy"
        np.save(avg_path, proto_mean)
        print(f"[proto] 域 {d} avg_embedding_{embedder_tag}.npy 已保存到: {avg_path} | 维度={proto_mean.shape}")

        # prototypes 字典里直接存 [K, C]（K=1 也 ok）
        prototypes[d] = torch.from_numpy(centers)

    # -------- 3) 保存所有 prototype (dict) --------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prototypes, out_path)
    print(f"\n🎉 全部 prototype 已保存到: {out_path}")


if __name__ == "__main__":
    main()

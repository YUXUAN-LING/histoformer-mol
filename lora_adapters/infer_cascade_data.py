# lora_adapters/infer_cascade_data.py
# -*- coding: utf-8 -*-
"""
自动检索 + 级联推理（Cascade Inference）脚本

核心思想：
- 混合退化常常是 “local(高频局部: rain/snow) + global(低频全局: haze/low)” 叠加
- Stage1 先处理 local，再 Stage2 处理 global（或反向对照）
- Stage2 的路由输入支持：
    out      : 用 stage1 输出 y1 路由
    residual : 用 residual r = x - y1 或 |x - y1| 路由
    both     : 两套都跑，输出两张结果用于对比

功能：
A) 输入：单图/文件夹 + pair_list(LQ GT) + gt_root 自动配对
B) 输出：base/stage1/stage2(out/res) 结果图 + concat 对比图
C) 指标：PSNR/SSIM per-image + mean summary，输出 CSV
D) 路由日志：stage1/stage2 topk 域、权重、margin/置信度写入 CSV
E) 阈值跳过：stage1_min_margin / stage2_min_margin（不确定时跳过避免破坏）
F) 可复现：seed + deterministic
G) 工程兼容：支持 python -m 与直接 python 文件运行；embedder_tag 对齐并做维度 mismatch fail-fast

用法示例：
python -m lora_adapters.infer_cascade_data \
  --input data/mix/test_lq \
  --pair_list data/mix/test_pairs.txt \
  --output_dir results/cascade_mix \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --loradb weights/lora \
  --local_domains rain,snow \
  --global_domains haze,low \
  --order local2global \
  --stage2_source both \
  --embedder dino_v2 \
  --prototypes weights/prototypes/dinov2_vitb14.pt \
  --topk1 1 --topk2 1 \
  --temperature1 0.05 --temperature2 0.05 \
  --stage2_min_margin 0.02 \
  --run_name exp_cascade_both \
  --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


# -----------------------------
# 兼容：直接 python 文件运行（避免 ModuleNotFoundError: lora_adapters）
# -----------------------------
if __name__ == "__main__" and __package__ is None:
    this = Path(__file__).resolve()
    repo_root = this.parent.parent
    sys.path.insert(0, str(repo_root))


# -----------------------------
# 可选依赖：seed / metrics（有则用你的实现；无则用 fallback）
# -----------------------------
def _set_seed(seed: int, deterministic: bool = False):
    try:
        # 你仓库里常见位置：lora_adapters/common/seed.py
        from lora_adapters.common.seed import set_seed  # type: ignore
        set_seed(seed=seed, deterministic=deterministic)
        return
    except Exception:
        pass

    try:
        from lora_adapters.seed import set_seed  # type: ignore
        set_seed(seed=seed, deterministic=deterministic)
        return
    except Exception:
        pass

    # fallback
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _tensor_psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """
    fallback PSNR: assume x,y in [0,1], shape [1,3,H,W]
    """
    mse = torch.mean((x - y) ** 2).clamp_min(eps)
    psnr = -10.0 * torch.log10(mse)
    return float(psnr.item())


def _gaussian_window(window_size: int, sigma: float, channel: int, device: torch.device):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,ws,ws]
    w = w.repeat(channel, 1, 1, 1)  # [C,1,ws,ws]
    return w


def _tensor_ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    """
    fallback SSIM (简版)：assume x,y in [0,1], shape [1,3,H,W]
    """
    assert x.shape == y.shape and x.dim() == 4 and x.size(0) == 1
    device = x.device
    channel = x.size(1)
    w = _gaussian_window(window_size, sigma, channel, device)

    mu_x = F.conv2d(x, w, padding=window_size // 2, groups=channel)
    mu_y = F.conv2d(y, w, padding=window_size // 2, groups=channel)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, w, padding=window_size // 2, groups=channel) - mu_x2
    sigma_y2 = F.conv2d(y * y, w, padding=window_size // 2, groups=channel) - mu_y2
    sigma_xy = F.conv2d(x * y, w, padding=window_size // 2, groups=channel) - mu_xy

    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12)
    return float(ssim_map.mean().item())


def _get_metrics_fns():
    """
    优先使用你仓库里的 tensor_psnr / tensor_ssim；没有就 fallback。
    """
    for mod in [
        "lora_adapters.common.metrics",
        "lora_adapters.metrics",
    ]:
        try:
            m = __import__(mod, fromlist=["tensor_psnr", "tensor_ssim"])
            return m.tensor_psnr, m.tensor_ssim
        except Exception:
            continue
    return _tensor_psnr, _tensor_ssim


tensor_psnr, tensor_ssim = _get_metrics_fns()


# -----------------------------
# 工具函数：文件/CSV/图像
# -----------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_images(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    out = []
    for dp, _, fnames in os.walk(root):
        for fn in fnames:
            ext = Path(fn).suffix.lower()
            if ext in IMG_EXTS:
                out.append(Path(dp) / fn)
    out.sort()
    return out


def parse_pair_list(pair_list: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    with pair_list.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            lq, gt = Path(parts[0]), Path(parts[1])
            pairs.append((lq, gt))
    return pairs


def resolve_gt_by_stem(lq_path: Path, gt_root: Path) -> Optional[Path]:
    """
    无 pair_list 时：用 gt_root + (basename 或 stem) 找 GT。
    """
    cand = gt_root / lq_path.name
    if cand.exists():
        return cand

    # fallback: stem match (允许扩展名不同)
    stem = lq_path.stem
    for ext in IMG_EXTS:
        cand2 = gt_root / f"{stem}{ext}"
        if cand2.exists():
            return cand2

    # 更宽松：遍历 gt_root 同 stem 的任意文件
    for p in gt_root.glob(stem + ".*"):
        if p.suffix.lower() in IMG_EXTS:
            return p

    return None


def append_csv_row(csv_path: Path, header: List[str], row: Dict):
    ensure_dir(csv_path.parent)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow(row)


def tensor_to_pil01(x: torch.Tensor) -> Image.Image:
    """
    x: [1,3,H,W] in [0,1]
    """
    x = x.detach().clamp(0, 1).cpu().squeeze(0)
    return T.ToPILImage()(x)


# -----------------------------
# LoRA 相关：清零 _Single
# -----------------------------
def zero_single_lora(net: torch.nn.Module, domain_name: str = "_Single"):
    """
    将注入到 LoRALinear 的 lora_down/lora_up 的指定 domain 权重清零。
    这样可以保证不同阶段加载 LoRA 时不会残留污染。
    """
    for m in net.modules():
        if hasattr(m, "lora_down") and hasattr(m, "lora_up"):
            try:
                down = m.lora_down[domain_name]
                up = m.lora_up[domain_name]
                if hasattr(down, "weight") and down.weight is not None:
                    down.weight.data.zero_()
                if hasattr(up, "weight") and up.weight is not None:
                    up.weight.data.zero_()
            except Exception:
                continue


# -----------------------------
# 路由记录结构
# -----------------------------
@dataclass
class RouteInfo:
    metric: str
    topk: List[Tuple[str, float]]            # (domain, weight) from select_topk
    scores: List[Tuple[str, float]]          # cosine(sim) or l2(dist)
    margin: Optional[float]                  # sim1-sim2 (cosine) or dist2-dist1 (l2)
    uncertain: bool                          # margin threshold triggered
    reason: str                              # "" or "low_margin" etc


def compute_scores(orch, emb: np.ndarray) -> List[Tuple[str, float]]:
    metric = getattr(orch, "sim_metric", "cosine")
    e = np.asarray(emb).reshape(-1).astype(np.float32)

    def cosine(a, b):
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(a, b))

    def l2(a, b):
        return float(np.linalg.norm(a - b))

    scores = []
    for name, dom in orch.domains.items():
        # 取多原型
        if hasattr(dom, "prototypes"):
            p = np.asarray(dom.prototypes)
            if p.ndim == 1:
                p = p[None, :]
        else:
            # 旧版本 fallback
            p = np.asarray(get_domain_embedding(dom))[None, :]

        if metric == "cosine":
            s = max(cosine(e, p[i]) for i in range(p.shape[0]))  # 越大越好
        elif metric in ("euclidean", "l2"):
            s = min(l2(e, p[i]) for i in range(p.shape[0]))      # 越小越好
        else:
            raise ValueError(f"Unknown sim_metric: {metric}")

        scores.append((name, float(s)))

    if metric == "cosine":
        scores.sort(key=lambda x: x[1], reverse=True)
    else:
        scores.sort(key=lambda x: x[1])
    return scores




def compute_margin(metric: str, sorted_scores: List[Tuple[str, float]]) -> Optional[float]:
    if len(sorted_scores) < 2:
        return None
    v1 = sorted_scores[0][1]
    v2 = sorted_scores[1][1]
    if metric == "cosine":
        return float(v1 - v2)
    else:
        # distance: dist2 - dist1 (越大越“自信”)
        return float(v2 - v1)

def get_domain_embedding(dom) -> np.ndarray:
    """
    兼容你当前仓库的 Domain:
      - dom.prototypes: np.ndarray, 形状可能是 (K,D) 或 (D,)
    也兼容旧版本：
      - dom.avg_embedding 等
    返回用于“维度检查/日志分数”的 1D 向量（D,）。
    """
    # 旧字段兼容
    for k in ["avg_embedding", "prototype", "proto", "embedding", "avg_emb"]:
        if hasattr(dom, k):
            v = np.asarray(getattr(dom, k))
            return v.reshape(-1)

    # 你现在的字段：prototypes
    if hasattr(dom, "prototypes"):
# 你现在的字段：prototypes（更稳：不用 hasattr）
        p = getattr(dom, "prototypes", None)
        if p is not None:
            p = np.asarray(p)
            if p.ndim == 2:
                return p.mean(axis=0).reshape(-1)
            if p.ndim == 1:
                return p.reshape(-1)
            raise AttributeError(f"Domain.prototypes has unsupported shape: {p.shape}")
    raise AttributeError(f"Domain object has no embedding field. Available: {dir(dom)}")


def assert_embed_dim_match(orch, emb: np.ndarray, hint: str):
    e = np.asarray(emb).reshape(-1)
    any_dom = next(iter(orch.domains.values()))

    # 用 prototypes/avg_embedding 得到一个代表向量
    proto_vec = get_domain_embedding(any_dom).reshape(-1)

    if e.shape[0] != proto_vec.shape[0]:
        raise ValueError(
            f"[embed-dim-mismatch] {hint}: embed_dim={e.shape[0]} "
            f"but prototype_dim={proto_vec.shape[0]}. "
            f"（你当前 prototype 是 KxD 形式时，我这里取 mean 后检查 D）"
        )


def route_with_margin(
    orch,
    emb: np.ndarray,
    top_k: int,
    temperature: float,
    min_margin: float,
    stage_name: str,
) -> RouteInfo:
    metric = getattr(orch, "sim_metric", "cosine")
    emb = np.asarray(emb).reshape(-1).astype(np.float32)

    assert_embed_dim_match(orch, emb, hint=stage_name)

    # raw scores for logging + margin
    all_scores = compute_scores(orch, emb)
    margin = compute_margin(metric, all_scores)

    # select weights (same逻辑走 orch.select_topk)
    topk = orch.select_topk(emb, top_k=top_k, temperature=temperature)

    uncertain = False
    reason = ""
    if margin is not None and min_margin is not None:
        if float(margin) < float(min_margin):
            uncertain = True
            reason = "low_margin"

    return RouteInfo(
        metric=metric,
        topk=topk,
        scores=all_scores[: max(top_k, 5)],  # 记录前几项即可
        margin=margin,
        uncertain=uncertain,
        reason=reason,
    )


# -----------------------------
# 推理核心：pad->net->crop
# -----------------------------
@torch.no_grad()
def forward_restore(net: torch.nn.Module, x: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """
    x: [1,3,H,W] in [0,1]
    """
    assert x.dim() == 4 and x.size(0) == 1
    _, _, h, w = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h or pad_w:
        x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    else:
        x_in = x
    y = net(x_in)
    y = y[:, :, :h, :w]
    return y


def infer_embedder_tag(args) -> str:
    if args.embedder_tag and args.embedder_tag.strip():
        return args.embedder_tag.strip()

    # ✅ 对齐 build_prototypes.py 的实现：get_embedder_tag(args)
    try:
        from lora_adapters.build_prototypes import get_embedder_tag  # type: ignore
        return get_embedder_tag(args)
    except Exception:
        pass

    # fallback（同样对齐命名风格）
    if args.embedder == "clip":
        model = getattr(args, "clip_model", "ViT-B-16")
        return f"clip_{model.replace('/', '_').lower()}"
    if args.embedder == "dino_v2":
        return "dinov2_vitb14"
    if args.embedder == "fft":
        return "fft_amp"
    if args.embedder == "fft_enhanced":
        return "fft_enh"
    return args.embedder


def build_embedder(args, device: str):
    """
    对齐 infer_data.py 的 embedder 选择风格（有的类可能在你仓库里）。
    """
    if args.embedder == "clip":
        from lora_adapters.embedding_clip import CLIPEmbedder  # type: ignore
        return CLIPEmbedder(model_name=args.clip_model, pretrained=args.clip_pretrained, device=device)

    if args.embedder == "dino_v2":
        # 你的仓库里一般是 embedding_dinov2.py（若名字不同你再改一下 import）
        from lora_adapters.embedding_dinov2 import DINOv2Embedder  # type: ignore
        return DINOv2Embedder(prototype_path=args.prototypes, device=device)

    if args.embedder == "fft":
        from lora_adapters.embedding_fft_amp import FFTAmplitudeEmbedder  # type: ignore
        return FFTAmplitudeEmbedder(resize=args.fft_resize, radial_bins=32, angle_bins=16)

    if args.embedder == "fft_enhanced":
        from lora_adapters.embedding_fft_enhanced import FFTEnhancedEmbedder  # type: ignore
        return FFTEnhancedEmbedder(
            resize=args.fft_resize,
            radial_bins=32,
            angle_bins=16,
            patch_size=args.fft_out_size,
            clean_proto_path=args.fft_clean_proto,
            use_residual=True,
        )

    raise ValueError(f"unknown embedder: {args.embedder}")


def parse_domain_list(s: str) -> List[str]:
    return [d.strip() for d in (s or "").split(",") if d.strip()]


def main():
    parser = argparse.ArgumentParser("infer_cascade_data.py - retrieval routed cascade inference")

    # ---- 输入/输出 ----
    parser.add_argument("--input", type=str, required=True, help="单图或文件夹（LQ）")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pair_list", type=str, default="", help="txt，每行：LQ_path GT_path（强推荐）")
    parser.add_argument("--gt_root", type=str, default="", help="无 pair_list 时，用 gt_root + basename/stem 自动配对")

    # ---- 模型 ----
    parser.add_argument("--base_ckpt", type=str, required=True)
    parser.add_argument("--yaml", type=str, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")

    # ---- LoRA 库与域划分 ----
    parser.add_argument("--loradb", type=str, required=True, help="LoRA 库根目录（里面按 domain 子文件夹组织）")
    parser.add_argument("--local_domains", type=str, default="rain,snow", help="local 域列表")
    parser.add_argument("--global_domains", type=str, default="haze,low", help="global 域列表")
    parser.add_argument("--order", type=str, default="local2global", choices=["local2global", "global2local"])

    # ---- Stage2 source ----
    parser.add_argument("--stage2_source", type=str, default="both", choices=["out", "residual", "both"])
    parser.add_argument("--residual_mode", type=str, default="abs", choices=["abs", "signed"], help="residual=x-y1 的 abs 或 signed")

    # ---- 路由参数（分阶段可调）----
    parser.add_argument("--sim_metric", type=str, default="cosine", choices=["cosine", "euclidean", "l2"])
    parser.add_argument("--topk1", type=int, default=1)
    parser.add_argument("--topk2", type=int, default=1)
    parser.add_argument("--temperature1", type=float, default=0.05)
    parser.add_argument("--temperature2", type=float, default=0.05)
    parser.add_argument("--stage1_min_margin", type=float, default=-1.0, help="stage1 margin<thr 视为不确定；<0 表示不启用")
    parser.add_argument("--stage2_min_margin", type=float, default=-1.0, help="stage2 margin<thr 跳过 stage2；<0 表示不启用")

    # ---- embedder ----
    parser.add_argument("--embedder", type=str, default="dino_v2",
                        choices=["dino_v2", "clip", "fft", "fft_enhanced"])
    parser.add_argument("--embedder_tag", type=str, default="", help="不填则自动推断（需与 build_prototypes 一致）")
    parser.add_argument("--prototypes", type=str, default="", help="DINOv2 prototype/model path（如果你的 DINO embedder 需要）")
    parser.add_argument("--clip_model", type=str, default="ViT-B-16")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--fft_resize", type=int, default=224)
    parser.add_argument("--fft_out_size", type=int, default=128)
    parser.add_argument("--fft_clean_proto", type=str, default="")

    # ---- 保存/记录 ----
    parser.add_argument("--save_base", action="store_true", help="保存 base 输出（默认不强制）")
    parser.add_argument("--save_stage_images", action="store_true", help="保存 stage1/stage2 输出图（默认不强制）")
    parser.add_argument("--save_concat", action="store_true", help="保存拼接对比图（默认不强制）")
    parser.add_argument("--run_name", type=str, default="", help="用于 CSV 记录的 run 名称")
    parser.add_argument("--metrics_csv", type=str, default="", help="per-image CSV（默认 output_dir/metrics.csv）")
    parser.add_argument("--summary_csv", type=str, default="", help="summary CSV（默认 output_dir/summary.csv）")

    # ---- 可复现 ----
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")

    # ---- 其他 ----
    parser.add_argument("--max_images", type=int, default=-1, help="调试用，限制处理数量")

    args = parser.parse_args()

    device = args.device
    _set_seed(args.seed, deterministic=args.deterministic)

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else (out_dir / "metrics.csv")
    summary_csv = Path(args.summary_csv) if args.summary_csv else (out_dir / "summary.csv")

    # ---- 构建 base model + inject LoRA(_Single) ----
    from lora_adapters.utils import build_histoformer  # type: ignore
    from lora_adapters.inject_lora import inject_lora  # type: ignore

    try:
    # 旧版本（如果你另一台机器/旧分支是这种）
        net = build_histoformer(args.yaml, args.base_ckpt, device=device)
    except TypeError:
        # 你当前仓库：build_histoformer(weights, yaml_file)
        net = build_histoformer(weights=args.base_ckpt, yaml_file=args.yaml)

    inject_lora(net, rank=args.rank, alpha=args.alpha, domain_list=["_Single"])
    net = net.to(device)
    net.eval()
    print("[DEBUG] net param device:", next(net.parameters()).device)
    # ---- embedder_tag 对齐 ----
    embedder_tag = infer_embedder_tag(args)
    print(f"[INFO] embedder={args.embedder} | embedder_tag={embedder_tag}")

    # ---- 构建 embedder ----
    emb = build_embedder(args, device=device)

    # ---- 构建 orchestrators（Stage1 & Stage2）----
    from lora_adapters.domain_orchestrator import DomainOrchestrator  # type: ignore

    local_domains = parse_domain_list(args.local_domains)
    global_domains = parse_domain_list(args.global_domains)

    if args.order == "local2global":
        stage1_domains = local_domains
        stage2_domains = global_domains
        stage1_name = "local"
        stage2_name = "global"
    else:
        stage1_domains = global_domains
        stage2_domains = local_domains
        stage1_name = "global"
        stage2_name = "local"

    if not stage1_domains:
        raise ValueError("stage1_domains 为空：请检查 --local_domains/--global_domains")
    if not stage2_domains:
        raise ValueError("stage2_domains 为空：请检查 --local_domains/--global_domains")

    orch1 = DomainOrchestrator(
        stage1_domains,
        lora_db_path=args.loradb,
        sim_metric=args.sim_metric,
        temperature=args.temperature1,
        embedder_tag=embedder_tag,
    )
    orch2 = DomainOrchestrator(
        stage2_domains,
        lora_db_path=args.loradb,
        sim_metric=args.sim_metric,
        temperature=args.temperature2,
        embedder_tag=embedder_tag,
    )

    tfm_to_tensor = T.ToTensor()

    # ---- 输入组织：pairs 优先，否则 input + gt_root ----
    pair_list = Path(args.pair_list) if args.pair_list else None
    gt_root = Path(args.gt_root) if args.gt_root else None

    items: List[Tuple[Path, Optional[Path]]] = []
    if pair_list and pair_list.exists():
        pairs = parse_pair_list(pair_list)
        for lq, gt in pairs:
            items.append((lq, gt))
    else:
        lq_list = list_images(Path(args.input))
        for lq in lq_list:
            gt = resolve_gt_by_stem(lq, gt_root) if gt_root else None
            items.append((lq, gt))

    if args.max_images and args.max_images > 0:
        items = items[: args.max_images]

    print(f"[INFO] total images: {len(items)}")

    # ---- 输出子目录 ----
    dir_base = out_dir / "base"
    dir_s1 = out_dir / "stage1"
    dir_s2_out = out_dir / "stage2_out"
    dir_s2_res = out_dir / "stage2_res"
    dir_concat = out_dir / "concat"

    if args.save_base:
        ensure_dir(dir_base)
    if args.save_stage_images:
        ensure_dir(dir_s1)
        ensure_dir(dir_s2_out)
        ensure_dir(dir_s2_res)
    if args.save_concat:
        ensure_dir(dir_concat)

    # ---- CSV header ----
    per_image_header = [
        "run_name",
        "lq_path", "gt_path", "stem",
        "order", "stage1_group", "stage2_group", "stage2_source",
        "embedder", "embedder_tag", "sim_metric",
        "topk1", "topk2", "temperature1", "temperature2",
        "stage1_min_margin", "stage2_min_margin",
        "stage1_topk", "stage1_scores", "stage1_margin", "stage1_uncertain", "stage1_reason",
        "stage2_out_topk", "stage2_out_scores", "stage2_out_margin", "stage2_out_uncertain", "stage2_out_reason",
        "stage2_res_topk", "stage2_res_scores", "stage2_res_margin", "stage2_res_uncertain", "stage2_res_reason",
        "psnr_lq", "ssim_lq",
        "psnr_base", "ssim_base",
        "psnr_stage1", "ssim_stage1",
        "psnr_stage2_out", "ssim_stage2_out",
        "psnr_stage2_res", "ssim_stage2_res",
        "out_base_path", "out_s1_path", "out_s2_out_path", "out_s2_res_path", "out_concat_path",
    ]

    summary_header = [
        "run_name",
        "n_images", "n_with_gt",
        "order", "stage1_group", "stage2_group", "stage2_source",
        "embedder", "embedder_tag", "sim_metric",
        "topk1", "topk2", "temperature1", "temperature2",
        "stage1_min_margin", "stage2_min_margin",
        "mean_psnr_lq", "mean_ssim_lq",
        "mean_psnr_base", "mean_ssim_base",
        "mean_psnr_stage1", "mean_ssim_stage1",
        "mean_psnr_stage2_out", "mean_ssim_stage2_out",
        "mean_psnr_stage2_res", "mean_ssim_stage2_res",
    ]

    # ---- run_name ----
    run_name = args.run_name.strip()
    if not run_name:
        run_name = f"cascade_{args.order}_{args.stage2_source}_{args.embedder}_{embedder_tag}_k{args.topk1}-{args.topk2}_T{args.temperature1}-{args.temperature2}"
    print(f"[INFO] run_name={run_name}")

    # ---- 统计 ----
    vals = {
        "psnr_lq": [], "ssim_lq": [],
        "psnr_base": [], "ssim_base": [],
        "psnr_s1": [], "ssim_s1": [],
        "psnr_s2o": [], "ssim_s2o": [],
        "psnr_s2r": [], "ssim_s2r": [],
    }
    n_with_gt = 0

    from lora_adapters.utils_merge import apply_weighted_lora, map_to_single_domain_keys  # type: ignore

    for idx, (lq_path, gt_path) in enumerate(items, 1):
        lq_path = Path(lq_path)
        gt_path = Path(gt_path) if gt_path else None
        stem = lq_path.stem

        # ---- load LQ ----
        img_lq = Image.open(lq_path).convert("RGB")
        x = tfm_to_tensor(img_lq).unsqueeze(0).to(device)  # [1,3,H,W]

        # ---- base ----
        zero_single_lora(net, "_Single")
        y_base = forward_restore(net, x)

        # ---- Stage1 路由：默认用原始 LQ 路由 ----
        v1 = emb.embed_image(img_lq)
        thr1 = args.stage1_min_margin if args.stage1_min_margin >= 0 else None
        r1 = route_with_margin(
            orch=orch1, emb=v1, top_k=args.topk1, temperature=args.temperature1,
            min_margin=(thr1 if thr1 is not None else -1e9),
            stage_name="stage1",
        )
        stage1_skip = (thr1 is not None and r1.uncertain)

        if stage1_skip:
            y1 = y_base
        else:
            merged1 = orch1.build_weighted_lora(r1.topk)
            merged1 = map_to_single_domain_keys(merged1, target_domain_name="_Single")
            zero_single_lora(net, "_Single")
            apply_weighted_lora(net, merged1)
            y1 = forward_restore(net, x)

        # ---- Stage2：输入始终是 y1；路由输入由 out/residual 控制 ----
        y2_out = None
        y2_res = None
        r2_out = RouteInfo(metric=args.sim_metric, topk=[], scores=[], margin=None, uncertain=False, reason="")
        r2_res = RouteInfo(metric=args.sim_metric, topk=[], scores=[], margin=None, uncertain=False, reason="")

        thr2 = args.stage2_min_margin if args.stage2_min_margin >= 0 else None

        def do_stage2(route_img: Image.Image, tag: str) -> Tuple[torch.Tensor, RouteInfo]:
            v2 = emb.embed_image(route_img)
            rr = route_with_margin(
                orch=orch2, emb=v2, top_k=args.topk2, temperature=args.temperature2,
                min_margin=(thr2 if thr2 is not None else -1e9),
                stage_name=f"stage2_{tag}",
            )
            if thr2 is not None and rr.uncertain:
                # 跳过 stage2：直接返回 y1
                return y1, rr

            merged2 = orch2.build_weighted_lora(rr.topk)
            merged2 = map_to_single_domain_keys(merged2, target_domain_name="_Single")
            zero_single_lora(net, "_Single")
            apply_weighted_lora(net, merged2)
            # stage2 输入是 y1
            y2 = forward_restore(net, y1)
            return y2, rr

        if args.stage2_source in ("out", "both"):
            img_route_out = tensor_to_pil01(y1)
            y2_out, r2_out = do_stage2(img_route_out, tag="out")

        if args.stage2_source in ("residual", "both"):
            if args.residual_mode == "abs":
                r = (x - y1).abs()
            else:
                r = (x - y1)
            r = r.clamp(0, 1)
            img_route_res = tensor_to_pil01(r)
            y2_res, r2_res = do_stage2(img_route_res, tag="res")

        # ---- 选择 concat 的 stage2 展示：out 优先，否则 res，否则 y1 ----
        show_s2 = None
        if y2_out is not None:
            show_s2 = y2_out
        elif y2_res is not None:
            show_s2 = y2_res
        else:
            show_s2 = y1

        # ---- GT & metrics ----
        psnr_lq = ssim_lq = None
        psnr_base = ssim_base = None
        psnr_s1 = ssim_s1 = None
        psnr_s2o = ssim_s2o = None
        psnr_s2r = ssim_s2r = None

        if gt_path and gt_path.exists():
            img_gt = Image.open(gt_path).convert("RGB")
            gt = tfm_to_tensor(img_gt).unsqueeze(0).to(device)

            # 统一裁剪到同尺寸（保险）
            H = min(gt.shape[-2], x.shape[-2])
            W = min(gt.shape[-1], x.shape[-1])
            gt_ = gt[:, :, :H, :W]
            x_ = x[:, :, :H, :W]
            yb_ = y_base[:, :, :H, :W]
            y1_ = y1[:, :, :H, :W]
            if y2_out is not None:
                y2o_ = y2_out[:, :, :H, :W]
            else:
                y2o_ = None
            if y2_res is not None:
                y2r_ = y2_res[:, :, :H, :W]
            else:
                y2r_ = None

            psnr_lq, ssim_lq = tensor_psnr(x_, gt_), tensor_ssim(x_, gt_)
            psnr_base, ssim_base = tensor_psnr(yb_, gt_), tensor_ssim(yb_, gt_)
            psnr_s1, ssim_s1 = tensor_psnr(y1_, gt_), tensor_ssim(y1_, gt_)

            if y2o_ is not None:
                psnr_s2o, ssim_s2o = tensor_psnr(y2o_, gt_), tensor_ssim(y2o_, gt_)
            if y2r_ is not None:
                psnr_s2r, ssim_s2r = tensor_psnr(y2r_, gt_), tensor_ssim(y2r_, gt_)

            vals["psnr_lq"].append(psnr_lq); vals["ssim_lq"].append(ssim_lq)
            vals["psnr_base"].append(psnr_base); vals["ssim_base"].append(ssim_base)
            vals["psnr_s1"].append(psnr_s1); vals["ssim_s1"].append(ssim_s1)
            if psnr_s2o is not None: vals["psnr_s2o"].append(psnr_s2o)
            if ssim_s2o is not None: vals["ssim_s2o"].append(ssim_s2o)
            if psnr_s2r is not None: vals["psnr_s2r"].append(psnr_s2r)
            if ssim_s2r is not None: vals["ssim_s2r"].append(ssim_s2r)

            n_with_gt += 1

        # ---- 保存图像 ----
        out_base_path = out_s1_path = out_s2_out_path = out_s2_res_path = out_concat_path = ""

        if args.save_base:
            out_base_path = str((dir_base / f"{stem}_base.png").as_posix())
            from lora_adapters.utils import save_image  # type: ignore
            save_image(y_base, out_base_path)

        if args.save_stage_images:
            from lora_adapters.utils import save_image  # type: ignore

            s1_tag = r1.topk[0][0] if r1.topk else "none"
            out_s1_path = str((dir_s1 / f"{stem}_s1_{stage1_name}_{s1_tag}.png").as_posix())
            save_image(y1, out_s1_path)

            if y2_out is not None:
                s2o_tag = r2_out.topk[0][0] if r2_out.topk else "none"
                out_s2_out_path = str((dir_s2_out / f"{stem}_s2_{stage2_name}_out_{s2o_tag}.png").as_posix())
                save_image(y2_out, out_s2_out_path)

            if y2_res is not None:
                s2r_tag = r2_res.topk[0][0] if r2_res.topk else "none"
                out_s2_res_path = str((dir_s2_res / f"{stem}_s2_{stage2_name}_res_{s2r_tag}.png").as_posix())
                save_image(y2_res, out_s2_res_path)

        if args.save_concat:
            from lora_adapters.utils import save_image  # type: ignore

            parts = [x, y_base, y1]
            if args.stage2_source == "both":
                if y2_out is not None:
                    parts.append(y2_out)
                if y2_res is not None:
                    parts.append(y2_res)
            else:
                parts.append(show_s2)

            concat = torch.cat(parts, dim=3)
            out_concat_path = str((dir_concat / f"{stem}_concat.png").as_posix())
            save_image(concat, out_concat_path)

        # ---- 写 per-image CSV ----
        row = {
            "run_name": run_name,
            "lq_path": str(lq_path),
            "gt_path": str(gt_path) if gt_path else "",
            "stem": stem,
            "order": args.order,
            "stage1_group": stage1_name,
            "stage2_group": stage2_name,
            "stage2_source": args.stage2_source,
            "embedder": args.embedder,
            "embedder_tag": embedder_tag,
            "sim_metric": args.sim_metric,
            "topk1": args.topk1,
            "topk2": args.topk2,
            "temperature1": args.temperature1,
            "temperature2": args.temperature2,
            "stage1_min_margin": args.stage1_min_margin,
            "stage2_min_margin": args.stage2_min_margin,
            "stage1_topk": json.dumps(r1.topk, ensure_ascii=False),
            "stage1_scores": json.dumps(r1.scores, ensure_ascii=False),
            "stage1_margin": (None if r1.margin is None else float(r1.margin)),
            "stage1_uncertain": bool(stage1_skip),
            "stage1_reason": r1.reason if stage1_skip else "",
            "stage2_out_topk": json.dumps(r2_out.topk, ensure_ascii=False),
            "stage2_out_scores": json.dumps(r2_out.scores, ensure_ascii=False),
            "stage2_out_margin": (None if r2_out.margin is None else float(r2_out.margin)),
            "stage2_out_uncertain": bool(thr2 is not None and r2_out.uncertain),
            "stage2_out_reason": r2_out.reason if (thr2 is not None and r2_out.uncertain) else "",
            "stage2_res_topk": json.dumps(r2_res.topk, ensure_ascii=False),
            "stage2_res_scores": json.dumps(r2_res.scores, ensure_ascii=False),
            "stage2_res_margin": (None if r2_res.margin is None else float(r2_res.margin)),
            "stage2_res_uncertain": bool(thr2 is not None and r2_res.uncertain),
            "stage2_res_reason": r2_res.reason if (thr2 is not None and r2_res.uncertain) else "",
            "psnr_lq": psnr_lq, "ssim_lq": ssim_lq,
            "psnr_base": psnr_base, "ssim_base": ssim_base,
            "psnr_stage1": psnr_s1, "ssim_stage1": ssim_s1,
            "psnr_stage2_out": psnr_s2o, "ssim_stage2_out": ssim_s2o,
            "psnr_stage2_res": psnr_s2r, "ssim_stage2_res": ssim_s2r,
            "out_base_path": out_base_path,
            "out_s1_path": out_s1_path,
            "out_s2_out_path": out_s2_out_path,
            "out_s2_res_path": out_s2_res_path,
            "out_concat_path": out_concat_path,
        }
        append_csv_row(metrics_csv, per_image_header, row)

        # ---- console 简要日志 ----
        msg = f"[{idx}/{len(items)}] {stem}"
        if psnr_base is not None:
            msg += f" | base {psnr_base:.2f}/{ssim_base:.4f} | s1 {psnr_s1:.2f}/{ssim_s1:.4f}"
            if psnr_s2o is not None:
                msg += f" | s2_out {psnr_s2o:.2f}/{ssim_s2o:.4f}"
            if psnr_s2r is not None:
                msg += f" | s2_res {psnr_s2r:.2f}/{ssim_s2r:.4f}"
        msg += f" | s1_top1={r1.topk[0][0] if r1.topk else 'none'}"
        if args.stage2_source in ("out", "both") and r2_out.topk:
            msg += f" | s2_out_top1={r2_out.topk[0][0]}"
        if args.stage2_source in ("residual", "both") and r2_res.topk:
            msg += f" | s2_res_top1={r2_res.topk[0][0]}"
        print(msg)

    # ---- summary ----
    def mean(xs: List[float]) -> Optional[float]:
        xs = [float(v) for v in xs if v is not None]
        if not xs:
            return None
        return float(sum(xs) / len(xs))

    summary_row = {
        "run_name": run_name,
        "n_images": len(items),
        "n_with_gt": n_with_gt,
        "order": args.order,
        "stage1_group": stage1_name,
        "stage2_group": stage2_name,
        "stage2_source": args.stage2_source,
        "embedder": args.embedder,
        "embedder_tag": embedder_tag,
        "sim_metric": args.sim_metric,
        "topk1": args.topk1,
        "topk2": args.topk2,
        "temperature1": args.temperature1,
        "temperature2": args.temperature2,
        "stage1_min_margin": args.stage1_min_margin,
        "stage2_min_margin": args.stage2_min_margin,
        "mean_psnr_lq": mean(vals["psnr_lq"]),
        "mean_ssim_lq": mean(vals["ssim_lq"]),
        "mean_psnr_base": mean(vals["psnr_base"]),
        "mean_ssim_base": mean(vals["ssim_base"]),
        "mean_psnr_stage1": mean(vals["psnr_s1"]),
        "mean_ssim_stage1": mean(vals["ssim_s1"]),
        "mean_psnr_stage2_out": mean(vals["psnr_s2o"]),
        "mean_ssim_stage2_out": mean(vals["ssim_s2o"]),
        "mean_psnr_stage2_res": mean(vals["psnr_s2r"]),
        "mean_ssim_stage2_res": mean(vals["ssim_s2r"]),
    }
    append_csv_row(summary_csv, summary_header, summary_row)

    print(f"[DONE] metrics_csv={metrics_csv}")
    print(f"[DONE] summary_csv={summary_csv}")


if __name__ == "__main__":
    main()

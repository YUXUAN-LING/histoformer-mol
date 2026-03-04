# lora_adapters/cascade.py
# 顺序修复 demo：先 haze2 LoRA，再 rain3 LoRA
# 同时对比 base 模型，并计算 PSNR / SSIM 指标

import os
import argparse
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from lora_adapters.utils import build_histoformer, save_image
from lora_adapters.inject_lora import inject_lora
from lora_adapters.utils_merge import map_to_single_domain_keys, apply_weighted_lora


# ====================== IQA 指标：PSNR / SSIM ======================

def tensor_psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> float:
    """
    pred, gt: [B,3,H,W], in [0,1]
    """
    mse = torch.mean((pred - gt) ** 2).clamp(min=1e-10)
    psnr = 10.0 * torch.log10((max_val ** 2) / mse)
    return psnr.item()


def _gaussian_kernel(window_size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g[:, None] @ g[None, :]
    kernel_2d = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return kernel_2d


def tensor_ssim(x: torch.Tensor, y: torch.Tensor,
                window_size: int = 11, sigma: float = 1.5,
                K1: float = 0.01, K2: float = 0.03) -> float:
    """
    x,y: [B,3,H,W], in [0,1]
    """
    assert x.shape == y.shape, "tensor_ssim: x,y shape mismatch"
    B, C, H, W = x.shape
    device = x.device

    kernel = _gaussian_kernel(window_size, sigma, C, device=device)

    mu_x = F.conv2d(x, kernel, padding=window_size // 2, groups=C)
    mu_y = F.conv2d(y, kernel, padding=window_size // 2, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=window_size // 2, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=window_size // 2, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=window_size // 2, groups=C) - mu_xy

    L = 1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))

    return ssim_map.mean().item()


# ====================== LoRA 工具函数 ======================

def zero_single_lora(net: torch.nn.Module):
    """
    将 .lora_down._Single.weight / .lora_up._Single.weight 置零，
    确保上一张图 / 上一次域的残差不会影响后续。
    """
    with torch.no_grad():
        for name, p in net.named_parameters():
            if (".lora_down._Single.weight" in name) or (".lora_up._Single.weight" in name):
                p.zero_()


def normalize_lora_keys(sd: dict) -> dict:
    """
    将可能带域名的键（...lora_down.<Domain>.weight）规范为域无关键（...lora_down.weight）
    逻辑和 DomainOrchestrator._normalize_keys 一致，
    这样不同域 LoRA ckpt 可以被统一映射到 _Single 分支。
    """
    out = {}
    for k, v in sd.items():
        if "lora_" not in k:
            continue
        parts = k.split(".")
        # 期望 [..., 'lora_down', '<maybeDomain>', 'weight']
        if (
            len(parts) >= 2
            and parts[-1] == 'weight'
            and parts[-3].startswith('lora_')
            and parts[-2] not in ('weight', 'bias')
        ):
            # 删除域名 token
            k2 = ".".join(parts[:-2] + parts[-1:])
            out[k2] = v
        elif parts[-1] == 'weight':
            out[k] = v
    return out


def load_domain_lora_to_single(
    net: torch.nn.Module,
    domain_name: str,
    loradb_root: str | Path = "weights/lora"
):
    """
    从 weights/lora/<domain_name> 下加载对应 LoRA .pth，
    归一化键名并映射到 _Single 分支，然后注入网络。
    """
    loradb_root = Path(loradb_root)
    dom_dir = loradb_root / domain_name

    if not dom_dir.is_dir():
        raise FileNotFoundError(f"[load_domain_lora_to_single] LoRA 目录不存在: {dom_dir}")

    # 优先使用 <domain>.pth，否则取该目录下第一个 .pth
    cand = dom_dir / f"{domain_name}.pth"
    if not cand.exists():
        pths = list(dom_dir.glob("*.pth"))
        if not pths:
            raise FileNotFoundError(f"[load_domain_lora_to_single] {dom_dir} 下没有 .pth 文件")
        cand = pths[0]

    print(f"[LoRA] 加载域 {domain_name} 的 LoRA 权重: {cand}")
    sd = torch.load(cand, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]

    sd_norm = normalize_lora_keys(sd)  # 去掉原域名
    # 映射到 _Single 分支：...lora_down.weight -> ...lora_down._Single.weight
    single_sd = map_to_single_domain_keys(sd_norm, target_domain_name="_Single")
    apply_weighted_lora(net, single_sd)


# ====================== pad / unpad ======================

def pad_to_multiple(x: torch.Tensor, factor: int = 8) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    将 [B,3,H,W] pad 到 H,W 都是 factor 的倍数，返回 padded 张量和 pad 信息。
    pad 信息格式为 (left, right, top, bottom)
    """
    b, c, h, w = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    # F.pad 的顺序是 (left, right, top, bottom)
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x_pad, (0, pad_w, 0, pad_h)


def unpad(x: torch.Tensor, pad: Tuple[int, int, int, int], orig_h: int, orig_w: int) -> torch.Tensor:
    """
    从 pad 后的张量中裁剪回原始 H,W。
    这里我们其实只需要根据 orig_h, orig_w 裁剪即可。
    """
    return x[..., :orig_h, :orig_w]


# ====================== 主逻辑：顺序修复 + 指标评估 ======================

def run_cascade(
    base_ckpt: str,
    yaml_file: str | None,
    input_path: str,
    output_dir: str,
    loradb_root: str = "weights/lora",
    gt_root: str | None = None,
    rank: int = 16,
    alpha: float = 16.0,
    device: str = "cuda",
    metrics_csv: str | None = None,
):
    """
    顺序修复主函数：
      1) 构建 base Histoformer + 注入 _Single LoRA 分支
      2) 对每张图依次执行：
         base: 不加载任何 LoRA
         haze2: 只加载 haze2 LoRA
         haze2->rain3: 顺序加载 haze2 / rain3
      3) 若提供 GT，则计算：
         LQ / base / haze2 / cascade 的 PSNR / SSIM
      4) 保存 [LQ | base | haze2 | cascade] 拼接结果
      5) 汇总打印平均指标，可选写入 CSV
    """
    device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    print("[INFO] 构建 base Histoformer...")
    net = build_histoformer(weights=base_ckpt, yaml_file=yaml_file).to(device)
    net = inject_lora(
        net,
        rank=rank,
        domain_list=["_Single"],
        alpha=alpha,
        target_names=None,
        patterns=None,
    )
    net.eval().to(device)

    tfm = T.ToTensor()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集待处理图像
    input_path = Path(input_path)
    if input_path.is_dir():
        img_files = sorted([
            p for p in input_path.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
        ])
    else:
        img_files = [input_path]

    print(f"[INFO] 共 {len(img_files)} 张图像待处理")
    has_gt = gt_root is not None
    metrics_list: List[Dict] = []

    for img_path in img_files:
        print(f"\n[PROC] {img_path}")
        img = Image.open(img_path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)  # [1,3,H,W]
        _, _, h, w = x.shape

        # ===== Base 输出（不加载 LoRA） =====
        zero_single_lora(net)
        x_pad, pad_info = pad_to_multiple(x, factor=8)
        with torch.no_grad():
            y_base = net(x_pad)
        y_base = unpad(y_base, pad_info, h, w)

        # ===== Step 1: 使用 第一个 LoRA 修复 =====
        zero_single_lora(net)
        load_domain_lora_to_single(net, "haze2", loradb_root=loradb_root)

        x_pad, pad_info = pad_to_multiple(x, factor=8)
        with torch.no_grad():
            y_haze = net(x_pad)
        y_haze = unpad(y_haze, pad_info, h, w)

        # ===== Step 2: 使用 第二个 LoRA 修复 (输入是 y_haze) =====
        zero_single_lora(net)
        load_domain_lora_to_single(net, "snow1", loradb_root=loradb_root)

        y_in = y_haze
        _, _, h2, w2 = y_in.shape
        y_in_pad, pad_info2 = pad_to_multiple(y_in, factor=8)
        with torch.no_grad():
            y_final = net(y_in_pad)
        y_final = unpad(y_final, pad_info2, h2, w2)

        # ===== 拼接并保存：LQ | base | haze2 | haze2→rain3 =====
        concat = torch.cat([x, y_base, y_haze, y_final], dim=3)  # [1,3,H,W*4]
        out_name = img_path.name
        out_path = output_dir / f"cascade_haze2_rain3_{out_name}"
        save_image(concat, str(out_path))
        print(f"[SAVE] {out_path}")

        # ===== 若有 GT，则计算指标 =====
        metric_row = {
            "name": img_path.name,
            "psnr_lq": None,
            "psnr_base": None,
            "psnr_haze2": None,
            "psnr_cascade": None,
            "ssim_lq": None,
            "ssim_base": None,
            "ssim_haze2": None,
            "ssim_cascade": None,
        }

        if has_gt:
            gt_path = Path(gt_root) / img_path.name
            if gt_path.is_file():
                gt_img = Image.open(gt_path).convert("RGB")
                gt = tfm(gt_img).unsqueeze(0).to(device)
                # 若尺寸不一致，resize GT 到 LQ 尺寸
                if gt.shape[2] != h or gt.shape[3] != w:
                    gt = F.interpolate(gt, size=(h, w), mode='bilinear', align_corners=False)

                psnr_lq = tensor_psnr(x, gt)
                psnr_base = tensor_psnr(y_base, gt)
                psnr_haze = tensor_psnr(y_haze, gt)
                psnr_cascade = tensor_psnr(y_final, gt)

                ssim_lq = tensor_ssim(x, gt)
                ssim_base = tensor_ssim(y_base, gt)
                ssim_haze = tensor_ssim(y_haze, gt)
                ssim_cascade = tensor_ssim(y_final, gt)

                metric_row.update({
                    "psnr_lq": psnr_lq,
                    "psnr_base": psnr_base,
                    "psnr_haze2": psnr_haze,
                    "psnr_cascade": psnr_cascade,
                    "ssim_lq": ssim_lq,
                    "ssim_base": ssim_base,
                    "ssim_haze2": ssim_haze,
                    "ssim_cascade": ssim_cascade,
                })

                print(f"[METRICS] {img_path.name} | "
                      f"PSNR: LQ={psnr_lq:.3f}, base={psnr_base:.3f}, mid={psnr_haze:.3f}, cascade={psnr_cascade:.3f} | "
                      f"SSIM: LQ={ssim_lq:.4f}, base={ssim_base:.4f}, mid={ssim_haze:.4f}, cascade={ssim_cascade:.4f}")
            else:
                print(f"[WARN] GT not found for {img_path.name}, expected: {gt_path}")

        metrics_list.append(metric_row)

    # ===== 汇总平均指标 =====
    if has_gt and len(metrics_list) > 0:
        valid = [m for m in metrics_list if m["psnr_lq"] is not None]
        if len(valid) > 0:
            import numpy as np
            mean_psnr_lq = float(np.mean([m["psnr_lq"] for m in valid]))
            mean_psnr_base = float(np.mean([m["psnr_base"] for m in valid]))
            mean_psnr_haze2 = float(np.mean([m["psnr_haze2"] for m in valid]))
            mean_psnr_cascade = float(np.mean([m["psnr_cascade"] for m in valid]))

            mean_ssim_lq = float(np.mean([m["ssim_lq"] for m in valid]))
            mean_ssim_base = float(np.mean([m["ssim_base"] for m in valid]))
            mean_ssim_haze2 = float(np.mean([m["ssim_haze2"] for m in valid]))
            mean_ssim_cascade = float(np.mean([m["ssim_cascade"] for m in valid]))

            print("\n===== MEAN METRICS (only images with GT) =====")
            print(f"PSNR  LQ      : {mean_psnr_lq:.3f}")
            print(f"PSNR  base    : {mean_psnr_base:.3f}")
            print(f"PSNR  mid   : {mean_psnr_haze2:.3f}")
            print(f"PSNR  cascade : {mean_psnr_cascade:.3f}")
            print(f"SSIM  LQ      : {mean_ssim_lq:.4f}")
            print(f"SSIM  base    : {mean_ssim_base:.4f}")
            print(f"SSIM  mid   : {mean_ssim_haze2:.4f}")
            print(f"SSIM  cascade : {mean_ssim_cascade:.4f}")

            # 可选：写入 CSV
            if metrics_csv is not None:
                import csv
                csv_path = Path(metrics_csv)
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                fieldnames = [
                    "name",
                    "psnr_lq", "psnr_base", "psnr_mid", "psnr_cascade",
                    "ssim_lq", "ssim_base", "ssim_mid", "ssim_cascade",
                ]
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in valid:
                        writer.writerow(row)
                    writer.writerow({
                        "name": "__mean__",
                        "psnr_lq": mean_psnr_lq,
                        "psnr_base": mean_psnr_base,
                        "psnr_mid": mean_psnr_haze2,
                        "psnr_cascade": mean_psnr_cascade,
                        "ssim_lq": mean_ssim_lq,
                        "ssim_base": mean_ssim_base,
                        "ssim_mid": mean_ssim_haze2,
                        "ssim_cascade": mean_ssim_cascade,
                    })
                print(f"[CSV] metrics saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cascade restoration: first haze2 LoRA, then rain3 LoRA, with PSNR/SSIM evaluation"
    )
    parser.add_argument("--base_ckpt", type=str, required=True,
                        help="Histoformer base 权重路径，例如 pretrained_models/histoformer_base.pth")
    parser.add_argument("--yaml", type=str, default=None,
                        help="Histoformer 配置 YAML（若 build_histoformer 需要）")
    parser.add_argument("--input", type=str, required=True,
                        help="输入 LQ 图像（单张或目录）")
    parser.add_argument("--output", type=str, required=True,
                        help="输出目录，用于保存拼接结果")
    parser.add_argument("--loradb_root", type=str, default="weights/lora",
                        help="LoRA 权重根目录，例如 weights/lora")
    parser.add_argument("--gt_root", type=str, default=None,
                        help="GT 清晰图根目录，会按文件名匹配")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--metrics_csv", type=str, default=None,
                        help="可选：将每张图和平均指标写入 CSV 文件")

    args = parser.parse_args()

    run_cascade(
        base_ckpt=args.base_ckpt,
        yaml_file=args.yaml,
        input_path=args.input,
        output_dir=args.output,
        loradb_root=args.loradb_root,
        gt_root=args.gt_root,
        rank=args.rank,
        alpha=args.alpha,
        device=args.device,
        metrics_csv=args.metrics_csv,
    )


if __name__ == "__main__":
    main()

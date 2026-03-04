# -*- coding: utf-8 -*-
"""
比较 base 模型 vs base+LoRA 的批量推理脚本 + IQA 评估 (PSNR / SSIM)

用法示例 1（用 txt 测试对）：
python -m lora_adapters.infer_folder_lora_compare \
  --pair_list data/blur/test_list_blur.txt \
  --output_dir results/blur_lora_compare \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --lora_ckpt weights/lora/blur/lora_best.pth \
  --domain blur \
  --domains rain,snow,fog,blur \
  --rank 32 \
  --alpha 32 \
  --device cuda

用法示例 2（文件夹模式）：
python -m lora_adapters.infer_folder_lora_compare \
  --input_dir data/blur/test_lq \
  --gt_dir    data/blur/test_gt \
  --output_dir results/blur_lora_compare \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --lora_ckpt weights/lora/blur/lora_best.pth \
  --domain blur \
  --domains rain,snow,fog,blur \
  --rank 32 \
  --alpha 32 \
  --device cuda

输出：
- xxx_base.png      : base 模型输出
- xxx_lora.png      : base+LoRA 输出
- xxx_compare3.png  : 左=LQ，中=base，右=LoRA 三图对比
- metrics_compare.txt: 每张图的 base/LoRA PSNR/SSIM 及全局均值
"""

import os
import argparse
from typing import List, Tuple, Optional

import torch
from PIL import Image
from torchvision import transforms as T
import torch.nn.functional as Fnn

from lora_adapters.inject_lora import inject_lora, iter_lora_modules
from lora_adapters.utils import build_histoformer, load_image, save_image


# ====================== 工具函数 ======================

def list_images(root: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in exts:
            files.append(p)
    files.sort()
    return files


def read_pairs(list_file: str) -> List[Tuple[str, str]]:
    """读取 txt：每行 <lq_path> <gt_path>"""
    pairs = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            lq, gt = parts[0], parts[1]
            pairs.append((lq, gt))
    return pairs


def set_single_domain(model: torch.nn.Module, domain: str):
    """将所有 LoRA 模块的 domain_weights 设为单一域 one-hot。"""
    for m in iter_lora_modules(model):
        if domain not in m.domain_list:
            raise ValueError(f"domain '{domain}' not in module.domain_list={m.domain_list}")
        import torch as _torch
        idx = m.domain_list.index(domain)
        w = _torch.zeros(len(m.domain_list))
        w[idx] = 1.0
        m.domain_weights = w


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

    mu_x = Fnn.conv2d(x, kernel, padding=window_size // 2, groups=C)
    mu_y = Fnn.conv2d(y, kernel, padding=window_size // 2, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = Fnn.conv2d(x * x, kernel, padding=window_size // 2, groups=C) - mu_x2
    sigma_y2 = Fnn.conv2d(y * y, kernel, padding=window_size // 2, groups=C) - mu_y2
    sigma_xy = Fnn.conv2d(x * y, kernel, padding=window_size // 2, groups=C) - mu_xy

    L = 1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))

    return ssim_map.mean().item()


# ====================== tile 推理（防 OOM） ======================

def run_once(model: torch.nn.Module, x: torch.Tensor, device: str):
    """
    不切 tile 的单次推理：只对齐到 8 的倍数，然后裁回来。
    """
    B, C, H, W = x.shape
    new_H = (H + 7) // 8 * 8
    new_W = (W + 7) // 8 * 8
    pad_bottom = new_H - H
    pad_right = new_W - W

    if pad_bottom > 0 or pad_right > 0:
        x_pad = Fnn.pad(x, (0, pad_right, 0, pad_bottom), mode="reflect")
    else:
        x_pad = x

    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
        out = model(x_pad)

    out = out[:, :, :H, :W]
    return out


def tile_inference(model: torch.nn.Module, x: torch.Tensor,
                   tile: int = 640, overlap: int = 32, device: str = "cuda"):
    """
    对大图做分块推理，避免 OOM。
    - 每块大小不超过 tile×tile
    - 只把每块 pad 到 8 的倍数，而不是 pad 到 tile
    - 简单加权平均融合重叠区域
    """
    B, C, H, W = x.shape

    if H <= tile and W <= tile:
        return run_once(model, x, device)

    stride = tile - overlap
    ys = list(range(0, H, stride))
    xs = list(range(0, W, stride))

    if ys[-1] + tile < H:
        ys.append(max(H - tile, 0))
    if xs[-1] + tile < W:
        xs.append(max(W - tile, 0))

    ys = sorted(set(ys))
    xs = sorted(set(xs))

    out_full = torch.zeros((1, C, H, W), device=x.device, dtype=torch.float32)
    weight = torch.zeros((1, 1, H, W), device=x.device, dtype=torch.float32)

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)

            patch = x[:, :, y0:y1, x0:x1]
            ph, pw = patch.shape[2], patch.shape[3]

            new_ph = (ph + 7) // 8 * 8
            new_pw = (pw + 7) // 8 * 8
            pad_bottom = new_ph - ph
            pad_right = new_pw - pw

            if pad_bottom > 0 or pad_right > 0:
                patch_pad = Fnn.pad(patch, (0, pad_right, 0, pad_bottom), mode="reflect")
            else:
                patch_pad = patch

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out_patch = model(patch_pad)

            out_patch = out_patch[:, :, :ph, :pw]

            out_full[:, :, y0:y1, x0:x1] += out_patch
            weight[:, :, y0:y1, x0:x1] += 1.0

    weight = torch.clamp(weight, min=1.0)
    out_full = out_full / weight
    return out_full


# ============================== 主流程 ==============================

def main():
    ap = argparse.ArgumentParser()
    # 数据来源
    ap.add_argument("--input_dir", help="待处理 LQ 图像文件夹（文件夹模式）")
    ap.add_argument("--gt_dir", default=None, help="GT 图像文件夹（可选，用于指标）")
    ap.add_argument("--pair_list", default=None,
                    help="测试对 txt，每行: <lq_path> <gt_path>。若提供则优先使用此模式。")

    # 模型 & LoRA
    ap.add_argument("--output_dir", required=True, help="输出结果文件夹")
    ap.add_argument("--base_ckpt", required=True, help="Histoformer base 权重路径")
    ap.add_argument("--lora_ckpt", required=True, help="单域 LoRA 权重（如 lora_best.pth）")
    ap.add_argument("--domain", required=True, help="本次推理使用的域名，如 rain")
    ap.add_argument("--domains", required=True, help="训练时的所有域名称列表，如 'rain,snow,fog'")
    ap.add_argument("--rank", type=int, default=8, help="LoRA rank（需与训练时一致）")
    ap.add_argument("--alpha", type=float, default=8, help="LoRA alpha（需与训练时一致）")
    ap.add_argument("--device", default="cuda", help="cuda 或 cpu")

    # 日志
    ap.add_argument("--log_file", default=None,
                    help="比较指标日志文件，默认 output_dir/metrics_compare.txt")

    args = ap.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # -------- 解析数据来源 --------
    pairs: List[Tuple[str, Optional[str]]] = []
    if args.pair_list is not None:
        print("[INFO] using pair_list:", args.pair_list)
        if not os.path.isfile(args.pair_list):
            raise FileNotFoundError(f"pair_list not found: {args.pair_list}")
        raw_pairs = read_pairs(args.pair_list)
        for lq, gt in raw_pairs:
            pairs.append((lq, gt))
        print(f"[INFO] loaded {len(pairs)} pairs from list.")
    else:
        if args.input_dir is None:
            raise ValueError("Either --pair_list or --input_dir must be provided.")
        print("[INFO] listing images from", args.input_dir)
        img_paths = list_images(args.input_dir)
        print(f"[INFO] found {len(img_paths)} images")
        if len(img_paths) == 0:
            print("[WARN] no images, exit.")
            return

        if args.gt_dir is not None:
            for p in img_paths:
                name = os.path.basename(p)
                gt_path = os.path.join(args.gt_dir, name)
                pairs.append((p, gt_path))
        else:
            for p in img_paths:
                pairs.append((p, None))

    if len(pairs) == 0:
        print("[WARN] no input images, exit.")
        return

    # -------- 构建 base & LoRA 模型 --------
    print("[INFO] building Histoformer (base)...")
    net_base = build_histoformer(
        weights=args.base_ckpt,
        yaml_file="lora_adapters/configs/histoformer_mol.yaml",
    )

    print("[INFO] building Histoformer (LoRA)...")
    net_lora = build_histoformer(
        weights=args.base_ckpt,
        yaml_file="lora_adapters/configs/histoformer_mol.yaml",
    )

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    net_lora = inject_lora(net_lora, rank=args.rank, domain_list=domains, alpha=args.alpha)

    print("[INFO] loading LoRA weights from", args.lora_ckpt)
    sd = torch.load(args.lora_ckpt, map_location="cpu")
    msd = net_lora.state_dict()
    msd.update(sd)
    net_lora.load_state_dict(msd, strict=False)

    set_single_domain(net_lora, args.domain)

    net_base.to(device).eval()
    net_lora.to(device).eval()

    to_tensor = T.ToTensor()

    # -------- 指标累积 --------
    psnr_base_sum = 0.0
    psnr_lora_sum = 0.0
    ssim_base_sum = 0.0
    ssim_lora_sum = 0.0
    metric_count = 0

    # 日志文件
    log_path = args.log_file or os.path.join(args.output_dir, "metrics_compare.txt")
    log_f = open(log_path, "w", encoding="utf-8")
    log_f.write("#name\tpsnr_base(dB)\tssim_base\tpsnr_lora(dB)\tssim_lora\td_psnr\td_ssim\n")

    # -------- 逐张推理 --------
    with torch.no_grad():
        for idx, (lq_path, gt_path) in enumerate(pairs):
            name = os.path.basename(lq_path)
            stem, _ = os.path.splitext(name)
            print(f"[{idx+1}/{len(pairs)}] processing {name} ...")

            # 读取 LQ
            x, (W, H) = load_image(lq_path)  # x:[1,3,H,W]
            x = x.to(device)

            B, C, h, w = x.shape
            new_h = (h // 8) * 8
            new_w = (w // 8) * 8
            if new_h == 0 or new_w == 0:
                print(f"  [WARN] image too small ({h}x{w}), skip.")
                continue

            if new_h != h or new_w != w:
                x = x[:, :, :new_h, :new_w]
                pil_lq = Image.open(lq_path).convert("RGB")
                pil_lq = pil_lq.crop((0, 0, new_w, new_h))
            else:
                pil_lq = Image.open(lq_path).convert("RGB")

            # base / lora 推理（tile 防 OOM）
            out_base = tile_inference(net_base, x, tile=640, overlap=32, device=device)
            out_lora = tile_inference(net_lora, x, tile=640, overlap=32, device=device)

            out_base_c = out_base.clamp(0.0, 1.0)
            out_lora_c = out_lora.clamp(0.0, 1.0)

            # --- 指标 ---
            psnr_b = psnr_l = ssim_b = ssim_l = None
            if gt_path is not None and os.path.isfile(gt_path):
                gt_pil = Image.open(gt_path).convert("RGB")
                if gt_pil.size != (new_w, new_h):
                    gt_pil = gt_pil.crop((0, 0, new_w, new_h))
                gt_t = to_tensor(gt_pil).unsqueeze(0).to(device)

                psnr_b = tensor_psnr(out_base_c, gt_t)
                psnr_l = tensor_psnr(out_lora_c, gt_t)
                ssim_b = tensor_ssim(out_base_c, gt_t)
                ssim_l = tensor_ssim(out_lora_c, gt_t)

                d_psnr = psnr_l - psnr_b
                d_ssim = ssim_l - ssim_b

                print(f"  base: PSNR={psnr_b:.2f} dB, SSIM={ssim_b:.4f}")
                print(f"  lora: PSNR={psnr_l:.2f} dB, SSIM={ssim_l:.4f}")
                print(f"  delta: dPSNR={d_psnr:+.2f} dB, dSSIM={d_ssim:+.4f}")

                psnr_base_sum += psnr_b
                psnr_lora_sum += psnr_l
                ssim_base_sum += ssim_b
                ssim_lora_sum += ssim_l
                metric_count += 1

                log_f.write(
                    f"{name}\t{psnr_b:.4f}\t{ssim_b:.6f}\t{psnr_l:.4f}\t{ssim_l:.6f}\t{d_psnr:+.4f}\t{d_ssim:+.6f}\n"
                )
            else:
                if gt_path is not None:
                    print(f"  [WARN] GT not found for {name}, expected: {gt_path}")
                log_f.write(f"{name}\tNA\tNA\tNA\tNA\tNA\tNA\n")

            # --- 保存 base & lora 输出 ---
            base_path = os.path.join(args.output_dir, f"{stem}_base.png")
            lora_path = os.path.join(args.output_dir, f"{stem}_lora.png")
            save_image(out_base_c, base_path)
            save_image(out_lora_c, lora_path)

            base_pil = Image.open(base_path).convert("RGB")
            lora_pil = Image.open(lora_path).convert("RGB")

            # --- 三图横向拼接：左=LQ，中=base，右=LoRA ---
            w0, h0 = pil_lq.size
            w1, h1 = base_pil.size
            w2, h2 = lora_pil.size
            h_min = min(h0, h1, h2)

            pil_lq_disp = pil_lq.resize((w0, h_min), Image.BICUBIC) if h0 != h_min else pil_lq
            base_disp = base_pil.resize((w1, h_min), Image.BICUBIC) if h1 != h_min else base_pil
            lora_disp = lora_pil.resize((w2, h_min), Image.BICUBIC) if h2 != h_min else lora_pil

            comp_w = w0 + w1 + w2
            comp = Image.new("RGB", (comp_w, h_min))
            x_offset = 0
            comp.paste(pil_lq_disp, (x_offset, 0)); x_offset += w0
            comp.paste(base_disp,   (x_offset, 0)); x_offset += w1
            comp.paste(lora_disp,   (x_offset, 0))

            comp_path = os.path.join(args.output_dir, f"{stem}_compare3.png")
            comp.save(comp_path)

    # -------- 统计整体均值 --------
    if metric_count > 0:
        mean_psnr_base = psnr_base_sum / metric_count
        mean_psnr_lora = psnr_lora_sum / metric_count
        mean_ssim_base = ssim_base_sum / metric_count
        mean_ssim_lora = ssim_lora_sum / metric_count
        d_psnr_mean = mean_psnr_lora - mean_psnr_base
        d_ssim_mean = mean_ssim_lora - mean_ssim_base

        print(f"[INFO] Average over {metric_count} images:")
        print(f"       base: PSNR={mean_psnr_base:.2f} dB, SSIM={mean_ssim_base:.4f}")
        print(f"       lora: PSNR={mean_psnr_lora:.2f} dB, SSIM={mean_ssim_lora:.4f}")
        print(f"       delta: dPSNR={d_psnr_mean:+.2f} dB, dSSIM={d_ssim_mean:+.4f}")

        log_f.write("\n# MEAN_BASE_PSNR\t%.4f\n" % mean_psnr_base)
        log_f.write("# MEAN_BASE_SSIM\t%.6f\n" % mean_ssim_base)
        log_f.write("# MEAN_LORA_PSNR\t%.4f\n" % mean_psnr_lora)
        log_f.write("# MEAN_LORA_SSIM\t%.6f\n" % mean_ssim_lora)
        log_f.write("# MEAN_dPSNR\t%+.4f\n" % d_psnr_mean)
        log_f.write("# MEAN_dSSIM\t%+.6f\n" % d_ssim_mean)
    else:
        print("[WARN] 没有成功计算任何一张图的 PSNR/SSIM。")
        log_f.write("\n# NO_VALID_METRICS\n")

    log_f.close()
    print(f"[INFO] metrics saved to: {log_path}")
    print("[INFO] done.")


if __name__ == "__main__":
    main()

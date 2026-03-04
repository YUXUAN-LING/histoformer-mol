# lora_adapters/infer_folder_lora_compare.py
# -*- coding: utf-8 -*-

import os
import math
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import amp
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as TV

from lora_adapters.utils import load_image, save_image, build_histoformer
from lora_adapters.inject_lora import inject_lora, iter_lora_modules


# =========================================
# 读 txt 配对列表
# 每行: lq_path gt_path
# =========================================
def read_pairs(txt_path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            lq, gt = parts[0], parts[1]
            pairs.append((lq, gt))
    return pairs


# =========================================
# 旧模式：列出文件夹图片（保留备用）
# =========================================
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


def set_single_domain(model: torch.nn.Module, domain: str):
    """将所有 LoRA 模块的 domain_weights 设为指定域的 one-hot。"""
    for m in iter_lora_modules(model):
        if domain not in m.domain_list:
            raise ValueError(f"domain '{domain}' not in module.domain_list={m.domain_list}")
        import torch as _torch
        idx = m.domain_list.index(domain)
        w = _torch.zeros(len(m.domain_list))
        w[idx] = 1.0
        m.domain_weights = w


# =========================================
# 指标：PSNR / SSIM
# =========================================
def calc_psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.clamp(0.0, 1.0)
    y = y.clamp(0.0, 1.0)
    mse = torch.mean((x - y) ** 2).item()
    if mse == 0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _gaussian_kernel(window_size: int, sigma: float, channels: int, device: torch.device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g[:, None] @ g[None, :]
    kernel_2d = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return kernel_2d


def calc_ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    x = x.clamp(0.0, 1.0)
    y = y.clamp(0.0, 1.0)
    _, C, _, _ = x.shape
    device = x.device

    kernel = _gaussian_kernel(window_size, sigma, C, device)

    mu_x = F.conv2d(x, kernel, padding=window_size // 2, groups=C)
    mu_y = F.conv2d(y, kernel, padding=window_size // 2, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=window_size // 2, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=window_size // 2, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=window_size // 2, groups=C) - mu_xy

    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )
    return ssim_map.mean().item()


# =========================================
# 大图 tile 推理
# =========================================
@torch.no_grad()
def run_once(model, x, device="cuda"):
    B, C, H, W = x.shape
    new_H = (H + 7) // 8 * 8
    new_W = (W + 7) // 8 * 8
    pad_bottom = new_H - H
    pad_right = new_W - W

    if pad_bottom > 0 or pad_right > 0:
        x = F.pad(x, (0, pad_right, 0, pad_bottom), mode="reflect")

    with amp.autocast(device_type="cuda", enabled=(device == "cuda")):
        out = model(x)

    return out[:, :, :H, :W]


@torch.no_grad()
def tile_inference(model, x, tile=640, overlap=32, device="cuda"):
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
                patch_pad = F.pad(patch, (0, pad_right, 0, pad_bottom), mode="reflect")
            else:
                patch_pad = patch

            with amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                out_patch = model(patch_pad)

            out_patch = out_patch[:, :, :ph, :pw]
            out_full[:, :, y0:y1, x0:x1] += out_patch
            weight[:, :, y0:y1, x0:x1] += 1.0

    out_full = out_full / torch.clamp(weight, min=1.0)
    return out_full


# =========================================
# 指标写入
# =========================================
def write_metrics(log_file, img_name,
                  psnr_base, ssim_base, l1_base,
                  psnr_lora, ssim_lora, l1_lora):
    diff_psnr = psnr_lora - psnr_base
    diff_ssim = ssim_lora - ssim_base
    diff_l1 = l1_lora - l1_base
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            f"{img_name}: "
            f"base_PSNR={psnr_base:.4f}, base_SSIM={ssim_base:.4f}, base_L1={l1_base:.6f}; "
            f"lora_PSNR={psnr_lora:.4f}, lora_SSIM={ssim_lora:.4f}, lora_L1={l1_lora:.6f}; "
            f"DiffPSNR={diff_psnr:+.4f}, DiffSSIM={diff_ssim:+.4f}, DiffL1={diff_l1:+.6f}\n"
        )


# =========================================
# MAIN
# =========================================
def main():
    ap = argparse.ArgumentParser()

    # 新模式：txt pairs
    ap.add_argument("--pair_list", default=None,
                    help="配对好的 txt 列表，每行：LQ_path GT_path。提供该参数则优先使用。")

    # 旧模式：文件夹（作为备用）
    ap.add_argument("--input_dir", default=None, help="LQ 图像文件夹（旧模式）")
    ap.add_argument("--gt_dir", default=None, help="GT 图像文件夹（旧模式）")

    ap.add_argument("--output_dir", required=True, help="输出结果文件夹")
    ap.add_argument("--base_ckpt", required=True, help="base Histoformer 权重路径")
    ap.add_argument("--lora_ckpt", required=True, help="LoRA 权重（训练得到的 lora.pth）")
    ap.add_argument("--domain", required=True, help="当前 LoRA 域名（如 rain/hazy）")
    ap.add_argument("--domains", required=True, help="所有 LoRA 域列表，用逗号分隔")
    ap.add_argument("--rank", type=int, default=8, help="LoRA rank（需与训练时一致）")
    ap.add_argument("--alpha", type=float, default=8, help="LoRA alpha（需与训练时一致）")
    ap.add_argument("--tile", type=int, default=640, help="tile 尺寸（过大 OOM 就调小）")
    ap.add_argument("--overlap", type=int, default=32, help="tile 之间 overlap 像素")
    ap.add_argument("--device", default="cuda", help="cuda 或 cpu")

    args = ap.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # -------------- 读取 pairs --------------
    if args.pair_list is not None:
        print("[INFO] reading pairs from", args.pair_list)
        pairs = read_pairs(args.pair_list)
    else:
        assert args.input_dir and args.gt_dir, \
            "Either --pair_list or (--input_dir and --gt_dir) must be provided."
        print("[INFO] listing images from", args.input_dir)
        img_paths = list_images(args.input_dir)
        pairs = []
        for p in img_paths:
            name = os.path.basename(p)
            gt_p = os.path.join(args.gt_dir, name)
            pairs.append((p, gt_p))

    print(f"[INFO] total pairs: {len(pairs)}")
    if len(pairs) == 0:
        print("[WARN] no pairs found, exit.")
        return

    log_file = os.path.join(args.output_dir, "metrics_log.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("# image_name | base vs lora PSNR/SSIM/L1\n")

    # -------------- 构建 base --------------
    print("[INFO] building base Histoformer...")
    net_base = build_histoformer(
        weights=args.base_ckpt,
        yaml_file="lora_adapters/configs/histoformer_mol.yaml",
    )

    # -------------- 构建 LoRA --------------
    print("[INFO] building Histoformer with LoRA...")
    net_lora = build_histoformer(
        weights=args.base_ckpt,
        yaml_file="lora_adapters/configs/histoformer_mol.yaml",
    )
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    net_lora = inject_lora(net_lora, rank=args.rank, domain_list=domains, alpha=args.alpha)

    print("[INFO] loading LoRA weights from", args.lora_ckpt)
    sd = torch.load(args.lora_ckpt, map_location="cpu")

    # 过滤 shape 不匹配的 key（更鲁棒）
    msd = net_lora.state_dict()
    filtered = {}
    skipped = 0
    for k, v in sd.items():
        if k in msd and msd[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped += 1
    print(f"[INFO] load lora: matched={len(filtered)} skipped={skipped}")

    msd.update(filtered)
    net_lora.load_state_dict(msd, strict=False)

    set_single_domain(net_lora, args.domain)

    net_base.to(device).eval()
    net_lora.to(device).eval()

    to_pil = TV.ToPILImage()

    # -------------- 逐张推理 --------------
    sum_psnr_base = sum_ssim_base = sum_l1_base = 0.0
    sum_psnr_lora = sum_ssim_lora = sum_l1_lora = 0.0
    count = 0

    with torch.no_grad():
        for idx, (lq_path, gt_path) in enumerate(tqdm(pairs, desc="Infer")):
            name = os.path.basename(lq_path)
            stem, _ = os.path.splitext(name)

            if not os.path.isfile(lq_path):
                print(f"[WARN] LQ missing: {lq_path}, skip.")
                continue
            if not os.path.isfile(gt_path):
                print(f"[WARN] GT missing: {gt_path}, skip.")
                continue

            x_lq, _ = load_image(lq_path)
            x_gt, _ = load_image(gt_path)
            x_lq = x_lq.to(device)
            x_gt = x_gt.to(device)

            # 对齐尺寸取 min
            _, _, h1, w1 = x_lq.shape
            _, _, h2, w2 = x_gt.shape
            H = min(h1, h2)
            W = min(w1, w2)
            if H < 8 or W < 8:
                print(f"[WARN] image too small ({H}x{W}), skip.")
                continue
            x_lq = x_lq[:, :, :H, :W]
            x_gt = x_gt[:, :, :H, :W]

            out_base = tile_inference(net_base, x_lq, tile=args.tile, overlap=args.overlap, device=device)
            out_lora = tile_inference(net_lora, x_lq, tile=args.tile, overlap=args.overlap, device=device)

            out_base_clamp = out_base.clamp(0, 1)
            out_lora_clamp = out_lora.clamp(0, 1)
            gt_clamp = x_gt.clamp(0, 1)

            l1_base = torch.mean(torch.abs(out_base_clamp - gt_clamp)).item()
            l1_lora = torch.mean(torch.abs(out_lora_clamp - gt_clamp)).item()
            psnr_base = calc_psnr(out_base_clamp, gt_clamp)
            psnr_lora = calc_psnr(out_lora_clamp, gt_clamp)
            ssim_base = calc_ssim(out_base_clamp, gt_clamp)
            ssim_lora = calc_ssim(out_lora_clamp, gt_clamp)

            write_metrics(log_file, name,
                          psnr_base, ssim_base, l1_base,
                          psnr_lora, ssim_lora, l1_lora)

            sum_psnr_base += psnr_base
            sum_ssim_base += ssim_base
            sum_l1_base += l1_base
            sum_psnr_lora += psnr_lora
            sum_ssim_lora += ssim_lora
            sum_l1_lora += l1_lora
            count += 1

            # 保存结果
            out_base_path = os.path.join(args.output_dir, f"{stem}_base.png")
            out_lora_path = os.path.join(args.output_dir, f"{stem}_lora.png")
            save_image(out_base_clamp, out_base_path)
            save_image(out_lora_clamp, out_lora_path)

            # 三联图
            pil_orig = to_pil(x_lq.squeeze(0).cpu())
            pil_base = to_pil(out_base_clamp.squeeze(0).cpu())
            pil_lora = to_pil(out_lora_clamp.squeeze(0).cpu())

            h_min = min(pil_orig.height, pil_base.height, pil_lora.height)

            def resize_h(im):
                if im.height == h_min:
                    return im
                new_w = int(im.width * (h_min / im.height))
                return im.resize((new_w, h_min), Image.BICUBIC)

            pil_orig_disp = resize_h(pil_orig)
            pil_base_disp = resize_h(pil_base)
            pil_lora_disp = resize_h(pil_lora)

            w_o2, w_b2, w_l2 = pil_orig_disp.width, pil_base_disp.width, pil_lora_disp.width
            total_w = w_o2 + w_b2 + w_l2
            comp3 = Image.new("RGB", (total_w, h_min))
            comp3.paste(pil_orig_disp, (0, 0))
            comp3.paste(pil_base_disp, (w_o2, 0))
            comp3.paste(pil_lora_disp, (w_o2 + w_b2, 0))

            comp3_path = os.path.join(args.output_dir, f"{stem}_compare3.png")
            comp3.save(comp3_path)

            torch.cuda.empty_cache()

    # 汇总平均指标
    if count > 0:
        avg_psnr_base = sum_psnr_base / count
        avg_ssim_base = sum_ssim_base / count
        avg_l1_base = sum_l1_base / count
        avg_psnr_lora = sum_psnr_lora / count
        avg_ssim_lora = sum_ssim_lora / count
        avg_l1_lora = sum_l1_lora / count

        print("========== Summary ==========")
        print(f"Num samples: {count}")
        print(f"Base : PSNR={avg_psnr_base:.4f}, SSIM={avg_ssim_base:.4f}, L1={avg_l1_base:.6f}")
        print(f"LoRA : PSNR={avg_psnr_lora:.4f}, SSIM={avg_ssim_lora:.4f}, L1={avg_l1_lora:.6f}")
        print(f"Diff : ΔPSNR={avg_psnr_lora-avg_psnr_base:+.4f}, "
              f"ΔSSIM={avg_ssim_lora-avg_ssim_base:+.4f}, "
              f"ΔL1={avg_l1_lora-avg_l1_base:+.6f}")

    print("[INFO] done.")
    print("Metrics written to:", log_file)


if __name__ == "__main__":
    main()

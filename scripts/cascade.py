# lora_adapters/cascade.py
# 顺序修复（级联）脚本：支持手动指定两个域 & 顺序
# - 保存对比图功能保持不变：输出拼接图 [LQ | stage1 | stage2]
# - 指标评估功能保持/增强：若提供 GT，则计算 PSNR/SSIM，并可导出 CSV

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from lora_adapters.utils import build_histoformer, save_image
from lora_adapters.inject_lora import inject_lora
from lora_adapters.utils_merge import map_to_single_domain_keys, apply_weighted_lora


# ====================== IQA 指标：PSNR / SSIM（与 infer_data 保持一致风格） ======================

def tensor_psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> float:
    """pred, gt: [B,3,H,W] in [0,1]"""
    mse = torch.mean((pred - gt) ** 2).clamp(min=1e-10)
    psnr = 10.0 * torch.log10((max_val ** 2) / mse)
    return psnr.item()


def tensor_ssim(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0, window_size: int = 11) -> float:
    """简化 SSIM：转灰度后计算"""
    if pred.shape[1] == 3:
        pred_y = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        gt_y   = 0.299 * gt[:, 0:1] + 0.587 * gt[:, 1:2] + 0.114 * gt[:, 2:3]
    else:
        pred_y, gt_y = pred, gt

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    def gaussian_window(ws: int, sigma: float = 1.5):
        gauss = torch.Tensor([
            math.exp(-(x - ws // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(ws)
        ])
        gauss = gauss / gauss.sum()
        return gauss

    window_1d = gaussian_window(window_size).to(pred.device)
    window_2d = (window_1d[:, None] * window_1d[None, :]).unsqueeze(0).unsqueeze(0)
    window_2d = window_2d.expand(1, 1, window_size, window_size)

    mu_x = F.conv2d(pred_y, window_2d, padding=window_size // 2, groups=1)
    mu_y = F.conv2d(gt_y,   window_2d, padding=window_size // 2, groups=1)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred_y * pred_y, window_2d, padding=window_size // 2, groups=1) - mu_x2
    sigma_y2 = F.conv2d(gt_y   * gt_y,   window_2d, padding=window_size // 2, groups=1) - mu_y2
    sigma_xy = F.conv2d(pred_y * gt_y,   window_2d, padding=window_size // 2, groups=1) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean().item()


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


def pad_to_multiple(x: torch.Tensor, factor: int = 8):
    """将 [B,3,H,W] pad 到 H,W 都是 factor 的倍数，返回 padded 张量和 pad 信息。"""
    b, c, h, w = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    # F.pad 顺序 (left, right, top, bottom)
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x_pad, (0, pad_w, 0, pad_h)


def unpad(x: torch.Tensor, pad: tuple[int, int, int, int], orig_h: int, orig_w: int):
    """从 pad 后的张量中裁剪回原始 H,W。"""
    return x[..., :orig_h, :orig_w]


def run_cascade(
    base_ckpt: str,
    yaml_file: str | None,
    input_path: str,
    output_dir: str,
    stage1: str = "haze2",
    stage2: str = "rain3",
    loradb_root: str = "weights/lora",
    rank: int = 16,
    alpha: float = 16.0,
    device: str = "cuda",
    pad_factor: int = 8,
    gt_root: str | None = None,
    metrics_csv: str | None = None,
    summary_csv: str | None = None,
    run_name: str = "cascade",
    save_prefix: str | None = None,
):
    """顺序修复（两阶段级联）。

    - 保存对比图：始终输出拼接图 [LQ | stage1_out | stage2_out]（不改变原有形式）
    - 若提供 gt_root 且存在同名 GT：计算 PSNR/SSIM（LQ/base/stage1/cascade）并可导出 CSV
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

    if save_prefix is None:
        save_prefix = f"cascade_{stage1}_{stage2}_"

    input_path = Path(input_path)
    if input_path.is_dir():
        img_files = [
            p for p in input_path.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
        ]
    else:
        img_files = [input_path]

    print(f"[INFO] stage1={stage1} -> stage2={stage2}")
    print(f"[INFO] 共 {len(img_files)} 张图像待处理")

    all_metrics: list[dict] = []

    for img_path in img_files:
        print(f"\n[PROC] {img_path}")
        img = Image.open(img_path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)  # [1,3,H,W]
        _, _, h, w = x.shape

        # pad 到 factor 的倍数，避免 PixelUnshuffle 等要求
        x_pad, pad_info = pad_to_multiple(x, factor=pad_factor)

        # ===== base（不加载 LoRA）=====
        zero_single_lora(net)
        with torch.no_grad():
            y_base = net(x_pad)
        y_base = unpad(y_base, pad_info, h, w)

        # ===== Step 1: stage1 LoRA 修复 =====
        zero_single_lora(net)
        load_domain_lora_to_single(net, stage1, loradb_root=loradb_root)
        with torch.no_grad():
            y_1 = net(x_pad)
        y_1 = unpad(y_1, pad_info, h, w)

        # ===== Step 2: stage2 LoRA 修复（输入是 stage1 输出） =====
        zero_single_lora(net)
        load_domain_lora_to_single(net, stage2, loradb_root=loradb_root)
        y_in = y_1
        _, _, h2, w2 = y_in.shape
        y_in_pad, pad_info2 = pad_to_multiple(y_in, factor=pad_factor)
        with torch.no_grad():
            y_final = net(y_in_pad)
        y_final = unpad(y_final, pad_info2, h2, w2)

        # ===== 拼接并保存：LQ | stage1 输出 | stage1→stage2 输出 =====
        # concat = torch.cat([x, y_1, y_final], dim=3)  # [1,3,H,W*3]
        # ===== 拼接并保存：LQ | base | stage1 输出 | stage1→stage2 输出 =====
        concat = torch.cat([x, y_base, y_1, y_final], dim=3)  # [1,3,H,W*4]

        out_name = img_path.name
        out_path = output_dir / f"{save_prefix}{out_name}"
        save_image(concat, str(out_path))
        print(f"[SAVE] {out_path}")

        # ===== 指标（可选）：仅当存在 GT 时计算 =====
        if gt_root is not None:
            gt_path = Path(gt_root) / out_name
            if gt_path.is_file():
                gt_img = Image.open(gt_path).convert("RGB")
                gt = tfm(gt_img).unsqueeze(0).to(device)
                if gt.shape[2] != h or gt.shape[3] != w:
                    gt = F.interpolate(gt, size=(h, w), mode='bilinear', align_corners=False)

                m = {
                    "name": out_name,
                    "stage1": stage1,
                    "stage2": stage2,
                    "psnr_lq": tensor_psnr(x, gt),
                    "psnr_base": tensor_psnr(y_base, gt),
                    "psnr_stage1": tensor_psnr(y_1, gt),
                    "psnr_cascade": tensor_psnr(y_final, gt),
                    "ssim_lq": tensor_ssim(x, gt),
                    "ssim_base": tensor_ssim(y_base, gt),
                    "ssim_stage1": tensor_ssim(y_1, gt),
                    "ssim_cascade": tensor_ssim(y_final, gt),
                }
                all_metrics.append(m)
                print(f"[metrics] {m}")
            else:
                print(f"[warn] GT not found for {out_name}, expected: {gt_path}")

    # ===== 汇总输出（可选）：只有存在 GT 才会有 mean =====
    if len(all_metrics) > 0:
        psnr_lq = [m["psnr_lq"] for m in all_metrics]
        psnr_base = [m["psnr_base"] for m in all_metrics]
        psnr_s1 = [m["psnr_stage1"] for m in all_metrics]
        psnr_cas = [m["psnr_cascade"] for m in all_metrics]
        ssim_lq = [m["ssim_lq"] for m in all_metrics]
        ssim_base = [m["ssim_base"] for m in all_metrics]
        ssim_s1 = [m["ssim_stage1"] for m in all_metrics]
        ssim_cas = [m["ssim_cascade"] for m in all_metrics]

        print("===== MEAN METRICS (only images with GT) =====")
        print(f"PSNR  LQ      : {sum(psnr_lq)/len(psnr_lq):.3f}")
        print(f"PSNR  base    : {sum(psnr_base)/len(psnr_base):.3f}")
        print(f"PSNR  {stage1:<7}: {sum(psnr_s1)/len(psnr_s1):.3f}")
        print(f"PSNR  cascade : {sum(psnr_cas)/len(psnr_cas):.3f}")
        print(f"SSIM  LQ      : {sum(ssim_lq)/len(ssim_lq):.4f}")
        print(f"SSIM  base    : {sum(ssim_base)/len(ssim_base):.4f}")
        print(f"SSIM  {stage1:<7}: {sum(ssim_s1)/len(ssim_s1):.4f}")
        print(f"SSIM  cascade : {sum(ssim_cas)/len(ssim_cas):.4f}")

        # ===== 写 CSV（可选） =====
        if metrics_csv is not None:
            metrics_csv = str(metrics_csv)
            Path(metrics_csv).parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "name", "stage1", "stage2",
                "psnr_lq", "psnr_base", "psnr_stage1", "psnr_cascade",
                "ssim_lq", "ssim_base", "ssim_stage1", "ssim_cascade",
                "run_name", "pad_factor", "rank", "alpha",
                "output_dir", "input_root", "gt_root",
            ]

            mean_row = {
                "name": "__mean__",
                "stage1": stage1,
                "stage2": stage2,
                "psnr_lq": float(sum(psnr_lq)/len(psnr_lq)),
                "psnr_base": float(sum(psnr_base)/len(psnr_base)),
                "psnr_stage1": float(sum(psnr_s1)/len(psnr_s1)),
                "psnr_cascade": float(sum(psnr_cas)/len(psnr_cas)),
                "ssim_lq": float(sum(ssim_lq)/len(ssim_lq)),
                "ssim_base": float(sum(ssim_base)/len(ssim_base)),
                "ssim_stage1": float(sum(ssim_s1)/len(ssim_s1)),
                "ssim_cascade": float(sum(ssim_cas)/len(ssim_cas)),
            }

            with open(metrics_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for m in all_metrics:
                    row = {
                        **m,
                        "run_name": run_name,
                        "pad_factor": pad_factor,
                        "rank": rank,
                        "alpha": alpha,
                        "output_dir": str(output_dir),
                        "input_root": str(input_path),
                        "gt_root": gt_root,
                    }
                    w.writerow(row)
                w.writerow({
                    **mean_row,
                    "run_name": run_name,
                    "pad_factor": pad_factor,
                    "rank": rank,
                    "alpha": alpha,
                    "output_dir": str(output_dir),
                    "input_root": str(input_path),
                    "gt_root": gt_root,
                })

            print(f"[CSV] metrics saved to {metrics_csv}")

        # ===== summary_csv（可选，append 一行） =====
        if summary_csv is not None:
            summary_csv = str(summary_csv)
            Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
            write_header = (not Path(summary_csv).exists())
            with open(summary_csv, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "run_name", "stage1", "stage2",
                        "num_images",
                        "psnr_lq", "psnr_base", "psnr_stage1", "psnr_cascade",
                        "ssim_lq", "ssim_base", "ssim_stage1", "ssim_cascade",
                        "pad_factor", "rank", "alpha",
                        "output_dir", "input_root", "gt_root",
                    ])
                writer.writerow([
                    run_name, stage1, stage2,
                    len(all_metrics),
                    float(sum(psnr_lq)/len(psnr_lq)),
                    float(sum(psnr_base)/len(psnr_base)),
                    float(sum(psnr_s1)/len(psnr_s1)),
                    float(sum(psnr_cas)/len(psnr_cas)),
                    float(sum(ssim_lq)/len(ssim_lq)),
                    float(sum(ssim_base)/len(ssim_base)),
                    float(sum(ssim_s1)/len(ssim_s1)),
                    float(sum(ssim_cas)/len(ssim_cas)),
                    pad_factor, rank, alpha,
                    str(output_dir), str(input_path), gt_root,
                ])
            print(f"[CSV] summary appended to {summary_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Cascade restoration (2-stage): apply stage1 LoRA then stage2 LoRA"
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

    # 两阶段域与顺序
    parser.add_argument("--stage1", type=str, default="haze2",
                        help="第一阶段域名（默认 haze2）")
    parser.add_argument("--stage2", type=str, default="rain3",
                        help="第二阶段域名（默认 rain3）")
    parser.add_argument("--domains", type=str, default=None,
                        help="可选：用 'a,b' 一次性指定两个域（会覆盖 --stage1/--stage2）")
    parser.add_argument("--reverse", action="store_true",
                        help="可选：交换顺序（stage2 -> stage1）")
    parser.add_argument("--save_prefix", type=str, default=None,
                        help="输出文件名前缀。默认自动用 cascade_<stage1>_<stage2>_")

    # padding / 评价
    parser.add_argument("--pad_factor", type=int, default=8,
                        help="pad 到该倍数（默认 8）")
    parser.add_argument("--gt_root", type=str, default=None,
                        help="GT 根目录（若提供则计算 PSNR/SSIM，GT 默认同名文件）")
    parser.add_argument("--metrics_csv", type=str, default=None,
                        help="保存每张图 PSNR/SSIM 的 CSV（可选）")
    parser.add_argument("--summary_csv", type=str, default=None,
                        help="追加一行 mean 到总表 CSV（可选）")
    parser.add_argument("--run_name", type=str, default="cascade",
                        help="实验名称（写入 CSV 用）")

    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    stage1, stage2 = args.stage1, args.stage2
    if args.domains is not None:
        parts = [p.strip() for p in args.domains.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError("--domains 必须是 'a,b' 两个域名")
        stage1, stage2 = parts[0], parts[1]
    if args.reverse:
        stage1, stage2 = stage2, stage1

    run_cascade(
        base_ckpt=args.base_ckpt,
        yaml_file=args.yaml,
        input_path=args.input,
        output_dir=args.output,
        stage1=stage1,
        stage2=stage2,
        loradb_root=args.loradb_root,
        rank=args.rank,
        alpha=args.alpha,
        device=args.device,
        pad_factor=args.pad_factor,
        gt_root=args.gt_root,
        metrics_csv=args.metrics_csv,
        summary_csv=args.summary_csv,
        run_name=args.run_name,
        save_prefix=args.save_prefix,
    )


if __name__ == "__main__":
    main()

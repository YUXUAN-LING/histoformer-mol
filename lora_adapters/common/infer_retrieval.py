# lora_adapters/infer_retrieval.py

import os, math, csv, torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn.functional as Fnn

from lora_adapters.utils import build_histoformer, save_image
from lora_adapters.inject_lora import inject_lora
from lora_adapters.embedding_dinov2 import DINOv2Embedder
from lora_adapters.embedding_clip import CLIPEmbedder
from lora_adapters.domain_orchestrator import DomainOrchestrator


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


def zero_single_lora(net: torch.nn.Module):
    """
    将 .lora_down._Single.weight / .lora_up._Single.weight 置零，
    确保 base 输出不受上一张图的 LoRA 残留影响。
    """
    with torch.no_grad():
        for name, p in net.named_parameters():
            if (".lora_down._Single.weight" in name) or (".lora_up._Single.weight" in name):
                p.zero_()


def infer_embedder_tag(args) -> str:
    """
    根据 embedder 配置生成一个 tag，用来匹配 avg_embedding_<tag>.npy
    要和 build_prototypes.py 里保持一致：
      - dino_v2 默认：dinov2_vitb14
      - clip：clip_<model>
      - fft：fft_amp
    """
    if args.embedder == "dino_v2":
        # 目前我们默认用的是 vit_base_patch14_dinov2，对应之前构建的 dinov2_vitb14
        return "dinov2_vitb14"
    elif args.embedder == "clip":
        model = args.clip_model  # e.g. "ViT-B-16"
        return f"clip_{model.replace('/', '_').lower()}"
    elif args.embedder == "fft":
        return "fft_amp"
    else:
        return args.embedder


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_ckpt', required=True)
    ap.add_argument('--input', required=True)              # 单图或目录（LQ）
    ap.add_argument('--output', required=True)
    ap.add_argument('--loradb', default='loradb',          # loradb/<Domain>/{avg_embedding_*.npy, *.pth}
                    help='LoRA 数据库根目录，例如 weights/lora')
    ap.add_argument('--domains', default='Rain,Snow,Fog,Clear')
    ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=8.0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--yaml', type=str, default=None,
                    help='和 base_ckpt 对应的 YAML 配置文件')

    # 嵌入 backbone 选择
    ap.add_argument(
        '--embedder', type=str, default='dino_v2',
        choices=['dino_v2', 'clip', 'fft'],
        help='选择特征模型：dino_v2 / clip / fft'
    )
    ap.add_argument('--clip_model', type=str, default='ViT-B-16')
    ap.add_argument('--clip_pretrained', type=str, default='openai')

    # FFT embedding 参数（如未使用 fft，可忽略）
    ap.add_argument('--fft_resize', type=int, default=256)
    ap.add_argument('--fft_center_crop', type=int, default=128)
    ap.add_argument('--fft_out_size', type=int, default=32)

    # 相似度度量 & 温度（影响权重尖锐程度）
    ap.add_argument(
        '--sim_metric', type=str, default='cosine',
        choices=['cosine', 'euclidean'],
        help='原型检索的相似度度量：cosine 或 euclidean(L2 距离)'
    )
    ap.add_argument(
        '--temperature', type=float, default=0.07,
        help='原型权重 softmax 的温度，越小权重越尖锐（更接近 top-1 / top-2）'
    )

    # embedder_tag 用来自动匹配 avg_embedding_<tag>.npy
    ap.add_argument(
        '--embedder_tag', type=str, default=None,
        help='用于寻找 avg_embedding_<tag>.npy 的 tag；若不提供，则根据 --embedder 自动推断'
    )

    # GT 根目录 & 指标 csv
    ap.add_argument('--gt_root', type=str, default=None,
                    help='GT 清晰图所在根目录，例如 samples/clear，会按文件名匹配')
    ap.add_argument('--metrics_csv', type=str, default=None,
                    help='若提供，将把每张图的 PSNR/SSIM 写入该 CSV 文件')

    args = ap.parse_args()

    device = 'cuda' if (args.device.startswith('cuda') and torch.cuda.is_available()) else 'cpu'

    # 自动推断 embedder_tag
    if args.embedder_tag is None:
        args.embedder_tag = infer_embedder_tag(args)
    print(f"[INFO] embedder = {args.embedder}, embedder_tag = {args.embedder_tag}")

    # 1) 主干 & 注入单域 LoRA
    print('[INFO] building base Histoformer...')
    net = build_histoformer(weights=args.base_ckpt, yaml_file=args.yaml).to(device)
    net = inject_lora(
        net,
        rank=args.rank,
        domain_list=['_Single'],
        alpha=args.alpha,
        target_names=None,
        patterns=None
    )
    net.eval().to(device)

    # 2) 选择 embedder
    if args.embedder == 'dino_v2':
        emb = DINOv2Embedder(device=device)
        print(f"[INFO] using DINOv2 embedder on {device}")
    elif args.embedder == 'clip':
        emb = CLIPEmbedder(
            device=device,
            model_name=args.clip_model,
            pretrained=args.clip_pretrained
        )
        print(f"[INFO] using CLIP embedder {args.clip_model}/{args.clip_pretrained} on {device}")
    elif args.embedder == 'fft':
        from lora_adapters.embedding_fft import FFTAmplitudeEmbedder
        emb = FFTAmplitudeEmbedder(
            device=device,
            resize=args.fft_resize,
            center_crop=args.fft_center_crop,
            out_size=args.fft_out_size,
        )
        print(f"[INFO] using FFTAmplitudeEmbedder (resize={args.fft_resize}, "
              f"center_crop={args.fft_center_crop}, out_size={args.fft_out_size}) on {device}")
    else:
        raise ValueError(f"unknown embedder: {args.embedder}")

    # 3) 构建 DomainOrchestrator（自动从 loradb/<d> 找 avg_embedding_<embedder_tag>.npy）
    doms = [d.strip() for d in args.domains.split(',') if d.strip()]
    orch = DomainOrchestrator(
        doms,
        lora_db_path=args.loradb,
        sim_metric=args.sim_metric,
        temperature=args.temperature,
        embedder_tag=args.embedder_tag
    )

    tfm_to_tensor = T.ToTensor()
    gt_root = args.gt_root

    def run_one(img_path: str):
        """
        对单张 LQ 图像：
        - base 输出（LoRA=0）
        - 动态加权 LoRA 输出
        - 若存在 GT，则返回各自的 psnr/ssim
        同时返回拼接图 [LQ | base | mix]
        """
        from lora_adapters.utils_merge import apply_weighted_lora, map_to_single_domain_keys

        img = Image.open(img_path).convert('RGB')
        x = tfm_to_tensor(img).unsqueeze(0).to(device)   # [1,3,H,W]
        _, _, h, w = x.shape

        # pad 到 8 的倍数，避免 PixelUnshuffle 报错
        factor = 8
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h != 0 or pad_w != 0:
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_in = x

        # 1) base 输出（先清零 LoRA）
        zero_single_lora(net)
        with torch.no_grad():
            y_base = net(x_in)
        y_base = y_base[:, :, :h, :w]

        # 2) embedding + 检索 + 动态加权 LoRA
        v = emb.embed_image(img)
        picks = orch.select_topk(v, top_k=args.topk)
        print('[topk]', picks)
        merged = orch.build_weighted_lora(picks)
        merged_for_model = map_to_single_domain_keys(merged, target_domain_name='_Single')
        apply_weighted_lora(net, merged_for_model)

        # 3) 混合 LoRA 输出
        with torch.no_grad():
            y_mix = net(x_in)
        y_mix = y_mix[:, :, :h, :w]

        # 4) 拼接图像：LQ | base | mix
        concat = torch.cat([x, y_base, y_mix], dim=3)    # 宽度拼接 -> [1,3,H,3W]

        # 5) 若存在 GT，则计算指标
        metrics = None
        if gt_root is not None:
            gt_path = os.path.join(gt_root, os.path.basename(img_path))
            if os.path.isfile(gt_path):
                gt_img = Image.open(gt_path).convert('RGB')
                gt = tfm_to_tensor(gt_img).unsqueeze(0).to(device)
                # 如果尺寸不一致，resize GT 到 LQ 尺寸
                if gt.shape[2] != h or gt.shape[3] != w:
                    gt = F.interpolate(gt, size=(h, w), mode='bilinear', align_corners=False)

                psnr_lq   = tensor_psnr(x,  gt)      # x: [1,3,H,W]
                psnr_base = tensor_psnr(y_base, gt)
                psnr_mix  = tensor_psnr(y_mix,  gt)

                ssim_lq   = tensor_ssim(x,  gt)
                ssim_base = tensor_ssim(y_base, gt)
                ssim_mix  = tensor_ssim(y_mix,  gt)

                metrics = {
                    "name": os.path.basename(img_path),
                    "psnr_lq": psnr_lq,
                    "psnr_base": psnr_base,
                    "psnr_mix": psnr_mix,
                    "ssim_lq": ssim_lq,
                    "ssim_base": ssim_base,
                    "ssim_mix": ssim_mix,
                }
                print(f"[metrics] {metrics}")
            else:
                print(f"[warn] GT not found for {img_path}, expected: {gt_path}")

        return concat, metrics

    os.makedirs(args.output, exist_ok=True)
    all_metrics = []

    # 处理目录 / 单图
    if os.path.isdir(args.input):
        for fn in os.listdir(args.input):
            if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                continue
            in_path = os.path.join(args.input, fn)
            concat, m = run_one(in_path)
            out_path = os.path.join(args.output, f"cmp_{fn}")
            save_image(concat, out_path)
            print('saved:', out_path)
            if m is not None:
                all_metrics.append(m)
    else:
        concat, m = run_one(args.input)
        # 若 output 是目录，就用默认文件名
        if os.path.isdir(args.output):
            outp = os.path.join(args.output, "cmp_restored.png")
        else:
            outp = args.output
        save_image(concat, outp)
        print('saved:', outp)
        if m is not None:
            all_metrics.append(m)

    # 写入 CSV
    if args.metrics_csv is not None and len(all_metrics) > 0:
        csv_path = args.metrics_csv
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "psnr_lq", "psnr_base", "psnr_mix",
                    "ssim_lq", "ssim_base", "ssim_mix",
                ]
            )
            writer.writeheader()
            for row in all_metrics:
                writer.writerow(row)
        print(f"[metrics] CSV saved to: {csv_path}")


if __name__ == '__main__':
    main()

# lora_adapters/infer_data.py

import os, math, csv, torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn.functional as Fnn

from lora_adapters.utils import build_histoformer, save_image
from lora_adapters.inject_lora import inject_lora
from lora_adapters.embedding_dinov2 import DINOv2Embedder
from lora_adapters.embedding_clip import CLIPEmbedder
from lora_adapters.domain_orchestrator import DomainOrchestrator
from lora_adapters.lora_linear import LoRALinear, LoRAConv2d


# ====================== IQA 指标：PSNR / SSIM ======================

def tensor_psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> float:
    """
    pred, gt: [B,3,H,W], in [0,1]
    """
    mse = torch.mean((pred - gt) ** 2).clamp(min=1e-10)
    psnr = 10.0 * torch.log10((max_val ** 2) / mse)
    return psnr.item()


def tensor_ssim(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0, window_size: int = 11) -> float:
    """
    一个简单的 SSIM 实现，pred, gt: [B,3,H,W], in [0,1]
    """
    # 转成灰度简化（也可以对 3 通道分别算再平均）
    if pred.shape[1] == 3:
        pred_y = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        gt_y   = 0.299 * gt[:, 0:1] + 0.587 * gt[:, 1:2] + 0.114 * gt[:, 2:3]
    else:
        pred_y, gt_y = pred, gt

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # 高斯窗口
    def gaussian_window(window_size: int, sigma: float = 1.5):
        gauss = torch.Tensor([
            math.exp(-(x - window_size//2)**2 / float(2*sigma**2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        return gauss

    window_1d = gaussian_window(window_size).to(pred.device)
    window_2d = (window_1d[:, None] * window_1d[None, :]).unsqueeze(0).unsqueeze(0)
    window_2d = window_2d.expand(1, 1, window_size, window_size)

    mu_x = Fnn.conv2d(pred_y, window_2d, padding=window_size//2, groups=1)
    mu_y = Fnn.conv2d(gt_y,   window_2d, padding=window_size//2, groups=1)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = Fnn.conv2d(pred_y * pred_y, window_2d, padding=window_size//2, groups=1) - mu_x2
    sigma_y2 = Fnn.conv2d(gt_y   * gt_y,   window_2d, padding=window_size//2, groups=1) - mu_y2
    sigma_xy = Fnn.conv2d(pred_y * gt_y,   window_2d, padding=window_size//2, groups=1) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))

    return ssim_map.mean().item()


def zero_single_lora(net: torch.nn.Module):
    """
    旧接口：将 .lora_down._Single.weight / .lora_up._Single.weight 置零，
    现在不再使用，只是为了兼容保留。
    """
    with torch.no_grad():
        for name, p in net.named_parameters():
            if (".lora_down._Single.weight" in name) or (".lora_up._Single.weight" in name):
                p.zero_()


def map_lora_keys_to_domain(sd: dict, domain_name: str) -> dict:
    """将 state_dict 中的 LoRA 权重映射到指定域名的键上。

    支持两种常见形式：
    1) ... lora_down.<domain>.weight
    2) ... lora_down.weight  （无域名，我们自动插入 domain_name）
    """
    out = {}
    for k, v in sd.items():
        if "lora_" not in k or not k.endswith("weight"):
            continue
        parts = k.split(".")
        # 期望结尾形如 [..., 'lora_down', '<maybeDomain>', 'weight']
        if len(parts) >= 3 and parts[-1] == "weight" and parts[-3].startswith("lora_") and parts[-2] not in ("weight", "bias"):
            # 已有域名，直接替换为当前 domain
            parts[-2] = domain_name
        elif len(parts) >= 2 and parts[-1] == "weight" and parts[-2].startswith("lora_"):
            # 无域名：..., 'lora_down', 'weight' -> 插入 domain
            parts = parts[:-1] + [domain_name, parts[-1]]
        else:
            # 回退：在 weight 前插入 domain
            parts = parts[:-1] + [domain_name, parts[-1]]
        new_k = ".".join(parts)
        out[new_k] = v
    return out


def load_all_domain_loras(net: torch.nn.Module, orch: DomainOrchestrator):
    """一次性把每个域的 LoRA 权重加载到模型对应的域分支中。"""
    for name, dom in orch.domains.items():
        sd = torch.load(dom.lora_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        mapped = map_lora_keys_to_domain(sd, name)
        net.load_state_dict(mapped, strict=False)
        # 如需调试可以打印 missing / unexpected
        # missing, unexpected = net.load_state_dict(mapped, strict=False)
        # print(f"[LoRA] loaded domain={name}, missing={len(missing)}, unexpected={len(unexpected)}")


def set_all_lora_domain_weights(
    net: torch.nn.Module,
    weights: dict | None,
):
    """为模型中的所有 LoRA 层统一设置 domain_weights。

    - weights 为 None 时：多域 LoRA 不参与前向，相当于 pure base。
    - weights 为 dict[name->float]：所有 LoRA 层按同一组域权重做加权。
    """
    for m in net.modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            m.set_domain_weights(weights)

def _mask_and_renorm(weights: dict, allowed: list[str], renorm: bool = True):
    """从全局权重中取出 allowed 域，必要时组内归一化。"""
    w = {d: float(weights.get(d, 0.0)) for d in allowed}
    s = sum(w.values())
    if renorm and s > 1e-12:
        w = {d: v / s for d, v in w.items()}
    # 去掉 0
    return {d: v for d, v in w.items() if abs(v) > 0}

def set_lora_domain_weights_layerwise(
    net,
    w_global: dict | None,
    local_domains: list[str],
    global_domains: list[str],
    renorm: bool = True,
):
    """
    两级层级路由：
      - 浅层(local)：只允许 local_domains
      - 深层(global)：只允许 global_domains
      - 其他层：用 w_global（或 None）
    """
    if w_global is None:
        # pure base
        for name, m in net.named_modules():
            if isinstance(m, (LoRALinear, LoRAConv2d)):
                m.set_domain_weights(None)
        return

    # 你可以按经验先用这一版分层（非常适配 Histoformer encoder-decoder）
    local_prefix = (
        "encoder_level1", "encoder_level2",
        "decoder_level1", "refinement",
    )
    global_prefix = (
        "encoder_level3", "latent",
        "decoder_level3", "decoder_level2",
    )

    w_local  = _mask_and_renorm(w_global, local_domains,  renorm=renorm)
    w_glob   = _mask_and_renorm(w_global, global_domains, renorm=renorm)

    for name, m in net.named_modules():
        if not isinstance(m, (LoRALinear, LoRAConv2d)):
            continue
        if name.startswith(local_prefix):
            m.set_domain_weights(w_local)
        elif name.startswith(global_prefix):
            m.set_domain_weights(w_glob)
        else:
            m.set_domain_weights(w_global)


def get_embedder_tag(args) -> str:
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


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--base_ckpt', required=True)
    ap.add_argument('--yaml', default='configs/histoformer_base.yaml')

    ap.add_argument('--input', required=True, help='LQ 图像目录或单张图片')
    ap.add_argument('--output', required=True, help='输出目录 / 单张输出图')

    ap.add_argument('--loradb', required=True, help='LoRA 权重所在根目录，例如 weights/lora')
    ap.add_argument('--domains', required=True, help='参与混合的域列表，如 "rain2,low,rain,..."')
    ap.add_argument('--topk', type=int, default=3)

    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=float, default=16.0)
    ap.add_argument('--device', default='cuda')

    # embedder 相关
    ap.add_argument('--embedder', choices=['dino_v2', 'clip', 'fft', 'fft_enhanced'], default='dino_v2')
    ap.add_argument(
    "--dino_ckpt", type=str,
    default="weights/dinov2/vit_base_patch14_dinov2.lvd142m-384.pth",
    help="本地 DINOv2 权重（仅当 embedder=dino_v2 时使用）"
    )
    ap.add_argument('--clip_model', default='ViT-B/16')
    ap.add_argument('--clip_pretrained',type=str, default='weights/clip/open_clip_pytorch_model.bin',
                    help="本地 clip权重（仅当 embedder=clip 时使用）")

    ap.add_argument('--fft_resize', type=int, default=256)
    ap.add_argument('--fft_center_crop', type=int, default=0)
    ap.add_argument('--fft_out_size', type=int, default=32)
    ap.add_argument('--fft_clean_proto', type=str, default=None)

    ap.add_argument('--sim_metric', choices=['cosine', 'euclidean'], default='cosine')
    ap.add_argument('--temperature', type=float, default=0.07)
    ap.add_argument('--embedder_tag', type=str, default=None)
    ap.add_argument('--run_name', type=str, default='')

    ap.add_argument('--gt_root', type=str, default=None)
    ap.add_argument('--metrics_csv', type=str, default=None)
    ap.add_argument('--summary_csv', type=str, default=None)

    # ✅ 新增：支持用 trainlist / pair_list 作为输入（不影响原有功能）
    ap.add_argument(
        '--pair_list',
        type=str,
        default=None,
        help='可选：TXT 文件，每行 "LQ_path [GT_path]"；若提供则覆盖 --input 目录遍历'
    )

    args = ap.parse_args()

    device = 'cuda' if (args.device.startswith('cuda') and torch.cuda.is_available()) else 'cpu'

    # 自动推 embedder_tag（用于 avg_embedding_*.npy 命名）
    args.embedder_tag = args.embedder_tag or get_embedder_tag(args)
    
    print(f"[INFO] run_name = {args.run_name}")
    print(f"[INFO] embedder = {args.embedder}, embedder_tag = {args.embedder_tag}")

    # 0) 解析域列表（用于多域 LoRA 注入与路由）
    doms = [d.strip() for d in args.domains.split(',') if d.strip()]
    print(f"[INFO] domains = {doms}")

    # 1) 主干 & 注入多域 LoRA
    print('[INFO] building base Histoformer...')
    net = build_histoformer(weights=args.base_ckpt, yaml_file=args.yaml).to(device)
    net = inject_lora(
        net,
        rank=args.rank,
        domain_list=doms,
        alpha=args.alpha,
        target_names=None,
        patterns=None
    )
    net.eval().to(device)

    # 2) 选择 embedder
    if args.embedder == 'dino_v2':
        emb = DINOv2Embedder(device=device, ckpt_path=args.dino_ckpt)

    elif args.embedder == 'clip':
        emb = CLIPEmbedder(
            device=device,
            model_name=args.clip_model,
            pretrained=args.clip_pretrained
        )
    elif args.embedder == 'fft':
        from lora_adapters.embedding_fft import FFTAmplitudeEmbedder
        emb = FFTAmplitudeEmbedder(
            device=device,
            resize=args.fft_resize,
            center_crop=args.fft_center_crop,
            out_size=args.fft_out_size
        )
    elif args.embedder == 'fft_enhanced':
        from lora_adapters.embedding_fft import FFTEnhancedEmbedder
        emb = FFTEnhancedEmbedder(
            device=device,
            resize=args.fft_resize,
            center_crop=args.fft_center_crop,
            out_size=args.fft_out_size,
            clean_proto_path=args.fft_clean_proto
        )
    else:
        raise ValueError(f"unknown embedder: {args.embedder}")

    # 3) 构建 DomainOrchestrator（自动从 loradb/<d> 找 avg_embedding_<embedder_tag>.npy）
    orch = DomainOrchestrator(
        doms,
        lora_db_path=args.loradb,
        sim_metric=args.sim_metric,
        temperature=args.temperature,
        embedder_tag=args.embedder_tag
    )

    # 3.1) 将各域 LoRA 权重加载到模型对应域分支
    load_all_domain_loras(net, orch)

    tfm_to_tensor = T.ToTensor()
    gt_root = args.gt_root

    def run_one(img_path: str, gt_path_override: str | None = None):
        """
        对单张 LQ 图像：
        - base 输出（LoRA 关闭）
        - 动态加权 LoRA 输出
        - 若存在 GT，则返回各自的 psnr/ssim
        同时返回拼接图 [LQ | base | mix]

        gt_path_override:
            如果不为 None，则优先使用该路径作为 GT（来自 pair_list），
            否则退回到原来的 gt_root + 文件名 逻辑。
        """
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

        # 1) base 输出：关闭所有 LoRA（domain_weights=None）
        set_all_lora_domain_weights(net, None)
        with torch.no_grad():
            y_base = net(x_in)
        y_base = y_base[:, :, :h, :w]

        # 2) embedding + 原型检索 -> 获得每个域的权重
        v = emb.embed_image(img)
        picks = orch.select_topk(v, top_k=args.topk)
        print('[topk]', picks)
        weights_dict = {name: float(w) for name, w in picks}

        # 将域权重设置到所有 LoRA 层
        set_all_lora_domain_weights(net, weights_dict)
        # 分层路由
        # set_lora_domain_weights_layerwise(
        #     net,
        #     w_global=weights_dict,
        #     local_domains=["rain", "rain2", "rain3", "rainy", "snow", "snow1"],
        #     global_domains=["haze", "haze1", "haze2", "low", "low1"],
        #     renorm=True,
        # )

        # 3) 混合 LoRA 输出
        with torch.no_grad():
            y_mix = net(x_in)
        y_mix = y_mix[:, :, :h, :w]

        # 4) 拼接图像：LQ | base | mix
        concat = torch.cat([x, y_base, y_mix], dim=3)    # 宽度拼接 -> [1,3,H,3W]

        # 5) 若存在 GT，则计算指标
        metrics = None

        # 优先使用 pair_list 提供的 GT 路径，其次使用 gt_root + 文件名
        gt_path = None
        if gt_path_override is not None:
            gt_path = gt_path_override
        elif gt_root is not None:
            gt_path = os.path.join(gt_root, os.path.basename(img_path))

        if gt_path is not None:
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

    # ===== 处理输入：支持原有 input（目录/单图） + 新增 pair_list 模式 =====
    use_pair_list = args.pair_list is not None
    pair_items: list[tuple[str, str | None]] = []

    if use_pair_list:
        print(f"[INFO] 使用 pair_list = {args.pair_list}")
        with open(args.pair_list, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    continue
                items = line.split()
                if len(items) == 1:
                    lq_path = items[0]
                    gt_path = None
                else:
                    lq_path, gt_path = items[0], items[1]
                pair_items.append((lq_path, gt_path))

    # 处理目录 / 单图 / pair_list
    if use_pair_list:
        for lq_path, gt_path in pair_items:
            concat, m = run_one(lq_path, gt_path_override=gt_path)
            fn = os.path.basename(lq_path)
            out_path = os.path.join(args.output, f"cmp_{fn}")
            save_image(concat, out_path)
            print('saved:', out_path)
            if m is not None:
                all_metrics.append(m)
    else:
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

    # ===== 写入 per-image metrics CSV（带配置 + 平均行） =====
    if args.metrics_csv is not None and len(all_metrics) > 0:
        csv_path = args.metrics_csv
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # 先计算平均值
        psnr_lq   = [m["psnr_lq"]   for m in all_metrics]
        psnr_base = [m["psnr_base"] for m in all_metrics]
        psnr_mix  = [m["psnr_mix"]  for m in all_metrics]
        ssim_lq   = [m["ssim_lq"]   for m in all_metrics]
        ssim_base = [m["ssim_base"] for m in all_metrics]
        ssim_mix  = [m["ssim_mix"]  for m in all_metrics]

        mean_row = {
            "name": "__mean__",
            "psnr_lq":   float(np.mean(psnr_lq)),
            "psnr_base": float(np.mean(psnr_base)),
            "psnr_mix":  float(np.mean(psnr_mix)),
            "ssim_lq":   float(np.mean(ssim_lq)),
            "ssim_base": float(np.mean(ssim_base)),
            "ssim_mix":  float(np.mean(ssim_mix)),
        }

        # 每行都附带配置字段，方便后面统一读多个 CSV
        fieldnames = [
            "name",
            "psnr_lq", "psnr_base", "psnr_mix",
            "ssim_lq", "ssim_base", "ssim_mix",
            "run_name", "embedder", "embedder_tag",
            "sim_metric", "temperature", "topk",
            "num_images", "output_dir", "input_root", "gt_root",
        ]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in all_metrics:
                row = {
                    **m,
                    "run_name": args.run_name,
                    "embedder": args.embedder,
                    "embedder_tag": args.embedder_tag,
                    "sim_metric": args.sim_metric,
                    "temperature": args.temperature,
                    "topk": args.topk,
                    "num_images": len(all_metrics),
                    "output_dir": args.output,
                    "input_root": args.input,
                    "gt_root": args.gt_root,
                }
                writer.writerow(row)

            # 写最后一行 mean
            row = {
                **mean_row,
                "run_name": args.run_name,
                "embedder": args.embedder,
                "embedder_tag": args.embedder_tag,
                "sim_metric": args.sim_metric,
                "temperature": args.temperature,
                "topk": args.topk,
                "num_images": len(all_metrics),
                "output_dir": args.output,
                "input_root": args.input,
                "gt_root": args.gt_root,
            }
            writer.writerow(row)

        print(f"[CSV] metrics saved to {csv_path}")

    # ===== 额外 summary_csv（只写一行 mean，方便多个实验 append） =====
    if args.summary_csv is not None and len(all_metrics) > 0:
        sum_path = args.summary_csv
        os.makedirs(os.path.dirname(sum_path), exist_ok=True)

        # 若文件不存在，写 header
        write_header = (not os.path.exists(sum_path))
        with open(sum_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "run_name",
                    "embedder", "embedder_tag",
                    "sim_metric", "temperature", "topk",
                    "num_images",
                    "psnr_lq", "psnr_base", "psnr_mix",
                    "ssim_lq", "ssim_base", "ssim_mix",
                    "output_dir", "input_root", "gt_root",
                ])

            writer.writerow([
                args.run_name,
                args.embedder, args.embedder_tag,
                args.sim_metric, args.temperature, args.topk,
                len(all_metrics),
                float(np.mean(psnr_lq)),
                float(np.mean(psnr_base)),
                float(np.mean(psnr_mix)),
                float(np.mean(ssim_lq)),
                float(np.mean(ssim_base)),
                float(np.mean(ssim_mix)),
                args.output, args.input, args.gt_root,
            ])

        print(f"[CSV] summary appended to {sum_path}")


if __name__ == "__main__":
    main()

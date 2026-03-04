# lora_adapters/infer_dual_lora_compare.py
import os, csv, math
from typing import List, Tuple, Optional

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

from lora_adapters.utils import build_histoformer, save_image
from lora_adapters.inject_lora import inject_lora
from lora_adapters.lora_linear import LoRALinear, LoRAConv2d
from lora_adapters.domain_orchestrator import DomainOrchestrator

# 复用 infer_data 里的工具函数（避免重复造轮子）
from lora_adapters.infer_data import (
    tensor_psnr,
    tensor_ssim,
    load_all_domain_loras,
    set_all_lora_domain_weights,
)

from lora_adapters.common.seed import set_seed


def read_pair_list(pair_list: str) -> List[Tuple[str, Optional[str]]]:
    """
    每行：LQ_path [GT_path]
    - 只写一个路径：GT_path 为空
    - 支持空行和 # 注释
    """
    pairs = []
    with open(pair_list, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 1:
                pairs.append((parts[0], None))
            else:
                pairs.append((parts[0], parts[1]))
    return pairs


def _to_tensor(img: Image.Image) -> torch.Tensor:
    return T.ToTensor()(img)  # [3,H,W] in [0,1]


def _parse_csv_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def _startswith_any(name: str, prefixes):
    return any(name.startswith(p) for p in prefixes)


def set_dual_layerwise_domain_weights(
    net,
    global_domain: str,
    local_domain: str,
    wg: float,
    wl: float,
    global_prefixes,
    local_prefixes,
    default_policy: str = "none",  # none | both | global | local
):
    """
    对每个 LoRA 层按名字前缀分配不同域权重：
    - 匹配 local_prefixes -> 只开 local_domain
    - 匹配 global_prefixes -> 只开 global_domain
    - 其它层 -> 按 default_policy
    """
    if default_policy == "both":
        default_w = {global_domain: wg, local_domain: wl}
    elif default_policy == "global":
        default_w = {global_domain: wg}
    elif default_policy == "local":
        default_w = {local_domain: wl}
    else:
        default_w = None  # 其它层不启用 LoRA

    for name, m in net.named_modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            if _startswith_any(name, local_prefixes):
                m.set_domain_weights({local_domain: wl})
            elif _startswith_any(name, global_prefixes):
                m.set_domain_weights({global_domain: wg})
            else:
                m.set_domain_weights(default_w)


def _sigmoid(x: float) -> float:
    # 稳定一些的 sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def set_dual_ramp_domain_weights(
    net,
    global_domain: str,
    local_domain: str,
    wg: float,
    wl: float,
    p0: float = 0.55,      # 过渡中心（0~1）
    k: float = 12.0,       # 过渡陡峭度（越大越接近硬切）
    eps: float = 1e-4,     # 小于 eps 的权重裁掉
    default_policy: str = "none",  # 预留接口：none | both | global | local（目前 ramp 会覆盖所有 LoRA 层）
):
    """
    Ramp 版层级路由（连续、可重叠）：
    - 遍历所有 LoRA 层（按出现顺序定义深度 p=i/(N-1)）
    - g = sigmoid(k*(p - p0)) : 越深 g 越接近 1（global 占主）
    - 本层权重：
        w_global = wg * g
        w_local  = wl * (1 - g)
    - 小于 eps 的权重裁掉；若都裁掉则设 None（不启用 LoRA）
    """
    loras = []
    for name, m in net.named_modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            loras.append((name, m))

    n = len(loras)
    if n == 0:
        return

    for i, (name, m) in enumerate(loras):
        p = 0.0 if n == 1 else (i / (n - 1))
        g = _sigmoid(k * (p - p0))

        w_g = float(wg) * float(g)
        w_l = float(wl) * float(1.0 - g)

        w_dict = {}
        if w_g >= eps:
            w_dict[global_domain] = w_g
        if w_l >= eps:
            w_dict[local_domain] = w_l

        m.set_domain_weights(w_dict if w_dict else None)


def main():
    import argparse
    ap = argparse.ArgumentParser("Dual-domain LoRA manual activation test (Base vs Dual-LoRA)")

    # data
    ap.add_argument("--pair_list", type=str, required=True,
                    help='TXT 每行 "LQ_path [GT_path]"')
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--save_images", action="store_true",
                    help="保存 base / dual 输出图（可选）")

    # model
    ap.add_argument("--base_ckpt", type=str, required=True)
    ap.add_argument("--yaml", type=str, default=None)
    ap.add_argument("--loradb", type=str, default="loradb",
                    help="LoRA 库根目录（每域子目录在此下）")
    ap.add_argument("--domains", type=str, required=True,
                    help="可用域列表（用于注入 + 加载 LoRA），如 rain,haze,snow,low")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=8.0)

    # dual choice
    ap.add_argument("--domain_a", type=str, required=True)
    ap.add_argument("--domain_b", type=str, required=True)
    ap.add_argument("--wa", type=float, default=0.5)
    ap.add_argument("--wb", type=float, default=0.5)
    ap.add_argument("--normalize_weights", action="store_true",
                    help="将 wa,wb 归一化到和为 1（可选）")

    # runtime
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")

    # dual variant
    ap.add_argument("--dual_variant", type=str, default="all",
                    choices=[
                        "all",
                        "layerwise",
                        "ramp",
                        "both",               # all + layerwise
                        "all_ramp",           # all + ramp
                        "layerwise_ramp",     # layerwise + ramp
                        "all_layerwise_ramp"  # all + layerwise + ramp
                    ],
                    help="all=全层双LoRA；layerwise=prefix硬分层；ramp=连续深度过渡；其余为组合对比")

    # layerwise params
    ap.add_argument("--global_domain", type=str, default=None,
                    help="layerwise/ramp: 全局域(默认=domain_a)，如 haze/low")
    ap.add_argument("--local_domain", type=str, default=None,
                    help="layerwise/ramp: 局部域(默认=domain_b)，如 rain/snow")

    ap.add_argument("--global_prefixes", type=str,
                    default="encoder_level3,latent,decoder_level2,decoder_level3",
                    help="layerwise: 哪些模块名前缀算全局层(逗号分隔)")
    ap.add_argument("--local_prefixes", type=str,
                    default="encoder_level1,encoder_level2,decoder_level1,refinement",
                    help="layerwise: 哪些模块名前缀算局部层(逗号分隔)")

    ap.add_argument("--default_policy", type=str, default="none",
                    choices=["none", "both", "global", "local"],
                    help="layerwise: 未命中前缀的层如何处理")

    # ramp params
    ap.add_argument("--ramp_p0", type=float, default=0.55,
                    help="ramp: 过渡中心 p0（0~1），越大越偏深层才global")
    ap.add_argument("--ramp_k", type=float, default=12.0,
                    help="ramp: 过渡陡峭度 k，越大越接近硬切")
    ap.add_argument("--ramp_eps", type=float, default=1e-4,
                    help="ramp: 小权重裁剪阈值 eps")

    # csv
    ap.add_argument("--csv_name", type=str, default="dual_lora_compare.csv")

    args = ap.parse_args()

    # reproducibility
    set_seed(args.seed, deterministic=args.deterministic)

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    doms = [d.strip() for d in args.domains.split(",") if d.strip()]
    assert args.domain_a in doms, f"--domain_a={args.domain_a} 不在 --domains={doms}"
    assert args.domain_b in doms, f"--domain_b={args.domain_b} 不在 --domains={doms}"

    wa, wb = float(args.wa), float(args.wb)
    if args.normalize_weights:
        s = wa + wb
        if s > 1e-8:
            wa, wb = wa / s, wb / s

    global_domain = args.global_domain or args.domain_a
    local_domain  = args.local_domain  or args.domain_b
    global_prefixes = _parse_csv_list(args.global_prefixes)
    local_prefixes  = _parse_csv_list(args.local_prefixes)

    print(f"[INFO] device={device}")
    print(f"[INFO] domains(all)={doms}")
    print(f"[INFO] dual activate (all): {args.domain_a}={wa:.4f}, {args.domain_b}={wb:.4f}")
    print(f"[INFO] dual_variant={args.dual_variant}")
    if args.dual_variant in ("layerwise", "both", "layerwise_ramp", "all_layerwise_ramp"):
        print(f"[INFO] layerwise: global={global_domain}({wa}), local={local_domain}({wb})")
        print(f"[INFO] layerwise global_prefixes={global_prefixes}")
        print(f"[INFO] layerwise local_prefixes={local_prefixes}")
        print(f"[INFO] layerwise default_policy={args.default_policy}")
    if args.dual_variant in ("ramp", "all_ramp", "layerwise_ramp", "all_layerwise_ramp"):
        print(f"[INFO] ramp: global={global_domain}({wa}), local={local_domain}({wb})")
        print(f"[INFO] ramp params: p0={args.ramp_p0}, k={args.ramp_k}, eps={args.ramp_eps}")

    # 1) build base + inject LoRA
    print("[INFO] building base Histoformer...")
    net = build_histoformer(weights=args.base_ckpt, yaml_file=args.yaml).to(device)
    net = inject_lora(
        net,
        rank=args.rank,
        domain_list=doms,
        alpha=args.alpha,
        target_names=None,
        patterns=None,
    ).to(device)
    net.eval()

    # 2) build orchestrator only for loading LoRA paths (不走路由)
    orch = DomainOrchestrator(
        doms,
        lora_db_path=args.loradb,
        sim_metric="cosine",
        temperature=0.07,
        embedder_tag="noop"
    )

    print("[INFO] loading all domain LoRA weights into model branches...")
    load_all_domain_loras(net, orch)

    # 3) read pairs
    pairs = read_pair_list(args.pair_list)
    print(f"[INFO] total pairs = {len(pairs)}")

    # 4) loop
    rows = []
    psnr_base_list, ssim_base_list = [], []

    # 兼容旧字段：存“主 dual”（优先 layerwise，其次 ramp，其次 all）
    psnr_dual_list, ssim_dual_list = [], []

    psnr_all_list,  ssim_all_list  = [], []
    psnr_lw_list,   ssim_lw_list   = [], []
    psnr_ramp_list, ssim_ramp_list = [], []

    factor = 8  # pad to multiple of 8

    for idx, (lq_path, gt_path) in enumerate(pairs):
        # load lq
        lq = Image.open(lq_path).convert("RGB")
        x = _to_tensor(lq).unsqueeze(0).to(device)  # [1,3,H,W]
        _, _, h, w = x.shape

        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h or pad_w:
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_in = x

        # load gt if provided
        gt = None
        if gt_path is not None and os.path.isfile(gt_path):
            gt_img = Image.open(gt_path).convert("RGB")
            gt = _to_tensor(gt_img).unsqueeze(0).to(device)

        with torch.no_grad():
            # ---- Base (no LoRA) ----
            # 用 {} 真正关闭 LoRA，避免单域时 None 被解释为“默认开启”
            set_all_lora_domain_weights(net, {})
            y_base = net(x_in)
            if pad_h or pad_w:
                y_base = y_base[..., :h, :w]
            y_base = y_base.clamp(0, 1)

            y_dual_all  = None
            y_dual_lw   = None
            y_dual_ramp = None

            # ---- Dual LoRA (ALL layers) ----
            if args.dual_variant in ("all", "both", "all_ramp", "all_layerwise_ramp"):
                w_dict = {args.domain_a: wa, args.domain_b: wb}
                set_all_lora_domain_weights(net, w_dict)
                y_dual_all = net(x_in)
                if pad_h or pad_w:
                    y_dual_all = y_dual_all[..., :h, :w]
                y_dual_all = y_dual_all.clamp(0, 1)

            # ---- Dual LoRA (LAYERWISE hard split) ----
            if args.dual_variant in ("layerwise", "both", "layerwise_ramp", "all_layerwise_ramp"):
                set_dual_layerwise_domain_weights(
                    net,
                    global_domain=global_domain,
                    local_domain=local_domain,
                    wg=wa, wl=wb,
                    global_prefixes=global_prefixes,
                    local_prefixes=local_prefixes,
                    default_policy=args.default_policy,
                )
                y_dual_lw = net(x_in)
                if pad_h or pad_w:
                    y_dual_lw = y_dual_lw[..., :h, :w]
                y_dual_lw = y_dual_lw.clamp(0, 1)

            # ---- Dual LoRA (RAMP continuous blend) ----
            if args.dual_variant in ("ramp", "all_ramp", "layerwise_ramp", "all_layerwise_ramp"):
                set_dual_ramp_domain_weights(
                    net,
                    global_domain=global_domain,
                    local_domain=local_domain,
                    wg=wa, wl=wb,
                    p0=args.ramp_p0,
                    k=args.ramp_k,
                    eps=args.ramp_eps,
                    default_policy=args.default_policy,
                )
                y_dual_ramp = net(x_in)
                if pad_h or pad_w:
                    y_dual_ramp = y_dual_ramp[..., :h, :w]
                y_dual_ramp = y_dual_ramp.clamp(0, 1)

            # 统一一个 “y_dual” 方便兼容旧逻辑：优先 layerwise，其次 ramp，否则 all
            y_dual = y_dual_lw if (y_dual_lw is not None) else (y_dual_ramp if (y_dual_ramp is not None) else y_dual_all)

        # metrics
        psnr_base = ssim_base = None
        psnr_dual = ssim_dual = None
        psnr_all = ssim_all = None
        psnr_lw = ssim_lw = None
        psnr_ramp = ssim_ramp = None

        if gt is not None:
            psnr_base = tensor_psnr(y_base, gt)
            ssim_base = tensor_ssim(y_base, gt)
            psnr_base_list.append(psnr_base)
            ssim_base_list.append(ssim_base)

            if y_dual_all is not None:
                psnr_all = tensor_psnr(y_dual_all, gt)
                ssim_all = tensor_ssim(y_dual_all, gt)
                psnr_all_list.append(psnr_all)
                ssim_all_list.append(ssim_all)

            if y_dual_lw is not None:
                psnr_lw = tensor_psnr(y_dual_lw, gt)
                ssim_lw = tensor_ssim(y_dual_lw, gt)
                psnr_lw_list.append(psnr_lw)
                ssim_lw_list.append(ssim_lw)

            if y_dual_ramp is not None:
                psnr_ramp = tensor_psnr(y_dual_ramp, gt)
                ssim_ramp = tensor_ssim(y_dual_ramp, gt)
                psnr_ramp_list.append(psnr_ramp)
                ssim_ramp_list.append(ssim_ramp)

            # “主 dual”
            psnr_dual = psnr_lw if (psnr_lw is not None) else (psnr_ramp if (psnr_ramp is not None) else psnr_all)
            ssim_dual = ssim_lw if (ssim_lw is not None) else (ssim_ramp if (ssim_ramp is not None) else ssim_all)
            if psnr_dual is not None:
                psnr_dual_list.append(psnr_dual)
                ssim_dual_list.append(ssim_dual)

        # save images
        if args.save_images:
            base_out = os.path.join(args.output_dir, f"{idx:04d}_base.png")
            save_image(y_base[0].cpu(), base_out)

            if y_dual_all is not None:
                all_out = os.path.join(args.output_dir, f"{idx:04d}_dualALL_{args.domain_a}-{args.domain_b}.png")
                save_image(y_dual_all[0].cpu(), all_out)

            if y_dual_lw is not None:
                lw_out = os.path.join(args.output_dir, f"{idx:04d}_dualLW_{global_domain}-{local_domain}.png")
                save_image(y_dual_lw[0].cpu(), lw_out)

            if y_dual_ramp is not None:
                ramp_out = os.path.join(
                    args.output_dir,
                    f"{idx:04d}_dualRAMP_{global_domain}-{local_domain}_p0{args.ramp_p0}_k{args.ramp_k}.png"
                )
                save_image(y_dual_ramp[0].cpu(), ramp_out)

        rows.append({
            "idx": idx,
            "lq_path": lq_path,
            "gt_path": gt_path or "",
            "domain_a": args.domain_a,
            "domain_b": args.domain_b,
            "wa": wa, "wb": wb,
            "dual_variant": args.dual_variant,
            "global_domain": global_domain,
            "local_domain": local_domain,
            "ramp_p0": args.ramp_p0,
            "ramp_k": args.ramp_k,
            "ramp_eps": args.ramp_eps,
            "psnr_base": psnr_base if psnr_base is not None else "",
            "ssim_base": ssim_base if ssim_base is not None else "",
            "psnr_dual": psnr_dual if psnr_dual is not None else "",
            "ssim_dual": ssim_dual if ssim_dual is not None else "",
            "psnr_dual_all": psnr_all if psnr_all is not None else "",
            "ssim_dual_all": ssim_all if ssim_all is not None else "",
            "psnr_dual_lw": psnr_lw if psnr_lw is not None else "",
            "ssim_dual_lw": ssim_lw if ssim_lw is not None else "",
            "psnr_dual_ramp": psnr_ramp if psnr_ramp is not None else "",
            "ssim_dual_ramp": ssim_ramp if ssim_ramp is not None else "",
        })

        if (idx + 1) % 10 == 0:
            print(f"[INFO] processed {idx+1}/{len(pairs)}")

    # 5) write csv
    csv_path = os.path.join(args.output_dir, args.csv_name)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

        # summary
        if len(psnr_base_list) > 0:
            mean_psnr_base = float(np.mean(psnr_base_list))
            mean_ssim_base = float(np.mean(ssim_base_list))

            mean_psnr_dual = float(np.mean(psnr_dual_list)) if len(psnr_dual_list) else None
            mean_ssim_dual = float(np.mean(ssim_dual_list)) if len(ssim_dual_list) else None

            mean_psnr_all = float(np.mean(psnr_all_list)) if len(psnr_all_list) else None
            mean_ssim_all = float(np.mean(ssim_all_list)) if len(ssim_all_list) else None

            mean_psnr_lw = float(np.mean(psnr_lw_list)) if len(psnr_lw_list) else None
            mean_ssim_lw = float(np.mean(ssim_lw_list)) if len(ssim_lw_list) else None

            mean_psnr_ramp = float(np.mean(psnr_ramp_list)) if len(psnr_ramp_list) else None
            mean_ssim_ramp = float(np.mean(ssim_ramp_list)) if len(ssim_ramp_list) else None

            writer.writerow({})
            writer.writerow({
                "idx": "MEAN",
                "psnr_base": mean_psnr_base,
                "ssim_base": mean_ssim_base,
                "psnr_dual": mean_psnr_dual if mean_psnr_dual is not None else "",
                "ssim_dual": mean_ssim_dual if mean_ssim_dual is not None else "",
                "psnr_dual_all": mean_psnr_all if mean_psnr_all is not None else "",
                "ssim_dual_all": mean_ssim_all if mean_ssim_all is not None else "",
                "psnr_dual_lw": mean_psnr_lw if mean_psnr_lw is not None else "",
                "ssim_dual_lw": mean_ssim_lw if mean_ssim_lw is not None else "",
                "psnr_dual_ramp": mean_psnr_ramp if mean_psnr_ramp is not None else "",
                "ssim_dual_ramp": mean_ssim_ramp if mean_ssim_ramp is not None else "",
            })

            print("[SUMMARY]")
            print(f"  BASE    : PSNR={mean_psnr_base:.4f}, SSIM={mean_ssim_base:.4f}")
            if mean_psnr_all is not None:
                print(f"  DUAL_ALL: PSNR={mean_psnr_all:.4f}, SSIM={mean_ssim_all:.4f}")
            if mean_psnr_lw is not None:
                print(f"  DUAL_LW : PSNR={mean_psnr_lw:.4f}, SSIM={mean_ssim_lw:.4f}")
            if mean_psnr_ramp is not None:
                print(f"  DUAL_RAMP: PSNR={mean_psnr_ramp:.4f}, SSIM={mean_ssim_ramp:.4f}")
            if mean_psnr_dual is not None:
                print(f"  DUAL    : PSNR={mean_psnr_dual:.4f}, SSIM={mean_ssim_dual:.4f}")
                print(f"  DELTA   : dPSNR={mean_psnr_dual-mean_psnr_base:+.4f}, dSSIM={mean_ssim_dual-mean_ssim_base:+.4f}")

    print(f"[DONE] csv saved to {csv_path}")
    print(f"[DONE] outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()

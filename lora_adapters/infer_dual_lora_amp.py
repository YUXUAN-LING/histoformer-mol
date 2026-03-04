# lora_adapters/infer_dual_lora_compare.py
import os, csv, math, time
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

from lora_adapters.utils import build_histoformer, save_image
from lora_adapters.inject_lora import inject_lora
from lora_adapters.lora_linear import LoRALinear, LoRAConv2d
from lora_adapters.domain_orchestrator import DomainOrchestrator

# 复用 infer_data 里的工具函数
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
    if s is None:
        return []
    s = s.replace("，", ",")
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> List[float]:
    s = (s or "").replace("，", ",")
    out = []
    for t in s.split(","):
        t = t.strip()
        if not t:
            continue
        out.append(float(t))
    return out


def _parse_int_list(s: str) -> List[int]:
    s = (s or "").replace("，", ",")
    out = []
    for t in s.split(","):
        t = t.strip()
        if not t:
            continue
        out.append(int(float(t)))
    return out


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
    prefix 硬分层：
    - local_prefixes -> 只开 local_domain
    - global_prefixes -> 只开 global_domain
    - 其它层 -> default_policy
    """
    if default_policy == "both":
        default_w = {global_domain: wg, local_domain: wl}
    elif default_policy == "global":
        default_w = {global_domain: wg}
    elif default_policy == "local":
        default_w = {local_domain: wl}
    else:
        default_w = None

    for name, m in net.named_modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            if _startswith_any(name, local_prefixes):
                m.set_domain_weights({local_domain: wl})
            elif _startswith_any(name, global_prefixes):
                m.set_domain_weights({global_domain: wg})
            else:
                m.set_domain_weights(default_w)


def _sigmoid(x: float) -> float:
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
    p0: float = 0.55,
    k: float = 12.0,
    eps: float = 0.0,
):
    """
    Ramp 连续过渡（按 LoRA 模块出现顺序定义“深度”）：
      g = sigmoid(k*(p - p0))
      w_global = wg * g
      w_local  = wl * (1-g)
    eps>0 会裁剪很小的权重（更像硬切）；网格搜索建议 eps=0
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
        g = _sigmoid(float(k) * (p - float(p0)))

        w_g = float(wg) * float(g)
        w_l = float(wl) * float(1.0 - g)

        w_dict = {}
        if w_g >= eps:
            w_dict[global_domain] = w_g
        if w_l >= eps:
            w_dict[local_domain] = w_l

        m.set_domain_weights(w_dict if w_dict else None)


@torch.no_grad()
def _forward(net, x_in, h, w, pad_h, pad_w, amp: bool, device: str):
    if amp and device.startswith("cuda"):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = net(x_in)
    else:
        y = net(x_in)
    if pad_h or pad_w:
        y = y[..., :h, :w]
    return y.clamp(0, 1)


def _append_rows_csv(csv_path: str, fieldnames: List[str], rows: List[Dict]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser("Dual-domain LoRA compare + Ramp grid search")

    # data
    ap.add_argument("--pair_list", type=str, required=True, help='TXT 每行 "LQ_path [GT_path]"')
    ap.add_argument("--output_dir", type=str, required=True)

    # model
    ap.add_argument("--base_ckpt", type=str, required=True)
    ap.add_argument("--yaml", type=str, default=None)
    ap.add_argument("--loradb", type=str, default="loradb", help="LoRA 库根目录（每域子目录在此下）")
    ap.add_argument("--domains", type=str, required=True, help="可用域列表（用于注入 + 加载 LoRA）")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=8.0)

    # dual choice
    ap.add_argument("--domain_a", type=str, required=True)
    ap.add_argument("--domain_b", type=str, required=True)
    ap.add_argument("--wa", type=float, default=1.0)
    ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--normalize_weights", action="store_true")

    # runtime
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--amp", action="store_true", help="推理使用 autocast(fp16) 省显存/加速")

    # baseline variants
    ap.add_argument("--save_images", action="store_true", help="保存 base/all/lw 输出图（网格模式默认不保存）")

    # layerwise params
    ap.add_argument("--global_domain", type=str, default=None)
    ap.add_argument("--local_domain", type=str, default=None)
    ap.add_argument("--global_prefixes", type=str,
                    default="encoder_level3,latent,decoder_level2,decoder_level3")
    ap.add_argument("--local_prefixes", type=str,
                    default="encoder_level1,encoder_level2,decoder_level1,refinement")
    ap.add_argument("--default_policy", type=str, default="none",
                    choices=["none", "both", "global", "local"])

    # ramp single-run params
    ap.add_argument("--ramp_p0", type=float, default=0.55)
    ap.add_argument("--ramp_k", type=float, default=12.0)
    ap.add_argument("--ramp_eps", type=float, default=0.0)

    # ---- NEW: ramp grid search ----
    ap.add_argument("--grid_ramp", action="store_true",
                    help="开启 ramp 网格搜索：遍历多个 p0/k 并把 summary 追加到总 CSV")
    ap.add_argument("--grid_ramp_ks", type=str, default="4,6,8,12",
                    help="网格 k 列表（逗号分隔）")
    ap.add_argument("--grid_ramp_p0s", type=str, default="0.40,0.45,0.50,0.55,0.65",
                    help="网格 p0 列表（逗号分隔）")
    ap.add_argument("--grid_ramp_eps", type=float, default=0.0,
                    help="网格搜索固定 eps（建议 0）")
    ap.add_argument("--grid_csv_name", type=str, default="ramp_grid_summary.csv",
                    help="总 summary CSV 文件名（在 output_dir 下；存在则追加）")

    args = ap.parse_args()

    # reproducibility
    set_seed(args.seed, deterministic=args.deterministic)

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    doms = [d.strip() for d in _parse_csv_list(args.domains) if d.strip()]
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

    print(f"[INFO] device={device} amp={args.amp}")
    print(f"[INFO] domains(all)={doms}")
    print(f"[INFO] dual ALL: {args.domain_a}={wa:.4f}, {args.domain_b}={wb:.4f}")
    print(f"[INFO] layerwise: global={global_domain}, local={local_domain}, default_policy={args.default_policy}")
    print(f"[INFO] pair_list={args.pair_list}")

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

    # 2) build orchestrator only for loading LoRA weights
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

    # grid setup
    grid_ks = _parse_int_list(args.grid_ramp_ks) if args.grid_ramp else []
    grid_p0s = _parse_float_list(args.grid_ramp_p0s) if args.grid_ramp else []
    grid_eps = float(args.grid_ramp_eps)

    if args.grid_ramp:
        print(f"[GRID] ks={grid_ks}")
        print(f"[GRID] p0s={grid_p0s}")
        print(f"[GRID] eps={grid_eps}")

    # accumulators (baseline)
    base_psnr, base_ssim, n_eval = 0.0, 0.0, 0
    all_psnr,  all_ssim  = 0.0, 0.0
    lw_psnr,   lw_ssim   = 0.0, 0.0

    # accumulators (grid ramp)
    ramp_sum = {}   # (p0,k) -> dict(sum_psnr,sum_ssim,count)

    for p0 in grid_p0s:
        for k in grid_ks:
            ramp_sum[(float(p0), int(k))] = {"psnr": 0.0, "ssim": 0.0, "count": 0}

    factor = 8  # pad to multiple of 8

    t0 = time.time()

    # 4) loop images
    for idx, (lq_path, gt_path) in enumerate(pairs):
        if gt_path is None or (not os.path.isfile(gt_path)):
            # 网格搜索一定要有 GT，否则 summary 没意义
            print(f"[warn] skip (no GT): {lq_path} | gt={gt_path}")
            continue

        lq = Image.open(lq_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        x = _to_tensor(lq).unsqueeze(0).to(device)
        gt = _to_tensor(gt_img).unsqueeze(0).to(device)

        _, _, h, w = x.shape
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else x

        # ---- BASE ----
        set_all_lora_domain_weights(net, None)
        y_base = _forward(net, x_in, h, w, pad_h, pad_w, args.amp, device)

        # ---- DUAL_ALL ----
        set_all_lora_domain_weights(net, {args.domain_a: wa, args.domain_b: wb})
        y_all = _forward(net, x_in, h, w, pad_h, pad_w, args.amp, device)

        # ---- DUAL_LW ----
        set_dual_layerwise_domain_weights(
            net,
            global_domain=global_domain,
            local_domain=local_domain,
            wg=wa, wl=wb,
            global_prefixes=global_prefixes,
            local_prefixes=local_prefixes,
            default_policy=args.default_policy,
        )
        y_lw = _forward(net, x_in, h, w, pad_h, pad_w, args.amp, device)

        # metrics baseline
        base_psnr += tensor_psnr(y_base, gt)
        base_ssim += tensor_ssim(y_base, gt)
        all_psnr  += tensor_psnr(y_all, gt)
        all_ssim  += tensor_ssim(y_all, gt)
        lw_psnr   += tensor_psnr(y_lw, gt)
        lw_ssim   += tensor_ssim(y_lw, gt)
        n_eval += 1

        # optional save baseline images (只保存一次即可)
        if args.save_images and (idx == 0):
            save_image(y_base[0].cpu(), os.path.join(args.output_dir, f"{idx:04d}_base.png"))
            save_image(y_all[0].cpu(),  os.path.join(args.output_dir, f"{idx:04d}_dualALL_{args.domain_a}-{args.domain_b}.png"))
            save_image(y_lw[0].cpu(),   os.path.join(args.output_dir, f"{idx:04d}_dualLW_{global_domain}-{local_domain}.png"))

        # ---- GRID RAMP ----
        if args.grid_ramp:
            for (p0, k), acc in ramp_sum.items():
                set_dual_ramp_domain_weights(
                    net,
                    global_domain=global_domain,
                    local_domain=local_domain,
                    wg=wa, wl=wb,
                    p0=p0,
                    k=float(k),
                    eps=grid_eps,
                )
                y_ramp = _forward(net, x_in, h, w, pad_h, pad_w, args.amp, device)
                acc["psnr"] += tensor_psnr(y_ramp, gt)
                acc["ssim"] += tensor_ssim(y_ramp, gt)
                acc["count"] += 1

        if (idx + 1) % 10 == 0:
            print(f"[INFO] processed {idx+1}/{len(pairs)} (eval={n_eval})")

    if n_eval == 0:
        print("[ERROR] no valid GT pairs evaluated. Check your pair_list format/paths.")
        return

    # baseline means
    mean_base_psnr = base_psnr / n_eval
    mean_base_ssim = base_ssim / n_eval
    mean_all_psnr  = all_psnr / n_eval
    mean_all_ssim  = all_ssim / n_eval
    mean_lw_psnr   = lw_psnr / n_eval
    mean_lw_ssim   = lw_ssim / n_eval

    print("[SUMMARY]")
    print(f"  BASE    : PSNR={mean_base_psnr:.4f}, SSIM={mean_base_ssim:.4f}")
    print(f"  DUAL_ALL: PSNR={mean_all_psnr:.4f}, SSIM={mean_all_ssim:.4f}")
    print(f"  DUAL_LW : PSNR={mean_lw_psnr:.4f}, SSIM={mean_lw_ssim:.4f}")

    # 5) write grid summary csv
    if args.grid_ramp:
        grid_csv_path = os.path.join(args.output_dir, args.grid_csv_name)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        fieldnames = [
            "time",
            "pair_list",
            "n_eval",
            "base_ckpt",
            "yaml",
            "loradb",
            "domains",
            "rank",
            "alpha",
            "seed",
            "deterministic",
            "amp",
            "domain_a",
            "domain_b",
            "wa",
            "wb",
            "global_domain",
            "local_domain",
            "default_policy",
            "grid_eps",
            "p0",
            "k",
            "mean_psnr_base",
            "mean_ssim_base",
            "mean_psnr_all",
            "mean_ssim_all",
            "mean_psnr_lw",
            "mean_ssim_lw",
            "mean_psnr_ramp",
            "mean_ssim_ramp",
            "dpsnr_ramp_vs_base",
            "dssim_ramp_vs_base",
            "dpsnr_ramp_vs_all",
            "dssim_ramp_vs_all",
        ]

        rows = []
        best_psnr = (-1e9, None)  # (psnr, (p0,k))
        best_ssim = (-1e9, None)

        for (p0, k), acc in sorted(ramp_sum.items(), key=lambda x: (x[0][0], x[0][1])):
            cnt = acc["count"]
            if cnt == 0:
                continue
            mean_ramp_psnr = acc["psnr"] / cnt
            mean_ramp_ssim = acc["ssim"] / cnt

            if mean_ramp_psnr > best_psnr[0]:
                best_psnr = (mean_ramp_psnr, (p0, k))
            if mean_ramp_ssim > best_ssim[0]:
                best_ssim = (mean_ramp_ssim, (p0, k))

            rows.append({
                "time": now,
                "pair_list": args.pair_list,
                "n_eval": n_eval,
                "base_ckpt": args.base_ckpt,
                "yaml": args.yaml or "",
                "loradb": args.loradb,
                "domains": ",".join(doms),
                "rank": args.rank,
                "alpha": args.alpha,
                "seed": args.seed,
                "deterministic": int(args.deterministic),
                "amp": int(args.amp),
                "domain_a": args.domain_a,
                "domain_b": args.domain_b,
                "wa": wa,
                "wb": wb,
                "global_domain": global_domain,
                "local_domain": local_domain,
                "default_policy": args.default_policy,
                "grid_eps": grid_eps,
                "p0": p0,
                "k": k,
                "mean_psnr_base": mean_base_psnr,
                "mean_ssim_base": mean_base_ssim,
                "mean_psnr_all": mean_all_psnr,
                "mean_ssim_all": mean_all_ssim,
                "mean_psnr_lw": mean_lw_psnr,
                "mean_ssim_lw": mean_lw_ssim,
                "mean_psnr_ramp": mean_ramp_psnr,
                "mean_ssim_ramp": mean_ramp_ssim,
                "dpsnr_ramp_vs_base": mean_ramp_psnr - mean_base_psnr,
                "dssim_ramp_vs_base": mean_ramp_ssim - mean_base_ssim,
                "dpsnr_ramp_vs_all": mean_ramp_psnr - mean_all_psnr,
                "dssim_ramp_vs_all": mean_ramp_ssim - mean_all_ssim,
            })

        _append_rows_csv(grid_csv_path, fieldnames, rows)

        print(f"[DONE] ramp grid summary appended to: {grid_csv_path}")
        if best_psnr[1] is not None:
            (p0b, kb) = best_psnr[1]
            print(f"[BEST-PSNR] p0={p0b} k={kb} | PSNR={best_psnr[0]:.4f}")
        if best_ssim[1] is not None:
            (p0b, kb) = best_ssim[1]
            print(f"[BEST-SSIM] p0={p0b} k={kb} | SSIM={best_ssim[0]:.4f}")

    print(f"[TIME] total_sec={time.time()-t0:.2f}")


if __name__ == "__main__":
    main()

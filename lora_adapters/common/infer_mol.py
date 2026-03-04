# lora_adapters/infer_mol.py
# -*- coding: utf-8 -*-
import os
import argparse
import torch
from PIL import Image
from torchvision import transforms as T

from lora_adapters.inject_lora import inject_lora
from lora_adapters.fusion.runtime import LoRARuntime
from lora_adapters.fusion.dino_weighting import load_prototypes
from lora_adapters.utils import build_histoformer, save_image, load_image

def parse():
    p = argparse.ArgumentParser(description="Dynamic fusion inference with multi-domain LoRA (DINOv2-guided)")
    p.add_argument('--input', required=True, help='输入图片或目录')
    p.add_argument('--output', required=True, help='输出图片或目录')
    p.add_argument('--base_ckpt', required=True, help='Histoformer 主干权重')
    p.add_argument('--prototypes', required=True, help='prototype 字典文件')
    p.add_argument('--domains', default="Rain,Snow,Fog,Clear", help='域名逗号分隔（与 prototypes 键一致）')
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--alpha', type=float, default=8.0)
    p.add_argument('--tau', type=float, default=7.5)
    p.add_argument('--device', default='cuda')
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse()
    device = 'cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'

    # 1) 构建/加载 Histoformer
    net = build_histoformer(weights=args.base_ckpt)

    # 2) 注入 LoRA
    domains = [d.strip() for d in args.domains.split(',') if d.strip()]
    net = inject_lora(net, rank=args.rank, domain_list=domains, alpha=args.alpha)

    # 3) 加载 LoRA 权重（可选：如果你为每个域单独存了 lora_{domain}.pth）
    #    只会加载 shape 匹配的键，不影响主干
    for d in domains:
        cand = [f"weights/lora/{d}/lora.pth", f"weights/lora/{d}/lora_{d}.pth"]
        for ck in cand:
            if os.path.isfile(ck):
                sd = torch.load(ck, map_location='cpu')
                net.load_state_dict(sd, strict=False)
                print(f"[load] LoRA for {d}: {ck}")
                break

    # 4) 载入 prototypes
    protos = load_prototypes(args.prototypes, device=device)

    # 5) 运行时（DINOv2 + 权重计算 + forward）
    runtime = LoRARuntime(net, protos, tau=args.tau, device=device)

    # 6) 处理输入
    os.makedirs(args.output, exist_ok=True)
    if os.path.isdir(args.input):
        files = [f for f in os.listdir(args.input) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif'))]
        for fn in files:
            t, _ = load_image(os.path.join(args.input, fn))
            y = runtime.infer_tensor(t)  # [1,3,H,W]
            save_image(y, os.path.join(args.output, f"restored_{fn}"))
            print(">>", fn)
    else:
        t, _ = load_image(args.input)
        y = runtime.infer_tensor(t)
        outp = args.output if args.output.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif')) \
               else os.path.join(args.output, "restored.png")
        save_image(y, outp)
        print("saved:", outp)

if __name__ == "__main__":
    main()

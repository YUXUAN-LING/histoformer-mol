# tools/infer_retrieval_dinov2.py
import os, torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from lora_adapters.utils import build_histoformer, save_image
from lora_adapters.inject_lora import inject_lora
from lora_adapters.embedding_dinov2 import DINOv2Embedder
from lora_adapters.domain_orchestrator import DomainOrchestrator
from lora_adapters.utils_merge import apply_weighted_lora, map_to_single_domain_keys

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--base_ckpt', required=True)
    ap.add_argument('--input', required=True)              # 单图或目录
    ap.add_argument('--output', required=True)
    ap.add_argument('--loradb', default='loradb')          # loradb/<Domain>/{avg_embedding.npy, *.pth}
    ap.add_argument('--domains', default='Rain,Snow,Fog,Clear')
    ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=8.0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--yaml', type=str, default=None,      # ⬅ 新增这一行
                    help='和 base_ckpt 对应的 YAML 配置文件')
    args=ap.parse_args()
    
    device='cuda' if (args.device=='cuda' and torch.cuda.is_available()) else 'cpu'

    # 1) 主干 + 注入“单域 LoRA 层”（键为 .lora_down._Single.weight / .lora_up._Single.weight）
    # net=build_histoformer(weights=args.base_ckpt)
    net = build_histoformer(weights=args.base_ckpt, yaml_file=args.yaml).to(device)
    net=inject_lora(net, rank=args.rank, domain_list=['_Single'], alpha=args.alpha, target_names=None, patterns=None)
    net.eval().to(device)

    # 2) 准备 DINOv2 embedder & LoRA 库
    emb=DINOv2Embedder(device=device)
    doms=[d.strip() for d in args.domains.split(',') if d.strip()]
    orch=DomainOrchestrator(doms, lora_db_path=args.loradb)

    def run_one(img_path):
        img = Image.open(img_path).convert('RGB')
        v = emb.embed_image(img)
        picks = orch.select_topk(v, top_k=args.topk, temperature=0.07)
        print('[topk]', picks)
        merged = orch.build_weighted_lora(picks)
        merged_for_model = map_to_single_domain_keys(merged, target_domain_name='_Single')
        apply_weighted_lora(net, merged_for_model)

        # --- 尺寸处理：pad 到 8 的倍数，再裁回 ---
        x = T.ToTensor()(img).unsqueeze(0).to(device)  # [1,3,H,W]
        _, _, h, w = x.shape

        factor = 8  # 如果后面还报 pixel_unshuffle，可以改成 16 试试
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor

        if pad_h != 0 or pad_w != 0:
            # F.pad 的顺序是 (left, right, top, bottom)
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_in = x

        with torch.no_grad():
            y = net(x_in)

        # 裁剪回原始 H,W，避免把 padding 区保留下来
        y = y[:, :, :h, :w]
        return y

    os.makedirs(args.output, exist_ok=True)
    if os.path.isdir(args.input):
        for fn in os.listdir(args.input):
            if not fn.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif')): continue
            y=run_one(os.path.join(args.input, fn))
            save_image(y, os.path.join(args.output, f"restored_{fn}"))
            print('>>', fn)
    else:
        y=run_one(args.input)
        outp=os.path.join(args.output, "restored.png") if os.path.isdir(args.output) else args.output
        save_image(y, outp)
        print('saved:', outp)

if __name__ == '__main__':
    main()

# lora_adapters/train_lora_mol.py
# -*- coding: utf-8 -*-
import os
import argparse
from typing import List, Tuple
import torch
import math
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms as T

from lora_adapters.inject_lora import inject_lora, lora_parameters, iter_lora_modules
from lora_adapters.utils import build_histoformer, load_image, save_image, _safe_load_state_dict

def parse():
    p = argparse.ArgumentParser(description="Train single-domain LoRA (freeze backbone)")
    p.add_argument('--domain', required=True, help='域名，如 Rain/Snow/Fog')
    p.add_argument('--domains', default="rain,snow,fog")
    p.add_argument('--base_ckpt', required=True, help='Histoformer 主干权重')
    p.add_argument('--train_list', required=True, help='训练列表文件；可为 单列(lq=gt) 或 双列(lq gt)')
    p.add_argument('--val_list', default=None, help='验证列表（可选）')
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--alpha', type=float, default=8.0)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--save_dir', default='weights/lora')
    p.add_argument('--device', default='cuda')
    return p.parse_args()

def read_pairs(list_file: str) -> List[Tuple[str,str]]:
    """
    读取行： 1) 'img.png' -> (img, img)  2) 'lq.png gt.png' / 'lq,gt'
    """
    pairs = []
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s: 
                continue
            if ',' in s:
                a,b = [x.strip() for x in s.split(',',1)]
            else:
                sp = s.split()
                if len(sp) == 1:
                    a,b = sp[0], sp[0]
                else:
                    a,b = sp[0], sp[1]
            pairs.append((a,b))
    return pairs

def tensor_psnr(x, y):
    # x,y: [B,3,H,W]，值范围 [0,1]
    mse = torch.mean((x - y) ** 2).item()
    if mse == 0:
        return 99.0
    return 10 * math.log10(1.0 / mse)


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs: List[Tuple[str,str]], tfm=None):
        self.pairs = pairs
        self.tfm = tfm or (lambda x: x)
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        lq_path, gt_path = self.pairs[idx]
        lq = Image.open(lq_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')
        to_tensor = T.ToTensor()
        return to_tensor(lq), to_tensor(gt)

def main():
    args = parse()
    device = 'cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'
    print("=== CUDA devices visible:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    print("Current device:", torch.cuda.current_device())

    # 1) 构建/加载 Histoformer
    net = build_histoformer(
    weights=args.base_ckpt,
    yaml_file='lora_adapters/configs/histoformer_mol.yaml',  # 用你刚才那个能跑通的路径
)

    # 2) 注入 LoRA
    domains = [d.strip() for d in args.domains.split(',') if d.strip()]
    net = inject_lora(net, rank=args.rank, domain_list=domains, alpha=args.alpha)

    # 3) 冻结主干，只训目标域 LoRA
    for p in net.parameters():
        p.requires_grad = False
    train_domain = args.domain
    for m in iter_lora_modules(net):
        for d in m.domain_list:
            req = (d == train_domain)
            m.lora_down[d].weight.requires_grad = req
            m.lora_up[d].weight.requires_grad = req
        # one-hot 权重，训练时常量
        import torch as _torch
        idx = m.domain_list.index(train_domain)
        w = _torch.zeros(len(m.domain_list))
        w[idx] = 1.0
        m.domain_weights = w

    # 4) 数据
    train_pairs = read_pairs(args.train_list)
    train_ds = PairDataset(train_pairs)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)

    if args.val_list and os.path.isfile(args.val_list):
        val_pairs = read_pairs(args.val_list)
        val_ds = PairDataset(val_pairs)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)
    else:
        val_loader = None

    net.to(device).train()
    opt = optim.Adam(lora_parameters(net, train_domain=train_domain), lr=args.lr)
    loss_fn = nn.L1Loss()

    # 5) 训练
    for ep in range(1, args.epochs+1):
        tot = 0.0
        for lq, gt in train_loader:
            lq, gt = lq.to(device), gt.to(device)
            opt.zero_grad()
            # ---- ⭐ 保证 H,W 是 8 的倍数（2 也行，我这里直接多给一点余量） ----
            B, C, H, W = lq.shape
            # Histoformer 有多级下采样，直接对齐到 8 的倍数最安全
            new_H = (H // 8) * 8
            new_W = (W // 8) * 8
            if new_H == 0 or new_W == 0:
                # 万一图太小了就跳过这个 batch
                print(f"[WARN] skip batch with size {H}x{W}")
                continue
            lq = lq[:, :, :new_H, :new_W]
            gt = gt[:, :, :new_H, :new_W]
            # ------------------------------------------------
            out = net(lq)
            loss = loss_fn(out, gt)
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"[ep {ep}] train L1: {tot/len(train_loader):.4f}")

        if val_loader:
            net.eval()
            with torch.no_grad():
                l1_sum = 0.0
                psnr_sum = 0.0

                for lq, gt in val_loader:
                    lq, gt = lq.to(device), gt.to(device)
                    out = net(lq)

                    l1_sum += loss_fn(out, gt).item()
                    psnr_sum += tensor_psnr(out.clamp(0,1), gt.clamp(0,1))

                val_l1 = l1_sum / len(val_loader)
                val_psnr = psnr_sum / len(val_loader)

                print(f"   val L1: {val_l1:.4f}, PSNR: {val_psnr:.2f} dB")
            net.train()


    # 6) 保存该域 LoRA
    os.makedirs(os.path.join(args.save_dir, train_domain), exist_ok=True)
    sd = {}
    for k, v in net.state_dict().items():
        if f".{train_domain}." in k:
            sd[k] = v.cpu()
    path = os.path.join(args.save_dir, train_domain, "lora.pth")
    torch.save(sd, path)
    print("saved:", path)

if __name__ == "__main__":
    main()

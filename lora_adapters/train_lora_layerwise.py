# lora_adapters/train_lora_layerwise.py
# -*- coding: utf-8 -*-
"""
Train / fine-tune a LoRA adapter under layerwise activation constraints.

Key idea:
- Enable (and train) LoRA only on selected module name prefixes.
- Disable LoRA on other layers by setting domain_weights to {domain: 0.0} (or empty dict).
- Optionally initialize from an existing full-layer LoRA checkpoint.

This aligns training behavior with your layerwise inference (dualLW),
so you can test whether interference is reduced after specialization.

Usage example:
python -m lora_adapters.train_lora_layerwise \
  --domain snow1_local \
  --domains snow1_local \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --train_list data/txt_lists/train_list_snow1.txt \
  --val_list data/txt_lists/val_list_snow1.txt \
  --rank 16 --alpha 16 \
  --epochs 5 --lr 5e-5 --batch 4 \
  --patch 256 --val_patch 256 \
  --enable_prefixes encoder_level1,encoder_level2,decoder_level1,refinement \
  --init_lora_ckpt weights/lora/snow1/lora.pth \
  --save_dir weights/lora_lw \
  --seed 42 --deterministic
"""

import os
import argparse
from typing import List, Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Fnn
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as VF

from lora_adapters.utils import build_histoformer
from lora_adapters.inject_lora import inject_lora
from lora_adapters.lora_linear import LoRALinear, LoRAConv2d

try:
    from lora_adapters.common.seed import set_seed
except Exception:
    # fallback
    def set_seed(seed: Optional[int], deterministic: bool = False):
        import os, random
        import numpy as np
        if seed is None:
            return
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

# optional: use your existing helper if present
try:
    from lora_adapters.lora_state import load_lora_state, map_lora_keys_to_domain
except Exception:
    def load_lora_state(ckpt_path: str, map_location: str = "cpu"):
        sd = torch.load(ckpt_path, map_location=map_location)
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        if not isinstance(sd, dict):
            raise ValueError(f"Unsupported ckpt format: {type(sd)}")
        return sd

    def map_lora_keys_to_domain(sd: dict, domain_name: str) -> dict:
        out = {}
        for k, v in sd.items():
            if "lora_" not in k or not k.endswith("weight"):
                continue
            parts = k.split(".")
            # ... lora_up.<maybeDomain>.weight
            if len(parts) >= 3 and parts[-1] == "weight" and parts[-3].startswith("lora_"):
                parts2 = parts[:-2] + [domain_name, "weight"]
                out[".".join(parts2)] = v
            # ... lora_up.weight (no domain token)
            elif len(parts) >= 2 and parts[-1] == "weight" and parts[-2].startswith("lora_"):
                parts2 = parts[:-1] + [domain_name, "weight"]
                out[".".join(parts2)] = v
        return out


def read_pairs(list_file: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "," in s:
                a, b = [x.strip() for x in s.split(",", 1)]
            else:
                sp = s.split()
                if len(sp) == 1:
                    a = sp[0]
                    b = sp[0]
                else:
                    a, b = sp[0], sp[1]
            pairs.append((a, b))
    return pairs


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, patch_size=None, is_train=True, val_patch_size=None):
        self.pairs = pairs
        self.patch_size = patch_size
        self.is_train = is_train
        self.val_patch_size = val_patch_size
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lq_path, gt_path = self.pairs[idx]
        try:
            lq = Image.open(lq_path).convert("RGB")
            gt = Image.open(gt_path).convert("RGB")
        except Exception:
            return None

        # train random crop
        if self.is_train and self.patch_size is not None and self.patch_size > 0:
            ps = self.patch_size
            w, h = lq.size
            if w < ps or h < ps:
                lq = lq.resize((ps, ps), Image.BICUBIC)
                gt = gt.resize((ps, ps), Image.BICUBIC)
            else:
                i, j, th, tw = T.RandomCrop.get_params(lq, output_size=(ps, ps))
                lq = VF.crop(lq, i, j, th, tw)
                gt = VF.crop(gt, i, j, th, tw)

        # val center crop
        if (not self.is_train) and (self.val_patch_size is not None) and (self.val_patch_size > 0):
            ps = self.val_patch_size
            w, h = lq.size
            if w < ps or h < ps:
                lq = lq.resize((ps, ps), Image.BICUBIC)
                gt = gt.resize((ps, ps), Image.BICUBIC)
            else:
                left = (w - ps) // 2
                top = (h - ps) // 2
                lq = VF.crop(lq, top, left, ps, ps)
                gt = VF.crop(gt, top, left, ps, ps)

        return self.to_tensor(lq), self.to_tensor(gt)


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    lq = torch.stack([b[0] for b in batch], dim=0)
    gt = torch.stack([b[1] for b in batch], dim=0)
    return lq, gt


def tensor_psnr(x, y):
    mse = Fnn.mse_loss(x, y).item()
    if mse == 0:
        return 99.0
    return -10 * math.log10(mse)


def parse():
    p = argparse.ArgumentParser("Train LoRA with layerwise activation constraints")
    p.add_argument("--domain", required=True, help="train domain name (can be new, e.g. snow1_local)")
    p.add_argument("--domains", required=True, help="comma-separated domain names to create LoRA branches (must include --domain)")
    p.add_argument("--base_ckpt", required=True)
    p.add_argument("--yaml", default="lora_adapters/configs/histoformer_mol.yaml")
    p.add_argument("--train_list", required=True)
    p.add_argument("--val_list", default=None)

    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--device", default="cuda")

    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--val_patch", type=int, default=256)
    p.add_argument("--val_every", type=int, default=1)

    p.add_argument("--save_dir", default="weights/lora_lw")

    # layerwise control
    p.add_argument("--enable_prefixes", type=str, required=True,
                   help="comma-separated module name prefixes where LoRA is ENABLED and TRAINED")
    p.add_argument("--default_off", action="store_true",
                   help="if set, unmatched layers will have LoRA weight=0 for this domain (recommended)")

    # init from existing ckpt
    p.add_argument("--init_lora_ckpt", type=str, default=None,
                   help="init from an existing LoRA ckpt (full-layer), will be remapped to --domain")
    p.add_argument("--zero_disabled_up", action="store_true",
                   help="after init, zero out lora_up weights on DISABLED layers for safety")

    # reproducibility
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")

    return p.parse_args()


def main():
    args = parse()
    set_seed(args.seed, deterministic=args.deterministic)

    device = "cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    train_domain = args.domain

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    if train_domain not in domains:
        domains.append(train_domain)

    enable_prefixes = [x.strip() for x in args.enable_prefixes.split(",") if x.strip()]
    print("[INFO] train_domain =", train_domain)
    print("[INFO] enable_prefixes =", enable_prefixes)
    print("[INFO] default_off =", args.default_off)

    # build + inject LoRA
    net = build_histoformer(weights=args.base_ckpt, yaml_file=args.yaml)
    net = inject_lora(net, rank=args.rank, domain_list=domains, alpha=args.alpha, enable_patch_lora=False)
    net.to(device)

    # optional init from existing ckpt (full-layer)
    if args.init_lora_ckpt:
        print("[INFO] init LoRA from:", args.init_lora_ckpt)
        sd = load_lora_state(args.init_lora_ckpt, map_location="cpu")
        sd = map_lora_keys_to_domain(sd, train_domain)
        net.load_state_dict(sd, strict=False)

    # freeze all params first
    for p in net.parameters():
        p.requires_grad = False

    # enable/train only selected LoRA layers for train_domain, and disable elsewhere
    enabled_cnt = 0
    total_lora_cnt = 0

    for name, m in net.named_modules():
        if not isinstance(m, (LoRALinear, LoRAConv2d)):
            continue
        total_lora_cnt += 1

        enabled = any(name.startswith(pf) for pf in enable_prefixes)
        if enabled:
            enabled_cnt += 1

        # set domain_weights explicitly so it works even if domain_list has only 1 domain
        if enabled:
            m.set_domain_weights({train_domain: 1.0})
        else:
            # disable this domain on this layer
            m.set_domain_weights({train_domain: 0.0} if args.default_off else None)

        # set requires_grad for LoRA params
        for d in m.domain_list:
            req = (d == train_domain) and enabled
            m.lora_down[d].weight.requires_grad = req
            m.lora_up[d].weight.requires_grad = req

        # optional safety: zero out disabled layer's lora_up so even if accidentally enabled, it contributes ~0
        if (not enabled) and args.zero_disabled_up and (train_domain in m.domain_list):
            with torch.no_grad():
                m.lora_up[train_domain].weight.zero_()

    print(f"[INFO] LoRA layers: enabled={enabled_cnt}/{total_lora_cnt}")

    # optimizer: only trainable params
    trainable = [p for p in net.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found. Check enable_prefixes / domain names.")

    opt = optim.Adam(trainable, lr=args.lr)
    loss_fn = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # data
    train_pairs = read_pairs(args.train_list)
    train_ds = PairDataset(train_pairs, patch_size=args.patch, is_train=True)
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
        collate_fn=collate_skip_none,
        generator=g,
    )
    print("[INFO] Training samples:", len(train_ds))

    val_loader = None
    if args.val_list and os.path.isfile(args.val_list):
        val_pairs = read_pairs(args.val_list)
        val_ds = PairDataset(val_pairs, patch_size=None, is_train=False, val_patch_size=args.val_patch)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=2,
            pin_memory=(device == "cuda"), collate_fn=collate_skip_none
        )
        print("[INFO] Validation samples:", len(val_ds))
    else:
        print("[INFO] No validation set provided.")

    # train
    for epoch in range(1, args.epochs + 1):
        net.train()
        running = 0.0
        steps = 0

        for it, batch in enumerate(train_loader, start=1):
            if batch is None:
                continue
            lq, gt = batch
            lq = lq.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            # divisible by 8
            _, _, H, W = lq.shape
            H2 = (H // 8) * 8
            W2 = (W // 8) * 8
            if H2 == 0 or W2 == 0:
                continue
            lq = lq[:, :, :H2, :W2]
            gt = gt[:, :, :H2, :W2]

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = net(lq)
                loss = loss_fn(out, gt)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            steps += 1
            if it % 50 == 0:
                print(f"[E{epoch:03d}][{it:05d}] loss={running/max(steps,1):.5f}")

        # val
        if val_loader is not None and (epoch % args.val_every == 0):
            net.eval()
            psnrs = []
            with torch.no_grad():
                for vb in val_loader:
                    if vb is None:
                        continue
                    lq, gt = vb
                    lq = lq.to(device)
                    gt = gt.to(device)
                    _, _, H, W = lq.shape
                    H2 = (H // 8) * 8
                    W2 = (W // 8) * 8
                    lq = lq[:, :, :H2, :W2]
                    gt = gt[:, :, :H2, :W2]
                    out = net(lq).clamp(0, 1)
                    psnrs.append(tensor_psnr(out, gt.clamp(0, 1)))
            if psnrs:
                print(f"[VAL] epoch={epoch} PSNR={sum(psnrs)/len(psnrs):.4f}")

        # save LoRA state for train_domain
        out_dir = os.path.join(args.save_dir, train_domain)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "lora.pth")

        lora_state = {
            k: v.detach().cpu()
            for k, v in net.state_dict().items()
            if (".lora_" in k and f".{train_domain}." in k)
        }
        torch.save(lora_state, out_path)
        print("[INFO] Saved:", out_path)

    print("[DONE]")


if __name__ == "__main__":
    main()

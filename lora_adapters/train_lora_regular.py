import os
import argparse
from typing import List, Tuple
import math

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torch.backends.cudnn as cudnn
import torch.nn.functional as Fnn

import glob
import random
import numpy as np

from lora_adapters.lora_state import find_lora_ckpt, load_lora_state, map_lora_keys_to_domain
from lora_adapters.regularizers.orthogonal import orth_lora_loss
try:
    from lora_adapters.common.seed import set_seed  # user-defined
except Exception:  # fallback
    def set_seed(seed: int, deterministic: bool = False):
        import os, random
        import numpy as _np
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        _np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

from lora_adapters.inject_lora import inject_lora, lora_parameters, iter_lora_modules
from lora_adapters.utils import build_histoformer

# ---- histoformer modules for sanity check ----
from basicsr.models.archs.histoformer_arch import OverlapPatchEmbed, Attention_histogram, FeedForward


# -----------------------------------------------------
# Pair list reading
# -----------------------------------------------------
def read_pairs(list_file: str) -> List[Tuple[str, str]]:
    """
    支持：
    1) "img.png" -> (img, img)
    2) "lq.png gt.png" / "lq,gt"
    """
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


# -----------------------------------------------------
# Dataset
# -----------------------------------------------------
class PairDataset(torch.utils.data.Dataset):
    """
    Dataset supporting optional random patch cropping for training,
    and optional center patch cropping for validation to avoid OOM.
    """
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        patch_size: int = None,
        is_train: bool = True,
        val_patch_size: int = None,
    ):
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
        except Exception as e:
            print(f"[WARN] 读取图像失败 (跳过): {lq_path} | 错误: {e}")
            return None

        # 训练集：随机裁剪统一 patch 大小
        if self.is_train and self.patch_size is not None:
            ps = self.patch_size
            w, h = lq.size
            if w < ps or h < ps:
                lq = lq.resize((ps, ps), Image.BICUBIC)
                gt = gt.resize((ps, ps), Image.BICUBIC)
            else:
                i, j, th, tw = T.RandomCrop.get_params(lq, output_size=(ps, ps))
                lq = F.crop(lq, i, j, th, tw)
                gt = F.crop(gt, i, j, th, tw)

        # 验证集：可选中心裁剪以避免整图 OOM
        if (not self.is_train) and (self.val_patch_size is not None):
            ps = self.val_patch_size
            w, h = lq.size
            if w < ps or h < ps:
                lq = lq.resize((ps, ps), Image.BICUBIC)
                gt = gt.resize((ps, ps), Image.BICUBIC)
            else:
                left = (w - ps) // 2
                top = (h - ps) // 2
                lq = F.crop(lq, top, left, ps, ps)
                gt = F.crop(gt, top, left, ps, ps)

        return self.to_tensor(lq), self.to_tensor(gt)


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    lq = torch.stack([b[0] for b in batch], dim=0)
    gt = torch.stack([b[1] for b in batch], dim=0)
    return lq, gt


# -----------------------------------------------------
# Validation metric
# -----------------------------------------------------
def psnr(x, y):
    mse = Fnn.mse_loss(x, y).item()
    if mse == 0:
        return 99.0
    return -10 * math.log10(mse)


# -----------------------------------------------------
# Argument Parser
# -----------------------------------------------------
def parse():
    p = argparse.ArgumentParser(description="Train single-domain LoRA for Histoformer")
    p.add_argument("--domain", required=True, help="train domain name")
    p.add_argument("--domains", required=True, help="comma-separated all domain names")
    p.add_argument("--base_ckpt", required=True, help="base histoformer ckpt")
    p.add_argument("--train_list", required=True, help="training pair list")
    p.add_argument("--val_list", type=str, default=None, help="validation pair list")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_dir", default="weights/lora")
    p.add_argument("--patch", type=int, default=256, help="random crop size for training")

    # ---- validation control ----
    p.add_argument("--no_val", action="store_true",
                   help="disable validation even if val_list is provided")
    p.add_argument("--val_every", type=int, default=1,
                   help="run validation every N epochs (default=1)")
    p.add_argument("--val_patch", type=int, default=None,
                   help="center crop patch for validation to avoid OOM, e.g. 192")

    # ---- reproducibility ----
    p.add_argument('--seed', type=int, default=42, help='random seed')
    p.add_argument('--deterministic', action='store_true', help='make cudnn deterministic (slower)')

    # ---- optional: init current domain LoRA from ckpt ----
    p.add_argument('--init_lora_ckpt', type=str, default=None,
                   help='initialize current domain LoRA from this ckpt (optional)')

    # ---- orthogonal regularization (optional) ----
    p.add_argument('--ortho_lambda', type=float, default=0.0,
                   help='weight of cross-domain orth regularizer; 0 disables')
    p.add_argument('--ortho_mode', choices=['up','down','both'], default='both')
    p.add_argument('--ortho_domains', type=str, default='',
                   help='comma-separated reference domains (default: all except --domain)')
    p.add_argument('--ortho_loradb', type=str, default='',
                   help='LoRA DB root to load reference LoRAs, e.g. weights/lora')

    return p.parse_args()


# -----------------------------------------------------
# Main Training
# -----------------------------------------------------
def main():
    args = parse()

    # [MOD] reproducibility
    set_seed(args.seed, deterministic=args.deterministic)
    # cudnn flags are handled in set_seed when deterministic=True
    if not args.deterministic:
        cudnn.benchmark = True  # accelerate convolutions

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print("=== CUDA devices visible:", torch.cuda.device_count())
    if device == "cuda":
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    print("Current device:", torch.cuda.current_device() if device == "cuda" else device)

    # -----------------------------------------------------
    # Build Histoformer + inject LoRA
    # -----------------------------------------------------
    print("[INFO] Building Histoformer...")
    net = build_histoformer(
        weights=args.base_ckpt,
        yaml_file="lora_adapters/configs/histoformer_mol.yaml",
    )

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    net = inject_lora(net, rank=args.rank, domain_list=domains, alpha=args.alpha, enable_patch_lora=False)

    # ====== 调试：检查注入位置 ======
    print("==== LoRA injection sanity check ====")
    for name, m in net.named_modules():
        if isinstance(m, OverlapPatchEmbed):
            print("[PATCH]", name, "proj type:", type(m.proj))
        if isinstance(m, Attention_histogram):
            print("[ATTN ]", name, "qkv:", type(m.qkv), "| proj_out:", type(m.project_out))
        if isinstance(m, FeedForward):
            print("[FFN  ]", name, "proj_in:", type(m.project_in), "| proj_out:", type(m.project_out))
    print("====================================")

    # -----------------------------------------------------
    # [MOD] Optional: initialize current domain LoRA / load reference LoRAs for orth-regularization
    # -----------------------------------------------------
    train_domain = args.domain

    # init current domain LoRA (optional)
    if args.init_lora_ckpt:
        print(f"[INFO] Init LoRA for domain '{train_domain}' from: {args.init_lora_ckpt}")
        sd0 = load_lora_state(args.init_lora_ckpt, map_location='cpu')
        sd0 = map_lora_keys_to_domain(sd0, train_domain)
        net.load_state_dict(sd0, strict=False)

    # load reference LoRAs for orth regularizer (optional)
    ref_domains = []
    if args.ortho_lambda and args.ortho_lambda > 0:
        if not args.ortho_loradb:
            raise ValueError("--ortho_loradb must be set when --ortho_lambda > 0")
        if args.ortho_domains.strip():
            ref_domains = [s.strip() for s in args.ortho_domains.split(',') if s.strip()]
        else:
            ref_domains = [d.strip() for d in domains if d.strip() and d.strip() != train_domain]
        # ensure no self
        ref_domains = [d for d in ref_domains if d != train_domain]
        print('[INFO] Ortho ref domains:', ref_domains)
        loaded = []
        for d in ref_domains:
            try:
                ckpt = find_lora_ckpt(args.ortho_loradb, d)
            except Exception as e:
                print(f"[WARN] skip ref domain '{d}': {e}")
                continue
            sd = load_lora_state(ckpt, map_location='cpu')
            sd = map_lora_keys_to_domain(sd, d)
            net.load_state_dict(sd, strict=False)
            loaded.append(d)
        ref_domains = loaded
        print('[INFO] Loaded ref LoRAs:', ref_domains)

    # Freeze backbone, only train specified domain
    # [MOD] train_domain already set above
    for p_ in net.parameters():
        p_.requires_grad = False

    for m in iter_lora_modules(net):
        for d in m.domain_list:
            req = (d == train_domain)
            m.lora_down[d].weight.requires_grad = req
            m.lora_up[d].weight.requires_grad = req

        # domain weights: one-hot
        idx = m.domain_list.index(train_domain)
        w = torch.zeros(len(m.domain_list))
        w[idx] = 1.0
        m.domain_weights = w

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------
    patch_size = args.patch
    print(f"[INFO] Train patch size = {patch_size}")

    train_pairs = read_pairs(args.train_list)
    train_ds = PairDataset(train_pairs, patch_size=patch_size, is_train=True)

    num_workers = 4
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_skip_none,
        generator=g,
    )
    print(f"[INFO] Training samples: {len(train_ds)}")

    if (not args.no_val) and args.val_list and os.path.isfile(args.val_list):
        val_pairs = read_pairs(args.val_list)

        # 验证默认整图；若传了 --val_patch，则中心裁 patch
        val_ds = PairDataset(
            val_pairs,
            patch_size=None,
            is_train=False,
            val_patch_size=args.val_patch
        )

        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=(device == "cuda"),
            collate_fn=collate_skip_none,
        )
        print(f"[INFO] Validation samples: {len(val_ds)}")
    else:
        val_loader = None
        print("[INFO] No validation set provided.")

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------
    net.to(device).train()
    opt = optim.Adam(lora_parameters(net, train_domain=train_domain), lr=args.lr)
    loss_fn = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        net.train()
        running_loss = 0.0

        for it, batch in enumerate(train_loader, start=1):
            if batch is None:
                continue
            lq, gt = batch
            lq = lq.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            # ensure divisible by 8 (avoid odd sizes)
            _, _, H, W = lq.shape
            new_H = (H // 8) * 8
            new_W = (W // 8) * 8
            if new_H == 0 or new_W == 0:
                print(f"[WARN] skipped tiny batch {H}x{W}")
                continue

            lq = lq[:, :, :new_H, :new_W]
            gt = gt[:, :, :new_H, :new_W]

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = net(lq)
                recon = loss_fn(out, gt)
                if args.ortho_lambda and args.ortho_lambda > 0 and len(ref_domains) > 0:
                    ortho = orth_lora_loss(net, train_domain=train_domain, ref_domains=ref_domains, mode=args.ortho_mode, normalize=True)
                    loss = recon + args.ortho_lambda * ortho
                else:
                    loss = recon

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item()

            if it % 50 == 0:
                msg = f"[E{epoch:03d}][{it:05d}] loss={running_loss/it:.5f}"
                if args.ortho_lambda and args.ortho_lambda > 0 and len(ref_domains) > 0:
                    msg += f" (recon={recon.item():.5f}, ortho~{float((loss-recon).item()/max(args.ortho_lambda,1e-12)):.5f})"
                print(msg)

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------
        if (val_loader is not None) and (epoch % args.val_every == 0):
            net.eval()
            psnr_list = []
            with torch.no_grad():
                for vb in val_loader:
                    if vb is None:
                        continue
                    lq, gt = vb
                    lq = lq.to(device)
                    gt = gt.to(device)

                    # same size handling
                    _, _, H, W = lq.shape
                    new_H = (H // 8) * 8
                    new_W = (W // 8) * 8
                    lq = lq[:, :, :new_H, :new_W]
                    gt = gt[:, :, :new_H, :new_W]

                    out = net(lq).clamp(0, 1)
                    psnr_list.append(psnr(out, gt))

            if psnr_list:
                print(f"[VAL] epoch={epoch} PSNR={sum(psnr_list)/len(psnr_list):.4f}")

        # -----------------------------------------------------
        # Save (每个 epoch 覆盖写一次；你也可以改成保存 best)
        # -----------------------------------------------------
        out_dir = os.path.join(args.save_dir, train_domain)
        os.makedirs(out_dir, exist_ok=True)

        # save only the LoRA state_dict for this domain
        lora_state = {
            k: v
            for k, v in net.state_dict().items()
            if f".lora_" in k and f".{train_domain}." in k
        }
        out_path = os.path.join(out_dir, "lora.pth")
        torch.save(lora_state, out_path)
        print(f"[INFO] Saved LoRA weights: {out_path}")


if __name__ == "__main__":
    main()

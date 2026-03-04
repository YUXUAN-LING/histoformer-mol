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

from lora_adapters.inject_lora import inject_lora, lora_parameters, iter_lora_modules
from lora_adapters.utils import build_histoformer


from basicsr.models.archs.histoformer_arch import OverlapPatchEmbed, Attention_histogram, FeedForward

# -----------------------------------------------------
# Utility
# -----------------------------------------------------
def read_pairs(list_file: str) -> List[Tuple[str, str]]:
    """
    Read lines in list_file.
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
                    a, b = sp[0], sp[0]
                else:
                    a, b = sp[0], sp[1]
            pairs.append((a, b))
    return pairs


def tensor_psnr(x, y):
    """Compute PSNR between two tensors in [0,1]."""
    mse = torch.mean((x - y) ** 2).item()
    if mse == 0:
        return 99.0
    return 10 * math.log10(1.0 / mse)


def _gaussian_kernel(window_size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    kernel_2d = g[:, None] @ g[None, :]
    kernel_2d = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return kernel_2d


def tensor_ssim(x, y, window_size=11, sigma=1.5, K1=0.01, K2=0.03):
    """
    x,y: [B,3,H,W], in [0,1]
    """
    assert x.shape == y.shape, "tensor_ssim: x,y mismatch"
    B, C, H, W = x.shape
    device = x.device

    kernel = _gaussian_kernel(window_size, sigma, C, device=device)

    # mean
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

    return p.parse_args()


# -----------------------------------------------------
# Main Training
# -----------------------------------------------------
def main():
    args = parse()
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
    net = inject_lora(net, rank=args.rank, domain_list=domains, alpha=args.alpha, enable_patch_lora=False)   # ★ 开启 patch LoRA
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


    # Freeze backbone, only train specified domain
    train_domain = args.domain
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
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(device == "cuda"),
    )

    if args.val_list and os.path.isfile(args.val_list):
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
    # Training loop
    # -----------------------------------------------------
    val_enabled = (val_loader is not None) and (not args.no_val)

    for ep in range(1, args.epochs + 1):
        net.train()
        tot = 0.0

        for lq, gt in train_loader:
            # 如果你的数据里可能有坏图返回 None，这里简单跳过
            if lq is None or gt is None:
                continue

            lq = lq.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            opt.zero_grad()

            # Ensure multiples of 8
            B, C, H, W = lq.shape
            new_H = (H // 8) * 8
            new_W = (W // 8) * 8
            if new_H == 0 or new_W == 0:
                print(f"[WARN] skipped tiny batch {H}x{W}")
                continue

            lq = lq[:, :, :new_H, :new_W]
            gt = gt[:, :, :new_H, :new_W]

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = net(lq)
                loss = loss_fn(out, gt)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tot += loss.item()

        print(f"[ep {ep}] train L1: {tot / max(1, len(train_loader)):.4f}")

        # -----------------------------------------------------
        # Validation (toggleable)
        # -----------------------------------------------------
        if val_enabled and (ep % max(1, args.val_every) == 0):
            net.eval()
            l1_sum = 0.0
            psnr_sum = 0.0
            ssim_sum = 0.0
            with torch.no_grad():
                for lq, gt in val_loader:
                    if lq is None or gt is None:
                        continue
                    lq = lq.to(device, non_blocking=True)
                    gt = gt.to(device, non_blocking=True)

                    B, C, H, W = lq.shape
                    new_H = (H // 8) * 8
                    new_W = (W // 8) * 8
                    if new_H == 0 or new_W == 0:
                        continue

                    lq = lq[:, :, :new_H, :new_W]
                    gt = gt[:, :, :new_H, :new_W]

                    out = net(lq)
                    l1_sum += loss_fn(out, gt).item()
                    psnr_sum += tensor_psnr(out.clamp(0, 1), gt.clamp(0, 1))
                    ssim_sum += tensor_ssim(out.clamp(0, 1), gt.clamp(0, 1))

            denom = max(1, len(val_loader))
            val_l1 = l1_sum / denom
            val_psnr = psnr_sum / denom
            val_ssim = ssim_sum / denom
            print(
                f"   val L1: {val_l1:.4f},  PSNR: {val_psnr:.2f} dB,  SSIM: {val_ssim:.4f}"
            )

        # -----------------------------------------------------
        # Save LoRA weights
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

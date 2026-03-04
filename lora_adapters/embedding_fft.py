# lora_adapters/embedding_fft.py
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


# =======================
# 1. 简单版：仅 amplitude（你之前用的版本）
# =======================

class FFTAmplitudeEmbedder:
    """
    最基础版本：只用振幅谱中心 patch 作为 embedding。
    现在保留作 baseline，方便对比。
    """

    def __init__(
        self,
        device: str = "cpu",
        resize: int = 256,
        center_crop: int = 128,
        out_size: int = 32,
    ):
        self.device = device
        self.resize = resize
        self.center_crop = center_crop
        self.out_size = out_size

        self.to_tensor = T.ToTensor()

    @torch.no_grad()
    def embed_image(self, img: Union[str, Path, Image.Image]) -> np.ndarray:
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert("L")
        else:
            img = img.convert("L")

        img = img.resize((self.resize, self.resize), Image.BICUBIC)
        x = self.to_tensor(img).to(self.device)[0]  # [H,W]

        F_complex = torch.fft.fft2(x)
        F_complex = torch.fft.fftshift(F_complex)

        amp = torch.abs(F_complex)
        amp = torch.log1p(amp)

        H, W = amp.shape
        c = self.center_crop // 2
        cy, cx = H // 2, W // 2
        y1, y2 = max(cy - c, 0), min(cy + c, H)
        x1, x2 = max(cx - c, 0), min(cx + c, W)
        amp_center = amp[y1:y2, x1:x2]

        amp_center = amp_center.unsqueeze(0).unsqueeze(0)
        amp_small = F.interpolate(
            amp_center,
            size=(self.out_size, self.out_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        vec = amp_small.flatten()
        vec = vec / (vec.norm(p=2) + 1e-6)
        return vec.cpu().numpy().astype(np.float32)


# =======================
# 2. 最强版：增强频域 embedding + clean prototype residual
# =======================

class FFTEnhancedEmbedder:
    """
    最强版 FFT embedding：
      - 频域振幅谱 + log 压缩
      - 径向功率谱 (radial power spectrum)
      - 方向分布 (angular spectrum)
      - 低频中心 patch (local amplitude)
      - （可选）减去 "clean prototype" 做 residual，再 L2-normalize

    最终 embedding 大致是：
      [radial_feat(Fr) ; angular_feat(Fθ) ; lowfreq_patch(P)]
      然后（可选）减去 μ_clean，再做 L2 归一。

    参数：
      resize:          输入 resize 到多少再 FFT
      radial_bins:     径向分桶个数（默认 32）
      angle_bins:      方向分桶个数（默认 16）
      patch_size:      低频中心 patch 的尺寸（默认 32 -> 1024维）
      clean_proto_path: 预先计算好的 clean prototype 向量（.npy），用于 residual
      use_residual:    是否使用 z = feat - μ_clean
    """

    def __init__(
        self,
        device: str = "cpu",
        resize: int = 256,
        radial_bins: int = 32,
        angle_bins: int = 16,
        patch_size: int = 32,
        clean_proto_path: Optional[Union[str, Path]] = None,
        use_residual: bool = True,
    ):
        self.device = device
        self.resize = resize
        self.radial_bins = radial_bins
        self.angle_bins = angle_bins
        self.patch_size = patch_size
        self.use_residual = use_residual

        self.to_tensor = T.ToTensor()

        # 缓存坐标网格，以免每张图重复创建
        self._coord_cache_size = None
        self._radius = None
        self._angle = None

        # clean prototype（频域上的“自然图像平均频谱”）
        self.clean_proto: Optional[np.ndarray] = None
        if clean_proto_path is not None:
            clean_proto_path = Path(clean_proto_path)
            if clean_proto_path.is_file():
                arr = np.load(clean_proto_path)
                self.clean_proto = arr.astype(np.float32)
                print(f"[FFTEnhancedEmbedder] loaded clean prototype from {clean_proto_path}, shape={arr.shape}")
            else:
                print(f"[FFTEnhancedEmbedder] WARNING: clean_proto_path not found: {clean_proto_path}")

    def _build_coord_cache(self, H: int, W: int, device: str):
        """
        构建 radius / angle 网格，只在尺寸变化时重建一次。
        radius:  归一化到 [0,1]
        angle:   归一化到 [0,1]，0~1 对应 0~2π
        """
        if self._coord_cache_size == (H, W):
            return

        ys = torch.arange(H, dtype=torch.float32, device=device)
        xs = torch.arange(W, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        cy = (H - 1) / 2.0
        cx = (W - 1) / 2.0
        yy = yy - cy
        xx = xx - cx

        radius = torch.sqrt(xx * xx + yy * yy)  # [H,W]
        radius = radius / (radius.max() + 1e-6)

        angle = torch.atan2(yy, xx)  # [-pi, pi]
        angle = (angle + torch.pi) / (2 * torch.pi)  # -> [0,1]

        self._radius = radius
        self._angle = angle
        self._coord_cache_size = (H, W)

    @torch.no_grad()
    def _compute_radial_feat(self, amp: torch.Tensor) -> torch.Tensor:
        """
        amp: [H,W], log-amplitude
        输出 radial_feat: [radial_bins]
        """
        H, W = amp.shape
        self._build_coord_cache(H, W, amp.device)
        radius = self._radius

        bins = torch.linspace(0.0, 1.0, steps=self.radial_bins + 1, device=amp.device)
        feats = []
        for i in range(self.radial_bins):
            r0, r1 = bins[i], bins[i + 1]
            mask = (radius >= r0) & (radius < r1)
            if mask.any():
                feats.append(amp[mask].mean())
            else:
                feats.append(torch.tensor(0.0, device=amp.device))
        radial_feat = torch.stack(feats, dim=0)  # [radial_bins]
        return radial_feat

    @torch.no_grad()
    def _compute_angular_feat(self, amp: torch.Tensor) -> torch.Tensor:
        """
        amp: [H,W], log-amplitude
        输出 angular_feat: [angle_bins]
        """
        H, W = amp.shape
        self._build_coord_cache(H, W, amp.device)
        angle = self._angle

        bins = torch.linspace(0.0, 1.0, steps=self.angle_bins + 1, device=amp.device)
        feats = []
        for i in range(self.angle_bins):
            a0, a1 = bins[i], bins[i + 1]
            mask = (angle >= a0) & (angle < a1)
            if mask.any():
                feats.append(amp[mask].mean())
            else:
                feats.append(torch.tensor(0.0, device=amp.device))
        angular_feat = torch.stack(feats, dim=0)  # [angle_bins]
        return angular_feat

    @torch.no_grad()
    def _compute_lowfreq_patch(self, amp: torch.Tensor) -> torch.Tensor:
        """
        取频谱中心一个 patch（默认 32x32），flatten -> [patch_size^2]
        """
        H, W = amp.shape
        ps = self.patch_size
        cy, cx = H // 2, W // 2
        ph = ps // 2
        y1, y2 = max(cy - ph, 0), min(cy + ph, H)
        x1, x2 = max(cx - ph, 0), min(cx + ph, W)
        patch = amp[y1:y2, x1:x2]  # [ph*2, ph*2] or clipped
        # 若 patch 尺寸不对，插值到 (ps,ps)
        if patch.shape[0] != ps or patch.shape[1] != ps:
            patch = patch.unsqueeze(0).unsqueeze(0)
            patch = F.interpolate(patch, size=(ps, ps), mode="bilinear", align_corners=False)[0, 0]
        return patch.flatten()  # [ps*ps]

    @torch.no_grad()
    def embed_image(self, img: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        返回：1D numpy 向量，float32，L2-normalized
        如果 self.clean_proto 不为空且 use_residual=True，则先做 residual：feat - clean_proto
        """
        # 1) 读图 & 灰度 & resize
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert("L")
        else:
            img = img.convert("L")

        img = img.resize((self.resize, self.resize), Image.BICUBIC)
        x = self.to_tensor(img).to(self.device)[0]  # [H,W]

        # 2) FFT + 振幅 + log
        F_complex = torch.fft.fft2(x)
        F_complex = torch.fft.fftshift(F_complex)
        amp = torch.abs(F_complex)
        amp = torch.log1p(amp)  # [H,W]

        # 3) 频域特征
        radial_feat = self._compute_radial_feat(amp)    # [R]
        angular_feat = self._compute_angular_feat(amp)  # [A]
        patch_feat = self._compute_lowfreq_patch(amp)   # [P]

        feat = torch.cat([radial_feat, angular_feat, patch_feat], dim=0)  # [R+A+P]
        feat_np = feat.cpu().numpy().astype(np.float32)

        # 4) clean prototype residual
        if self.clean_proto is not None and self.use_residual:
            if self.clean_proto.shape[0] != feat_np.shape[0]:
                print(f"[FFTEnhancedEmbedder] WARNING: clean_proto dim={self.clean_proto.shape[0]} "
                      f"!= feat_dim={feat_np.shape[0]}, skip residual.")
            else:
                feat_np = feat_np - self.clean_proto

        # 5) L2 normalize
        norm = np.linalg.norm(feat_np) + 1e-6
        feat_np = feat_np / norm

        return feat_np


# =======================
# 3. 计算 clean prototype 的辅助函数
# =======================

@torch.no_grad()
def compute_clean_prototype(
    clean_root: Union[str, Path],
    device: str = "cpu",
    resize: int = 256,
    radial_bins: int = 32,
    angle_bins: int = 16,
    patch_size: int = 32,
    max_images: int = 1000,
) -> np.ndarray:
    """
    从 clean_root 下的干净图像（无退化）中，计算最强 FFT embedding 的均值 μ_clean。

    注意：这里不会做 residual（因为就是要得到 μ_clean 本身）。
    保存后可以在 FFTEnhancedEmbedder 中作为 clean_proto 使用。
    """
    clean_root = Path(clean_root)
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    files: List[Path] = []
    for e in exts:
        files += list(clean_root.rglob(f"*{e}"))

    files = files[:max_images]
    if not files:
        raise RuntimeError(f"No images found under {clean_root}")

    print(f"[compute_clean_prototype] Found {len(files)} clean images.")

    # 用 use_residual=False，避免在这里就减某个东西
    emb = FFTEnhancedEmbedder(
        device=device,
        resize=resize,
        radial_bins=radial_bins,
        angle_bins=angle_bins,
        patch_size=patch_size,
        clean_proto_path=None,
        use_residual=False,
    )

    vecs = []
    for i, p in enumerate(files):
        try:
            v = emb.embed_image(p)
            vecs.append(v)
        except Exception as e:
            print(f"[ERROR] fail on {p}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(files)} images")

    if not vecs:
        raise RuntimeError("No valid embeddings computed for clean images.")

    arr = np.stack(vecs, axis=0).astype(np.float32)  # [N,C]
    mu_clean = arr.mean(axis=0)
    print(f"[compute_clean_prototype] mu_clean shape = {mu_clean.shape}")
    return mu_clean

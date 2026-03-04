# -*- coding: utf-8 -*-
"""lora_adapters.pipeline.metrics

PSNR / SSIM for tensors in [0,1].

These implementations are small, dependency-free, and good enough for tracking.
If you already use Basicsr's metric module, you can swap this file.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def tensor_psnr(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12) -> float:
    """PSNR for [1,C,H,W] tensors in [0,1]."""
    pred = pred.detach().clamp(0, 1)
    gt = gt.detach().clamp(0, 1)
    mse = F.mse_loss(pred, gt, reduction="mean").item()
    if mse <= eps:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device: str = "cpu"):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_1d = g.view(1, 1, -1)
    kernel_2d = g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)
    return kernel_2d


def tensor_ssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
    eps: float = 1e-12,
) -> float:
    """SSIM for [1,C,H,W] tensors in [0,1]."""
    pred = pred.detach().clamp(0, 1)
    gt = gt.detach().clamp(0, 1)
    device = pred.device

    C = pred.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, device=device)
    kernel = kernel.repeat(C, 1, 1, 1)  # groups=C

    mu1 = F.conv2d(pred, kernel, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(gt, kernel, padding=window_size // 2, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, kernel, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * gt, kernel, padding=window_size // 2, groups=C) - mu1_mu2

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + eps)
    return float(ssim_map.mean().item())

# lora_adapters/common/metrics.py
import math
import torch
import torch.nn.functional as F

def tensor_psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> float:
    """
    x,y: [B,C,H,W] in [0, data_range]
    """
    mse = torch.mean((x - y) ** 2).item()
    if mse <= 0:
        return 99.0
    return 10.0 * math.log10((data_range ** 2) / mse)

def _gaussian_kernel(window_size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2d = g[:, None] @ g[None, :]
    k2d = k2d.expand(channels, 1, window_size, window_size).contiguous()
    return k2d

def tensor_ssim(x: torch.Tensor, y: torch.Tensor,
                data_range: float = 1.0,
                window_size: int = 11, sigma: float = 1.5,
                K1=0.01, K2=0.03) -> float:
    """
    x,y: [B,C,H,W] in [0, data_range]
    """
    assert x.shape == y.shape
    B, C, H, W = x.shape
    device = x.device
    kernel = _gaussian_kernel(window_size, sigma, C, device=device)

    # normalize to [0,1] if data_range != 1
    if data_range != 1.0:
        x = x / data_range
        y = y / data_range

    mu_x = F.conv2d(x, kernel, padding=window_size // 2, groups=C)
    mu_y = F.conv2d(y, kernel, padding=window_size // 2, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=window_size // 2, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=window_size // 2, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=window_size // 2, groups=C) - mu_xy

    L = 1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))

    return ssim_map.mean().item()

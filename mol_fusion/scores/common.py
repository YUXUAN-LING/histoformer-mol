from __future__ import annotations

import torch


def delta_weight_abs_sum(module, domain: str) -> float:
    down = module.lora_down[domain].weight.detach()
    up = module.lora_up[domain].weight.detach()
    if down.ndim == 2 and up.ndim == 2:
        d = torch.matmul(up, down)
        return float(d.abs().sum().item())
    # Conv2d-friendly fallback: avoid 2D-only assumptions
    return float((up.abs().mean() * down.abs().mean()).item())


def base_weight_abs_sum(module) -> float:
    return float(module.base.weight.detach().abs().sum().item())

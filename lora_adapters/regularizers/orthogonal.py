# lora_adapters/regularizers/orthogonal.py
# -*- coding: utf-8 -*-
"""
Cross-domain LoRA orthogonal regularization.

Goal:
    Reduce interference when multiple domain LoRAs are activated together by
    encouraging updates from different domains to be (approximately) orthogonal
    under Frobenius inner product.

This module is training-only. It does NOT change inference behavior.

We regularize train_domain vs ref_domains across ALL LoRA modules:
    loss = mean_{modules, ref} ( <v_cur, v_ref>^2 )
where v is flattened LoRA weight (up / down / both). Optionally normalize so
the dot becomes cosine similarity (scale-invariant).
"""

from __future__ import annotations

from typing import List
import torch
import torch.nn.functional as F

from lora_adapters.lora_linear import LoRALinear, LoRAConv2d


def _flatten(w: torch.Tensor) -> torch.Tensor:
    return w.reshape(-1)


@torch.no_grad()
def _has_domain(m, domain: str) -> bool:
    return (hasattr(m, "lora_up") and domain in m.lora_up) and (hasattr(m, "lora_down") and domain in m.lora_down)


def orth_lora_loss(
    net: torch.nn.Module,
    train_domain: str,
    ref_domains: List[str],
    mode: str = "both",          # "up" | "down" | "both"
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute orthogonality loss between train_domain LoRA and reference domains.

    Args:
        net: model with multi-domain LoRA modules.
        train_domain: domain being optimized.
        ref_domains: list of fixed reference domains.
        mode: regularize 'up', 'down', or 'both'.
        normalize: if True, use cosine dot (scale-invariant).
    Returns:
        scalar tensor.
    """
    assert mode in ("up", "down", "both")
    device = next(net.parameters()).device
    total = torch.zeros([], device=device)
    denom = 0

    # NOTE: ref_domains are expected to be frozen (requires_grad=False) externally.
    for m in net.modules():
        if not isinstance(m, (LoRALinear, LoRAConv2d)):
            continue
        if not _has_domain(m, train_domain):
            continue

        def one_side(kind: str):
            nonlocal total, denom
            cur_w = (m.lora_up[train_domain].weight if kind == "up" else m.lora_down[train_domain].weight)
            cur_v = _flatten(cur_w).float()
            if normalize:
                cur_v = F.normalize(cur_v, dim=0, eps=eps)

            refs = []
            for d in ref_domains:
                if not _has_domain(m, d):
                    continue
                ref_w = (m.lora_up[d].weight if kind == "up" else m.lora_down[d].weight)
                ref_v = _flatten(ref_w).float()
                if normalize:
                    ref_v = F.normalize(ref_v, dim=0, eps=eps)
                refs.append(ref_v)

            if not refs:
                return

            R = torch.stack(refs, dim=0)          # [n_ref, P]
            dots = torch.matmul(R, cur_v)         # [n_ref]
            total = total + (dots ** 2).sum()
            denom += int(dots.numel())

        if mode in ("up", "both"):
            one_side("up")
        if mode in ("down", "both"):
            one_side("down")

    if denom == 0:
        return total
    return total / denom

from __future__ import annotations

import os
from typing import List

import torch
import torch.nn.functional as F

from lora_adapters.domain_orchestrator import DomainOrchestrator
from lora_adapters.infer_data import load_all_domain_loras, tensor_psnr
from lora_adapters.inject_lora import inject_lora
from lora_adapters.utils import build_histoformer

from mol_fusion.utils.image_utils import load_image_tensor


def forward_padded(net, x: torch.Tensor, factor: int = 8) -> torch.Tensor:
    _, _, h, w = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else x
    y = net(x_in)
    if isinstance(y, list):
        y = y[-1]
    return y[:, :, :h, :w]


def build_model(base_ckpt: str, yaml: str | None, rank: int, alpha: float, domains: List[str], device: str):
    net = build_histoformer(base_ckpt, yaml_file=yaml)
    net = inject_lora(net, rank=rank, domain_list=domains, alpha=alpha)
    net.to(device).eval()
    return net


def load_lora_bank(net, domains: List[str], lora_root: str):
    orch = DomainOrchestrator(domains=domains, lora_db_path=lora_root)
    load_all_domain_loras(net, orch)


def maybe_metric(y: torch.Tensor, gt_path: str | None):
    if not gt_path or not os.path.exists(gt_path):
        return None
    gt = load_image_tensor(gt_path).to(y.device)
    return tensor_psnr(y.clamp(0, 1), gt)

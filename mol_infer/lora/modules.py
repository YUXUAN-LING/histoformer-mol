# mol_infer/lora/modules.py
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch


# 适配你现有 LoRALinear / LoRAConv2d
try:
    from lora_adapter.lora_linear import LoRALinear, LoRAConv2d
except Exception as e:
    LoRALinear = None
    LoRAConv2d = None


def is_lora_module(m: Any) -> bool:
    """
    最稳的判定：结构特征
      - lora_up / lora_down (ModuleDict)
      - set_domain_weights(weights)
    """
    return hasattr(m, "lora_up") and hasattr(m, "lora_down") and hasattr(m, "set_domain_weights")


def iter_lora_modules(model: torch.nn.Module):
    """
    优先用 isinstance(LoRALinear/LoRAConv2d)，否则退化到结构特征判断。
    """
    for m in model.modules():
        if LoRALinear is not None and isinstance(m, LoRALinear):
            yield m
        elif LoRAConv2d is not None and isinstance(m, LoRAConv2d):
            yield m
        elif is_lora_module(m):
            yield m


def iter_lora_named_modules(model: torch.nn.Module) -> Iterator[Tuple[str, Any]]:
    for name, m in model.named_modules():
        if LoRALinear is not None and isinstance(m, LoRALinear):
            yield name, m
        elif LoRAConv2d is not None and isinstance(m, LoRAConv2d):
            yield name, m
        elif is_lora_module(m):
            yield name, m


def get_up_down(m: Any, domain: str):
    if domain not in m.lora_up or domain not in m.lora_down:
        raise KeyError(f"[lora] domain={domain} not in module lora_up/down keys.")
    return m.lora_up[domain], m.lora_down[domain]


def get_scale(m: Any) -> float:
    """
    你的实现是 alpha/rank
    """
    alpha = getattr(m, "alpha", None)
    rank = getattr(m, "rank", None)
    if alpha is not None and rank is not None and int(rank) > 0:
        return float(alpha) / float(rank)
    return 1.0


def set_all_lora_domain_weights_fallback(
    model: torch.nn.Module,
    weights: Optional[Union[Dict[str, float], torch.Tensor, List[float]]],
):
    """
    fallback：遍历所有 LoRA 模块，调用 m.set_domain_weights(weights)
    - weights=None => 单域训练（domain_list==1）会默认启用；多域会默认关闭（你的实现里就是这样）
    - weights=dict => 多域加权 / hard 选择
    """
    for m in iter_lora_modules(model):
        m.set_domain_weights(weights)

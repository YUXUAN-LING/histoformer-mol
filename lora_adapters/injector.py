# lora_adapters/injector.py
# 兼容旧代码的薄封装，直接转发到 inject_lora.py 中的实现

import torch.nn as nn
from typing import List, Optional, Iterable

from .inject_lora import inject_lora as _inject_lora
from .inject_lora import iter_lora_modules, lora_parameters, DEFAULT_PATTERNS


def inject_lora(
    model: nn.Module,
    rank: int,
    domain_list: List[str],
    alpha: float = 1.0,
    target_names: Optional[List[str]] = None,
    patterns: Optional[Iterable[str]] = DEFAULT_PATTERNS,
) -> nn.Module:
    """
    旧接口：
        injector.inject_lora(model, rank, domain_list, alpha, target_names)
    实际调用新的 inject_lora 并返回修改后的模型。
    """
    return _inject_lora(
        model=model,
        rank=rank,
        domain_list=domain_list,
        alpha=alpha,
        target_names=target_names,
        patterns=patterns,
    )

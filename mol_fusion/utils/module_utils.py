from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Tuple

from lora_adapters.lora_linear import LoRAConv2d, LoRALinear


def iter_lora_named_modules(model) -> Iterator[Tuple[str, object]]:
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            yield name, module


def collect_layer_names(model) -> List[str]:
    return [name for name, _ in iter_lora_named_modules(model)]


def zero_all_domains(module, domains: Iterable[str]) -> Dict[str, float]:
    return {d: 0.0 for d in domains}

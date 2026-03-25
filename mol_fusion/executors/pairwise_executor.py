from __future__ import annotations

from typing import Dict, Iterable

from mol_fusion.utils.module_utils import iter_lora_named_modules, zero_all_domains


class PairwiseExecutor:
    """Apply layer-wise pair weights onto existing multi-domain LoRA modules."""

    def __init__(self, model, all_domains: Iterable[str]):
        self.model = model
        self.all_domains = list(all_domains)

    def apply_layer_weights(self, dom1: str, dom2: str, layer_weights: Dict[str, Dict[str, float]]):
        for name, module in iter_lora_named_modules(self.model):
            ww = zero_all_domains(module, self.all_domains)
            lw = layer_weights.get(name, {dom1: 0.5, dom2: 0.5})
            ww[dom1] = float(lw.get(dom1, 0.0))
            ww[dom2] = float(lw.get(dom2, 0.0))
            s = max(ww[dom1] + ww[dom2], 1e-12)
            ww[dom1] /= s
            ww[dom2] /= s
            module.set_domain_weights(ww)

    def apply_single(self, domain: str):
        for _, module in iter_lora_named_modules(self.model):
            ww = zero_all_domains(module, self.all_domains)
            ww[domain] = 1.0
            module.set_domain_weights(ww)

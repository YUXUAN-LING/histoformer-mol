# mol_infer/lora/adapter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

# ---- your existing project imports (best-effort) ----
try:
    from lora_adapters.utils import build_histoformer
except Exception as e:
    raise ImportError(f"Cannot import build_histoformer from lora_adapters.utils: {e}")

try:
    from lora_adapter.inject_lora import inject_lora
except Exception as e:
    raise ImportError(f"Cannot import inject_lora from lora_adapters.inject_lora: {e}")

# reuse utilities you said are available in your repo
try:
    from lora_adapters.infer_data import (
        load_all_domain_loras,
        set_all_lora_domain_weights,
    )
except Exception as e:
    raise ImportError(
        f"Cannot import load_all_domain_loras/set_all_lora_domain_weights from lora_adapters.infer_data: {e}"
    )


@dataclass
class HistoformerLoRAAdapter:
    """
    Public contract used by Runner. Keep this stable.

    Responsibilities:
      - build net skeleton
      - load base weights
      - inject LoRA modules (needs domain_list!)
      - load all domain LoRA weights
      - set per-domain weights / disable
      - padded forward
    """
    base_ckpt: str
    yaml_file: Optional[str]
    device: str = "cuda"

    rank: int = 16
    alpha: float = 16.0
    enable_patch_lora: bool = False

    loradb_root: str = ""
    domains: Optional[List[str]] = None

    # runtime
    net: Optional[torch.nn.Module] = None

    def build(self):
        """
        Build model skeleton (no weights). Uses your build_histoformer().
        IMPORTANT: pass yaml_file as keyword (your build_histoformer supports this).
        """
        self.net = build_histoformer(weights=None, yaml_file=self.yaml_file)
        self.net.to(self.device)
        self.net.eval()

    def load_base(self):
        """
        Load base checkpoint robustly:
          - torch.load -> pick state_dict from common keys -> strict=False
        """
        assert self.net is not None, "Call build() first."
        ckpt = torch.load(self.base_ckpt, map_location="cpu")

        if isinstance(ckpt, dict):
            if "params" in ckpt and isinstance(ckpt["params"], dict):
                sd = ckpt["params"]
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                sd = ckpt["state_dict"]
            else:
                sd = ckpt
        else:
            raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

        self.net.load_state_dict(sd, strict=False)

    def inject_lora(self):
        """
        Inject LoRA modules into net. Your lora_adapters.inject_lora.inject_lora()
        REQUIRES domain_list, otherwise TypeError.
        """
        assert self.net is not None, "Call build() first."
        if not self.domains:
            raise ValueError(
                "[adapter] domains is empty. You must set adapter.domains before inject_lora()."
            )

        inject_lora(
            self.net,
            rank=self.rank,
            domain_list=self.domains,          # ✅ FIX: required by your inject_lora
            alpha=self.alpha,
            enable_patch_lora=self.enable_patch_lora,
        )

    def load_all_domain_loras(self):
        """
        Load LoRA weights for all domains into injected modules.

        NOTE:
        Your repo's lora_adapters.infer_data.load_all_domain_loras() signature differs across versions.
        Some use loradb_root, some use lora_db_path, loradb, lora_db, etc.

        We therefore call it with a compatibility shim that tries multiple keyword names,
        then falls back to positional args.
        """
        assert self.net is not None, "Call build() first."
        if not self.domains:
            raise ValueError("[adapter] domains is empty; cannot load LoRAs.")

        fn = load_all_domain_loras

        # try common kwarg variants
        trials = [
            dict(loradb_root=self.loradb_root, domains=self.domains, device=self.device),
            dict(lora_db_path=self.loradb_root, domains=self.domains, device=self.device),
            dict(loradb=self.loradb_root, domains=self.domains, device=self.device),
            dict(lora_db=self.loradb_root, domains=self.domains, device=self.device),
            dict(db_root=self.loradb_root, domains=self.domains, device=self.device),
        ]

        last_err = None
        for kw in trials:
            try:
                fn(self.net, **kw)
                return
            except TypeError as e:
                last_err = e

        # final fallback: positional call (most robust if signature is (net, db_path, domains, device))
        try:
            fn(self.net, self.loradb_root, self.domains, self.device)
            return
        except TypeError as e:
            last_err = e

        raise TypeError(
            f"[adapter] load_all_domain_loras signature mismatch. "
            f"Could not call with known kw variants or positional fallback. Last error: {last_err}"
        )

    def set_domain_weights(self, weights: Optional[Dict[str, float]]):
        """
        weights: {domain: weight}. If None -> disable all LoRA (base-only).
        """
        assert self.net is not None, "Call build() first."
        set_all_lora_domain_weights(self.net, weights)

    def set_all_domain_weights_zero(self):
        """
        Safer than None in some fusion modes: explicitly zero all known domains.
        Avoids residual weights from previous calls.
        """
        if not self.domains:
            self.set_domain_weights(None)
            return
        self.set_domain_weights({d: 0.0 for d in self.domains})

    @torch.inference_mode()
    def forward_padded(self, x: torch.Tensor, factor: int = 8) -> torch.Tensor:
        """
        Standard padded forward (reflect pad) like your old scripts.
        """
        assert self.net is not None, "Call build() first."
        _, _, h, w = x.shape
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h or pad_w:
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_in = x
        y = self.net(x_in)
        return y[:, :, :h, :w]

    @torch.inference_mode()
    def forward_base(self, x: torch.Tensor) -> torch.Tensor:
        """
        Base-only forward (disable LoRA).
        """
        self.set_domain_weights(None)
        return self.forward_padded(x)

    @torch.inference_mode()
    def forward_single(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        """
        Single-domain LoRA forward.
        """
        self.set_domain_weights({domain: 1.0})
        return self.forward_padded(x)

# lora_adapters/lora_state.py
# -*- coding: utf-8 -*-
"""
Lightweight LoRA checkpoint utilities (no CLIP/DINO deps).

- find_lora_ckpt(): locate a domain LoRA checkpoint file in a LoRA DB folder.
- load_lora_state(): load a torch checkpoint and return a plain state_dict.
- map_lora_keys_to_domain(): map LoRA keys to a given domain branch name.

This file is intentionally dependency-light so both training & inference can reuse.
"""

from __future__ import annotations

import os
import glob
from typing import Dict, Any
import torch


def find_lora_ckpt(loradb: str, domain: str) -> str:
    """
    Try best-effort to locate a LoRA checkpoint for a domain.

    Preferred patterns:
      1) {loradb}/{domain}/lora_best.pth
      2) {loradb}/{domain}/lora.pth
      3) {loradb}/{domain}.pth
      4) any *.pth under {loradb}/{domain}/  (sorted with a small priority)
    """
    # most common
    p = os.path.join(loradb, domain, "lora_best.pth")
    if os.path.isfile(p):
        return p
    p = os.path.join(loradb, domain, "lora.pth")
    if os.path.isfile(p):
        return p
    p = os.path.join(loradb, f"{domain}.pth")
    if os.path.isfile(p):
        return p

    # fallback: search folder
    ddir = os.path.join(loradb, domain)
    candidates = []
    if os.path.isdir(ddir):
        candidates = glob.glob(os.path.join(ddir, "*.pth"))

    if not candidates:
        raise FileNotFoundError(f"Cannot find LoRA ckpt for domain='{domain}' under '{loradb}'")

    # priority sort
    def _score(path: str) -> int:
        fn = os.path.basename(path).lower()
        if "lora_best" in fn:
            return 0
        if fn == "lora.pth":
            return 1
        if "best" in fn:
            return 2
        if "final" in fn:
            return 3
        return 9

    candidates = sorted(candidates, key=lambda x: (_score(x), x))
    return candidates[0]


def load_lora_state(ckpt_path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Load a torch checkpoint and return a plain state_dict.
    Supports:
      - direct state_dict
      - dict with key 'state_dict'
    """
    sd = torch.load(ckpt_path, map_location=map_location)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError(f"Unsupported ckpt format: {type(sd)} from {ckpt_path}")
    return sd


def map_lora_keys_to_domain(sd: Dict[str, Any], domain_name: str) -> Dict[str, Any]:
    """
    Map LoRA keys in a state_dict to the given domain branch name.

    Supports two common formats:
      1) ... lora_down.<domain>.weight / lora_up.<domain>.weight
      2) ... lora_down.weight / lora_up.weight  (no domain, insert domain_name)
    """
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        if "lora_" not in k or not k.endswith("weight"):
            continue
        parts = k.split(".")
        # expecting ending like [..., 'lora_down', '<maybeDomain>', 'weight']
        if len(parts) >= 3 and parts[-1] == "weight" and parts[-3].startswith("lora_"):
            # has domain already
            if parts[-2] == domain_name:
                out[k] = v
            else:
                # replace existing domain token with target domain
                parts2 = parts[:-2] + [domain_name, "weight"]
                out[".".join(parts2)] = v
        elif len(parts) >= 2 and parts[-1] == "weight" and parts[-2].startswith("lora_"):
            # no domain token: insert domain_name
            parts2 = parts[:-1] + [domain_name, "weight"]
            out[".".join(parts2)] = v
        else:
            continue
    return out

# lora_adapters/utils_merge.py
import torch

def map_to_single_domain_keys(merged_sd: dict, target_domain_name: str = '_Single') -> dict:
    """
    将无域名的 LoRA 键（"...lora_down.weight" / "...lora_up.weight"）
    映射为带 '_Single' 域名的键（"...lora_down._Single.weight"），
    以匹配当前模型中注入的 LoRAConv2d 结构。
    """
    out_sd = {}
    for k, v in merged_sd.items():
        if ".lora_down.weight" in k:
            new_k = k.replace(".lora_down.weight", f".lora_down.{target_domain_name}.weight")
        elif ".lora_up.weight" in k:
            new_k = k.replace(".lora_up.weight", f".lora_up.{target_domain_name}.weight")
        else:
            new_k = k
        out_sd[new_k] = v
    return out_sd

def apply_weighted_lora(model: torch.nn.Module, merged_sd: dict):
    """仅将 merged_sd 里 shape 匹配的 LoRA 张量加载到模型；主干不动。"""
    msd = model.state_dict()
    loadable = {k: v for k, v in merged_sd.items() if (k in msd and msd[k].shape == v.shape)}
    model.load_state_dict(loadable, strict=False)
    print(f"[apply] loaded {len(loadable)} lora tensors; skipped {len(merged_sd)-len(loadable)}")

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    print(f"[INFO] Set random seed = {seed}")
    # 1. Python 内置随机
    random.seed(seed)
    # 2. Numpy
    np.random.seed(seed)
    # 3. PyTorch CPU
    torch.manual_seed(seed)
    # 4. PyTorch GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多卡时用

    # 5. 一些环境相关的随机性（哈希）
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 6. cuDNN 的确定性设置（有些操作默认是非确定性的）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

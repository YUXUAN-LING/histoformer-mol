# lora_adapters/common/seed.py
import os
import random
import numpy as np
import torch

def set_seed(seed: int | None, deterministic: bool = False):
    """
    统一设置随机种子，保证可复现。
    deterministic=True 会让 cudnn 更确定但可能更慢。
    """
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

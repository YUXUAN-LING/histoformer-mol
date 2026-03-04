# lora_adapters/utils.py
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from PIL import Image
import yaml

def _safe_load_state_dict(model, sd):
    """仅加载与当前模型 shape 一致的权重，忽略不匹配项。"""
    msd = model.state_dict()
    filtered = {}
    drop = []
    for k, v in sd.items():
        if k in msd and msd[k].shape == v.shape:
            filtered[k] = v
        else:
            drop.append(k)
    missing = [k for k in msd.keys() if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    print(f"[load] matched={len(filtered)} | dropped={len(drop)} | missing={len(missing)}")
    if drop:
        print("  (info) dropped examples:", drop[:8], "..." if len(drop) > 8 else "")
    return filtered

def build_histoformer(weights: str|None=None, yaml_file: str|None=None):
    """
    依据 YAML 的 network_g 配置来构建 Histoformer；若提供权重，则安全加载。
    默认直接从 basicsr.models.archs.histoformer_arch 导入 Histoformer() 无参构造。
    """
    try:
        from basicsr.models.archs.histoformer_arch import Histoformer
    except Exception as e:
        raise RuntimeError(f"无法导入 Histoformer，请检查 basicsr 路径/类名。错误: {e}")

    # 可选：从 yaml 读取结构（若需要）
    if yaml_file and os.path.exists(yaml_file):
        with open(yaml_file, "r", encoding="utf-8") as f:
            opt = yaml.safe_load(f)
        if "network_g" in opt and isinstance(opt["network_g"], dict):
            kwargs = {k: v for k, v in opt["network_g"].items() if k != "type"}
            model = Histoformer(**kwargs)
        else:
            model = Histoformer()
    else:
        model = Histoformer()

    if weights:
        sd = torch.load(weights, map_location="cpu")
        if isinstance(sd, dict) and "params" in sd and isinstance(sd["params"], dict):
            sd = sd["params"]
        if not isinstance(sd, dict):
            raise RuntimeError("权重文件格式异常，不是 state_dict/dict。")
        _safe_load_state_dict(model, sd)
    return model

def load_image(path: str):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)  # 1x3xHxW
    return t, img.size

def save_image(t: torch.Tensor, path: str):
    t = t.clamp(0,1).detach().cpu().squeeze(0).permute(1,2,0).numpy()
    arr = (t*255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)

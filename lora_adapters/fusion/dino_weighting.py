# lora_adapters/fusion/dino_weighting.py
# -*- coding: utf-8 -*-
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DINOv2ViTB14(nn.Module):
    """
    DINOv2 ViT-B/14 特征编码器（基于 timm：'vit_base_patch14_dinov2.lvd142m'）
    输出：L2-normalized 的 CLS embedding，shape [B, C]
    """
    def __init__(self, freeze: bool = True,
                 ckpt_path: str = "weights/dinov2/vit_base_patch14_dinov2.lvd142m-384.pth"):
        super().__init__()
        import timm, torch
        # 不联网初始化
        self.backbone = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False)

        # 从本地加载权重
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"DINOv2 本地权重不存在: {ckpt_path}\n"
                f"请确认 ckpt 放在该路径，或者修改 DINOv2ViTB14.__init__ 的 ckpt_path。"
            )
        state = torch.load(ckpt_path, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        self.backbone.load_state_dict(state, strict=False)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W] 0..1
        返回：L2-normalized CLS token embedding [B,C]
        """
        feats = self.backbone.forward_features(x)
        if isinstance(feats, dict) and "x_norm_clstoken" in feats:
            v = feats["x_norm_clstoken"]
        elif isinstance(feats, torch.Tensor) and feats.dim() == 3:
            v = feats[:, 0]
        else:
            v = feats.mean(dim=1)
        return torch.nn.functional.normalize(v, dim=-1)


class PrototypeBank(nn.Module):
    """
    保存各域 prototype，并提供相似度->权重（softmax(τ·cos)) 的计算
    """
    def __init__(self, proto_dict: Dict[str, torch.Tensor], device='cpu'):
        super().__init__()
        self.domains: List[str] = list(proto_dict.keys())
        mat = torch.stack([F.normalize(p.float(), dim=-1) for p in proto_dict.values()], dim=0)  # [K,C]
        self.register_buffer("P", mat.to(device), persistent=False)

    @torch.no_grad()
    def weights(self, f: torch.Tensor, tau: float = 7.5) -> torch.Tensor:
        """
        f: [B,C] 已归一化
        返回：w: [B,K]
        """
        sim = f @ self.P.t()           # [B,K]
        return F.softmax(tau * sim, dim=-1)

def load_prototypes(path: str, device='cpu') -> Dict[str, torch.Tensor]:
    sd = torch.load(path, map_location=device)
    if not isinstance(sd, dict):
        raise RuntimeError(f"Invalid prototype file: {path}")
    return {k: v.to(device) for k, v in sd.items()}

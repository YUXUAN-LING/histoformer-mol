# lora_adapters/fusion/runtime.py
# -*- coding: utf-8 -*-
from typing import Dict, List, Optional
import torch
import torchvision.transforms as T
from PIL import Image

from .dino_weighting import DINOv2ViTB14, PrototypeBank

# ---- 工具：把权重写入所有 LoRA 模块 ----
def set_domain_weights(model: torch.nn.Module, w: torch.Tensor):
    """
    将权重向量 w:[B,K] 或 [K] 写入模型内所有 LoRA 模块 (module.domain_weights)
    推理一般是 [K]，训练时可能是 [B,K]；LoRA Linear 会自行广播。
    """
    for m in model.modules():
        if hasattr(m, "domain_weights"):
            m.domain_weights = w

class WeightComputer:
    """
    DINOv2 + PrototypeBank -> 计算权重向量 w(x)
    """
    def __init__(self, proto_state: Dict[str, torch.Tensor], tau: float = 7.5, device='cuda'):
        self.encoder = DINOv2ViTB14().to(device)
        self.bank = PrototypeBank(proto_state, device=device).to(device)
        self.tau = tau
        self.device = device
        # DINOv2 推理默认预处理
        self.tfm = T.Compose([
            T.Resize(518, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(518),
            T.Resize(518),
            T.CenterCrop(518),
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])

    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor | Image.Image | List[Image.Image]) -> torch.Tensor:
        """
        imgs: [B,3,H,W] 或 PIL 或 PIL 列表
        返回 w: [B,K]
        """
        if isinstance(imgs, torch.Tensor):
            x = imgs.to(self.device)
        else:
            if isinstance(imgs, Image.Image):
                imgs = [imgs]
            batch = [self.tfm(im.convert("RGB")) for im in imgs]
            x = torch.stack(batch, 0).to(self.device)
        f = self.encoder(x)                # [B,C]
        return self.bank.weights(f, self.tau)

class LoRARuntime:
    """
    运行时：给定 (model, prototypes)，对输入图像计算 w 并执行前向
    """
    def __init__(self, model: torch.nn.Module, proto_state: Dict[str, torch.Tensor],
                 tau: float = 7.5, device='cuda'):
        self.model = model.to(device).eval()
        self.device = device
        self.computer = WeightComputer(proto_state, tau=tau, device=device)

    @torch.no_grad()
    def infer_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W] 0..1
        """
        w = self.computer(x)  # [B,K]
        set_domain_weights(self.model, w)
        return self.model(x)

    @torch.no_grad()
    def infer_images(self, imgs: List[Image.Image] | Image.Image) -> torch.Tensor:
        w = self.computer(imgs)  # [B,K]
        set_domain_weights(self.model, w)
        if isinstance(imgs, Image.Image):
            imgs = [imgs]
        to_tensor = T.ToTensor()
        x = torch.stack([to_tensor(im.convert("RGB")) for im in imgs], 0).to(self.device)
        return self.model(x)

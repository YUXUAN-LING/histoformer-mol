import torch
import torch.nn as nn
from typing import Dict, List, Optional, Iterable, Union


class LoRALinear(nn.Module):
    """
    多域 LoRA 线性层封装。
    - self.base: 原始 Linear 层
    - 对每个域 d, 拥有一组 (lora_down[d], lora_up[d])
    - 训练单域时通常 domain_list = [train_domain]
    - 推理多域时 domain_list = ["rain","snow","fog",...]，并由外部设置 domain_weights 做加权融合
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        domain_list: List[str],
        alpha: float = 1.0,
        enable_bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = r
        self.alpha = alpha
        self.domain_list = list(domain_list)

        # 原始线性层（主干权重）
        self.base = nn.Linear(in_features, out_features, bias=enable_bias)

        # 对每个域建立一套 LoRA 分支
        self.lora_down = nn.ModuleDict()
        self.lora_up = nn.ModuleDict()
        if self.rank > 0:
            for d in self.domain_list:
                down = nn.Linear(in_features, r, bias=False)
                up = nn.Linear(r, out_features, bias=False)
                # 经典 LoRA 初始化：上层置零，下层小随机
                nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
                nn.init.zeros_(up.weight)
                self.lora_down[d] = down
                self.lora_up[d] = up

        # 推理时的域权重；可以是:
        # - None: 训练单域时通常使用，若 domain_list 只有一个域，则该域 LoRA 总是开启
        # - dict[name->float], 或 Tensor/list: 与 domain_list 对齐
        self.domain_weights: Optional[Union[Dict[str, float], torch.Tensor, List[float]]] = None

    def set_domain_weights(self, weights: Optional[Union[Dict[str, float], torch.Tensor, List[float]]]):
        self.domain_weights = weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.rank <= 0 or len(self.lora_down) == 0:
            return out

        # 训练单域的常见情况：domain_list 只有 1 个域，且没有显式设置 domain_weights
        if self.domain_weights is None:
            if len(self.domain_list) == 1:
                d = self.domain_list[0]
                return out + (self.alpha / self.rank) * self.lora_up[d](self.lora_down[d](x))
            # 多域但未提供权重 -> 默认不加 LoRA，避免意外行为
            return out

        # 推理多域：根据 domain_weights 加权
        if isinstance(self.domain_weights, dict):
            for d, w in self.domain_weights.items():
                if d in self.lora_down and w != 0:
                    out = out + (self.alpha / self.rank) * float(w) * self.lora_up[d](self.lora_down[d](x))
        else:
            # 顺序与 domain_list 对齐
            if torch.is_tensor(self.domain_weights):
                weights_list = self.domain_weights.tolist()
            else:
                weights_list = list(self.domain_weights)
            for i, d in enumerate(self.domain_list):
                w = weights_list[i]
                if w != 0:
                    out = out + (self.alpha / self.rank) * float(w) * self.lora_up[d](self.lora_down[d](x))

        return out


import math


class LoRAConv2d(nn.Module):
    """
    多域 LoRA 卷积层封装（Conv 版 LoRA）。
    思路：
      y = Conv_base(x) + Σ_i w_i * LoRA_i(x) * (alpha / r)

    其中每个 LoRA_i(x) = Conv_up_i( Conv_down_i(x) )
    - Conv_down: 1x1 conv, 降维到 rank
    - Conv_up:   kxk conv, 升维回 out_channels（共享 stride/padding/dilation/groups）

    参数:
        base_conv: 原始 nn.Conv2d 模块
        r:         LoRA 低秩 rank
        domain_list: 域名列表
        alpha:     缩放系数
    """

    def __init__(
        self,
        base_conv: nn.Conv2d,
        r: int,
        domain_list: List[str],
        alpha: float = 1.0,
    ):
        super().__init__()
        assert isinstance(base_conv, nn.Conv2d)
        self.rank = r
        self.alpha = alpha
        self.domain_list = list(domain_list)

        # 保存并注册原始 Conv 作为子模块
        self.base = base_conv

        in_ch = base_conv.in_channels
        out_ch = base_conv.out_channels
        k_h, k_w = base_conv.kernel_size
        stride = base_conv.stride
        padding = base_conv.padding
        dilation = base_conv.dilation
        groups = base_conv.groups

        self.lora_down = nn.ModuleDict()
        self.lora_up = nn.ModuleDict()

        if self.rank > 0:
            for d in self.domain_list:
                # down: 1x1 conv, 不改变 H,W
                down = nn.Conv2d(in_ch, r, kernel_size=1, stride=1, padding=0, bias=False)
                # up: 与原始卷积同 k/stride/pad/dilation/groups
                up = nn.Conv2d(
                    r,
                    out_ch,
                    kernel_size=(k_h, k_w),
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=False,
                )
                nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
                nn.init.zeros_(up.weight)
                self.lora_down[d] = down
                self.lora_up[d] = up

        # 推理时的域权重，含义同 LoRALinear
        self.domain_weights: Optional[Union[Dict[str, float], torch.Tensor, List[float]]] = None

    def set_domain_weights(self, weights: Optional[Union[Dict[str, float], torch.Tensor, List[float]]]):
        self.domain_weights = weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.rank <= 0 or len(self.lora_down) == 0:
            return out

        # 训练单域：常见情况 domain_list 只有一个域，未设置权重
        if self.domain_weights is None:
            if len(self.domain_list) == 1:
                d = self.domain_list[0]
                return out + (self.alpha / self.rank) * self.lora_up[d](self.lora_down[d](x))
            return out

        # 推理多域：加权融合
        if isinstance(self.domain_weights, dict):
            for d, w in self.domain_weights.items():
                if d in self.lora_down and w != 0:
                    out = out + (self.alpha / self.rank) * float(w) * self.lora_up[d](self.lora_down[d](x))
        else:
            if torch.is_tensor(self.domain_weights):
                weights_list = self.domain_weights.tolist()
            else:
                weights_list = list(self.domain_weights)
            for i, d in enumerate(self.domain_list):
                w = weights_list[i]
                if w != 0:
                    out = out + (self.alpha / self.rank) * float(w) * self.lora_up[d](self.lora_down[d](x))

        return out

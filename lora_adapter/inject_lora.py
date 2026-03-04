# lora_adapters/inject_lora.py
# -*- coding: utf-8 -*-
"""
结构化 LoRA 注入器（Histoformer 专用）

- 不再依赖正则匹配名字
- 直接根据模块类型 + 属性名注入 LoRAConv2d
- 目前支持：
    * OverlapPatchEmbed.proj         （可选，通过 enable_patch_lora 控制）
    * Attention_histogram.qkv
    * Attention_histogram.project_out

并提供：
    - inject_lora_structural(...)
    - inject_lora(...)              （兼容旧代码的别名）
    - iter_lora_modules(model)
    - lora_parameters(model, train_domain=None)

"""

from typing import List, Optional, Dict, Union
from basicsr.models.archs.histoformer_arch import FeedForward
import torch
import torch.nn as nn

from basicsr.models.archs.histoformer_arch import (
    OverlapPatchEmbed,
    Attention_histogram,
)
from .lora_linear import LoRALinear, LoRAConv2d


def inject_lora_structural(
    model: nn.Module,
    rank: int,
    domain_list: List[str],
    alpha: float = 1.0,
    enable_patch_lora: bool = False,
) -> nn.Module:
    """
    结构化地给 Histoformer 注入 LoRAConv2d：

    - 对所有 OverlapPatchEmbed 实例：
        若 enable_patch_lora=True，则将其 .proj (Conv2d) 替换为 LoRAConv2d

    - 对所有 Attention_histogram 实例：
        始终将 .qkv 和 .project_out (Conv2d) 替换为 LoRAConv2d

    说明：
        - base conv 权重被冻结（requires_grad=False）
        - 只训练 LoRA 分支 (down/up)
    """

    # 1) 处理 patch embedding（OverlapPatchEmbed）
    if enable_patch_lora:
        for m in model.modules():
            if isinstance(m, OverlapPatchEmbed):
                conv = m.proj
                if isinstance(conv, LoRAConv2d):
                    # 已经注入过，跳过
                    continue
                lora_conv = LoRAConv2d(
                    base_conv=conv,
                    r=rank,
                    domain_list=domain_list,
                    alpha=alpha,
                )
                # 冻结主干 conv 权重
                for p in lora_conv.base.parameters():
                    p.requires_grad = False
                m.proj = lora_conv

    # 2) 处理 Attention_histogram 内部的 qkv & project_out
    for m in model.modules():
        if isinstance(m, Attention_histogram):
            # qkv
            if isinstance(m.qkv, LoRAConv2d):
                # 已经是 LoRAConv2d 了就不重复包
                pass
            else:
                lora_qkv = LoRAConv2d(
                    base_conv=m.qkv,
                    r=rank,
                    domain_list=domain_list,
                    alpha=alpha,
                )
                for p in lora_qkv.base.parameters():
                    p.requires_grad = False
                m.qkv = lora_qkv

            # project_out
            if isinstance(m.project_out, LoRAConv2d):
                pass
            else:
                lora_po = LoRAConv2d(
                    base_conv=m.project_out,
                    r=rank,
                    domain_list=domain_list,
                    alpha=alpha,
                )
                for p in lora_po.base.parameters():
                    p.requires_grad = False
                m.project_out = lora_po

    # for m in model.modules():
    #     if isinstance(m, FeedForward):
    #         for name in ["project_in", "project_out"]:
    #             conv = getattr(m, name)
    #             if not isinstance(conv, LoRAConv2d):
    #                 lora_conv = LoRAConv2d(
    #                     base_conv=conv,
    #                     r=rank,
    #                     domain_list=domain_list,
    #                     alpha=alpha,
    #                 )
    #                 for p in lora_conv.base.parameters():
    #                     p.requires_grad = False
    #                 setattr(m, name, lora_conv)

    return model


def inject_lora(
    model: nn.Module,
    rank: int,
    domain_list: List[str],
    alpha: float = 1.0,
    enable_patch_lora: bool = False,
    **kwargs,
) -> nn.Module:
    """
    向后兼容的注入入口：

    之前的调用可能是：
        inject_lora(model, rank, domain_list, alpha, target_names=..., patterns=...)

    现在：
        - 我们完全忽略 target_names / patterns 等参数
        - 始终走结构化注入逻辑
    """
    return inject_lora_structural(
        model=model,
        rank=rank,
        domain_list=domain_list,
        alpha=alpha,
        enable_patch_lora=enable_patch_lora,
    )


def iter_lora_modules(model: nn.Module):
    """
    遍历模型中所有 LoRA 模块（包括 Linear 和 Conv 版本）。
    """
    for m in model.modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            yield m


def lora_parameters(model: nn.Module, train_domain: Optional[str] = None):
    """
    返回所有 LoRA 参数（down/up）的参数列表。

    参数:
        train_domain:
            - None: 返回所有域的 LoRA 参数；
            - "rain" / "snow" / ...: 只返回指定域的 LoRA 参数。
    """
    params = []
    for m in iter_lora_modules(model):
        domains = m.domain_list
        if train_domain is None:
            # 所有域
            for d in domains:
                params += list(m.lora_down[d].parameters())
                params += list(m.lora_up[d].parameters())
        else:
            # 只训练指定域
            if train_domain in domains:
                params += list(m.lora_down[train_domain].parameters())
                params += list(m.lora_up[train_domain].parameters())
    return params

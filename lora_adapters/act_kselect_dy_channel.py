# lora_adapters/act_kselect_dy_channel.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import math
import torch


def _rms(x: torch.Tensor, dim=None, eps: float = 1e-12):
    x = x.float()
    if dim is None:
        return torch.sqrt(torch.mean(x * x) + eps)
    return torch.sqrt(torch.mean(x * x, dim=dim) + eps)


def _channel_dim_from_tensor(y: torch.Tensor) -> int:
    # Histoformer 里 LoRA 主要包在 Conv2d 上：B,C,H,W → channel dim=1
    # 若是 Linear/Token 形式：B,T,C → channel dim=-1
    if y.ndim == 4:
        return 1
    return -1


def _apply_gate(dy: torch.Tensor, g: torch.Tensor, ch_dim: int) -> torch.Tensor:
    # g: [Cout]
    shape = [1] * dy.ndim
    shape[ch_dim] = -1
    return dy * g.view(*shape)


@dataclass
class ActDyChConfig:
    dom1: str
    dom2: str
    layer_topk: int = 30       # 选 topK 层（模块）
    layer_tau: float = 0.05    # 层阈值：score >= tau 才启用
    ch_topk: int = 32          # 每层保留 topK 通道
    enable_both: bool = True
    both_tau: float = 0.05
    both_ratio: float = 0.6
    eps: float = 1e-12
    print_top_modules: int = 30   # 打印 topN 层
    verbose: bool = True
    recompute_gate_on_apply: bool = False  # True: apply 时重新按当前输入计算 gate；False: 用 probe 的 gate


class ActKSelectDyChannelController:
    """
    两路 LoRA（dom1/dom2），在 LoRA module 级做 topK 层稀疏；
    在被选中的层内部做 channel gate（topK channel + both/hard pick）。

    通过 forward hook 实现，不需要改 LoRA 模块实现。
    """

    def __init__(self, net: torch.nn.Module, cfg: ActDyChConfig):
        self.net = net
        self.cfg = cfg
        self.mode = "probe"  # "probe" or "apply"
        self.hooks = []
        import torch.nn as nn
        # 所有“看起来像 LoRA module”的层：需要有 lora_up/lora_down dict
        self.modules: List[Tuple[str, torch.nn.Module]] = []
        for name, m in net.named_modules():
            if hasattr(m, "lora_up") and hasattr(m, "lora_down"):
                lu = getattr(m, "lora_up", None)
                ld = getattr(m, "lora_down", None)
                if isinstance(lu, (dict, nn.ModuleDict)) and isinstance(ld, (dict, nn.ModuleDict)):
                    self.modules.append((name, m))

        # probe 记录：每层 score / dy / gate 等
        self.records: Dict[str, Dict[str, Any]] = {}
        # apply 使用：哪些层启用、以及 gate
        self.selected: Dict[str, bool] = {}
        self.gates: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}  # name -> (g1,g2) on CPU
        self.dw_missing = 0
        self.hit = 0

        # 统计
        self.x_rms_list: List[float] = []
        self.score_list: List[float] = []

        self._register_hooks()

    def _register_hooks(self):
        for name, m in self.modules:
            h = m.register_forward_hook(self._make_hook(name, m))
            self.hooks.append(h)

    def remove(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks = []

    def set_mode(self, mode: str):
        assert mode in ["probe", "apply"]
        self.mode = mode

    def _get_scale(self, m: torch.nn.Module) -> float:
        # 兼容你现有 LoRA 实现：一般有 alpha/rank
        alpha = float(getattr(m, "alpha", 1.0))
        rank = float(getattr(m, "rank", 1.0))
        return alpha / max(rank, 1.0)

    def _compute_dy(self, m: torch.nn.Module, x: torch.Tensor, dom: str) -> Optional[torch.Tensor]:
        # dy = (alpha/rank) * up(down(x))
        if dom not in m.lora_up or dom not in m.lora_down:
            return None
        scale = self._get_scale(m)
        return scale * m.lora_up[dom](m.lora_down[dom](x))

    def _make_hook(self, name: str, m: torch.nn.Module):
        cfg = self.cfg

        def hook(_m, inputs, output):
            # 仅处理 tensor 输出
            if not torch.is_tensor(output):
                return output
            if len(inputs) == 0 or not torch.is_tensor(inputs[0]):
                return output

            x = inputs[0]
            self.hit += 1

            # 记录 x rms（用于 hook summary）
            xr = float(_rms(x).detach().cpu().item())
            self.x_rms_list.append(xr)

            # 计算两路 dy
            dy1 = self._compute_dy(_m, x, cfg.dom1)
            dy2 = self._compute_dy(_m, x, cfg.dom2)

            if dy1 is None or dy2 is None:
                # 这层缺少某个 domain 的 LoRA 权重
                self.dw_missing += 1
                if self.mode == "probe":
                    self.records[name] = dict(
                        name=name, score=0.0, rms_x=xr,
                        dy1=0.0, dy2=0.0,
                        pick=0, mode="missing",
                        nz1=0, nz2=0, both_ch=0, keep=0, K=cfg.ch_topk,
                        ch_dim=_channel_dim_from_tensor(output),
                    )
                return output

            # 按通道 RMS 得到 d1[i], d2[i]
            ch_dim = _channel_dim_from_tensor(dy1)
            dims = [i for i in range(dy1.ndim) if i != ch_dim]
            d1 = _rms(dy1, dim=dims, eps=cfg.eps)  # [C]
            d2 = _rms(dy2, dim=dims, eps=cfg.eps)  # [C]
            s = torch.maximum(d1, d2)              # [C]

            C = s.numel()
            K = min(cfg.ch_topk, C)
            # topK channels
            top_idx = torch.topk(s, k=K, largest=True).indices  # [K]

            keep_mask = torch.zeros_like(s, dtype=torch.bool)
            keep_mask[top_idx] = True

            # BOTH 条件
            mx = torch.maximum(d1, d2)
            mn = torch.minimum(d1, d2)
            ratio = mn / (mx + cfg.eps)

            both_mask = torch.zeros_like(s, dtype=torch.bool)
            if cfg.enable_both:
                both_mask = keep_mask & (s >= cfg.both_tau) & (ratio >= cfg.both_ratio)

            # hard pick
            pick1_mask = keep_mask & (~both_mask) & (d1 >= d2)
            pick2_mask = keep_mask & (~both_mask) & (d2 > d1)

            g1 = torch.zeros_like(s)
            g2 = torch.zeros_like(s)

            # hard pick: 1 or 2
            g1[pick1_mask] = 1.0
            g2[pick2_mask] = 1.0

            # both: proportional weights
            if both_mask.any():
                denom = (d1 + d2 + cfg.eps)
                g1[both_mask] = (d1 / denom)[both_mask]
                g2[both_mask] = (d2 / denom)[both_mask]

            nz1 = int((g1 > 0).sum().detach().cpu().item())
            nz2 = int((g2 > 0).sum().detach().cpu().item())
            both_ch = int(((g1 > 0) & (g2 > 0)).sum().detach().cpu().item())
            keep = int(keep_mask.sum().detach().cpu().item())

            # 层级 score：用 topK channels 的最大 s
            score = float(s[top_idx].max().detach().cpu().item()) if K > 0 else 0.0
            dy1_r = float(_rms(dy1).detach().cpu().item())
            dy2_r = float(_rms(dy2).detach().cpu().item())
            self.score_list.append(score)

            # module-level pick 用于汇总：0/1/2/3(both)
            if keep == 0:
                pick = 0
                mode_str = "none"
            else:
                if nz1 > 0 and nz2 == 0:
                    pick = 1
                elif nz2 > 0 and nz1 == 0:
                    pick = 2
                else:
                    pick = 3
                mode_str = f"gate(nz1={nz1},nz2={nz2},both_ch={both_ch},K={K})"

            if self.mode == "probe":
                # 存 gate 到 CPU，apply 时直接用（保证日志一致）
                self.gates[name] = (g1.detach().cpu(), g2.detach().cpu())
                self.records[name] = dict(
                    name=name, score=score, rms_x=xr,
                    dy1=dy1_r, dy2=dy2_r,
                    pick=pick, mode=mode_str,
                    nz1=nz1, nz2=nz2, both_ch=both_ch, keep=keep, K=K,
                    ch_dim=ch_dim,
                )
                return output

            # apply
            if not self.selected.get(name, False):
                return output

            # gate 选择：重算 or 用 probe gate
            if cfg.recompute_gate_on_apply:
                g1_use, g2_use = g1, g2
            else:
                g1_cpu, g2_cpu = self.gates.get(name, (None, None))
                if g1_cpu is None or g2_cpu is None:
                    return output
                g1_use = g1_cpu.to(device=output.device, dtype=output.dtype)
                g2_use = g2_cpu.to(device=output.device, dtype=output.dtype)

            dy1_add = _apply_gate(dy1.to(output.dtype), g1_use, ch_dim)
            dy2_add = _apply_gate(dy2.to(output.dtype), g2_use, ch_dim)
            return output + dy1_add + dy2_add

        return hook

    def finalize_selection(self):
        """
        probe 完成后：按 score 做 topK layer + tau 过滤 → selected
        """
        cfg = self.cfg
        items = list(self.records.items())
        items.sort(key=lambda kv: float(kv[1].get("score", 0.0)), reverse=True)

        topk = min(cfg.layer_topk, len(items))
        top_items = items[:topk]

        selected_names = []
        for n, rec in top_items:
            if float(rec.get("score", 0.0)) >= cfg.layer_tau:
                selected_names.append(n)

        # set selected dict
        self.selected = {n: False for n, _ in self.modules}
        for n in selected_names:
            self.selected[n] = True

        return selected_names, items

    def print_summary(self, items_sorted: List[Tuple[str, Dict[str, Any]]]):
        cfg = self.cfg
        total = len(self.modules)
        topk = min(cfg.layer_topk, total)

        # selected count
        selected = [n for n, _ in self.modules if self.selected.get(n, False)]
        selected_n = len(selected)

        # pick count over selected layers
        pick0 = pick1 = pick2 = both = 0
        for n in selected:
            rec = self.records.get(n, {})
            p = int(rec.get("pick", 0))
            if p == 0:
                pick0 += 1
            elif p == 1:
                pick1 += 1
            elif p == 2:
                pick2 += 1
            else:
                both += 1

        # x rms stats
        if self.x_rms_list:
            xmn = min(self.x_rms_list); xmx = max(self.x_rms_list)
            xmean = sum(self.x_rms_list) / len(self.x_rms_list)
        else:
            xmn = xmx = xmean = 0.0

        # score stats
        if self.score_list:
            smn = min(self.score_list); smx = max(self.score_list)
            smean = sum(self.score_list) / len(self.score_list)
        else:
            smn = smx = smean = 0.0

        print(
            f"[act_kselect_dy_ch] total={total} topk={topk} tau={cfg.layer_tau} selected={selected_n} "
            f"pick0={pick0} pick1={pick1} pick2={pick2} both={both} dw_missing={self.dw_missing} "
            f"(ch_topk={cfg.ch_topk} both_tau={cfg.both_tau} both_ratio={cfg.both_ratio} enable_both={int(cfg.enable_both)})"
        )
        print(
            f"[act_kselect_dy_ch][hook] hit={self.hit}/{total} "
            f"x_rms(min/max/mean)={xmn:.6g}/{xmx:.6g}/{xmean:.6g} "
            f"score(min/max/mean)={smn:.6g}/{smx:.6g}/{smean:.6g}"
        )

        # print scored modules
        print("[act_kselect_dy_ch] scored modules (sorted):")
        show = items_sorted[: cfg.print_top_modules]
        for i, (n, rec) in enumerate(show):
            print(
                f" {i:02d} {n} score={rec.get('score',0):.6g} rms_x={rec.get('rms_x',0):.6g} "
                f"dy1={rec.get('dy1',0):.6g} dy2={rec.get('dy2',0):.6g} "
                f"pick={rec.get('pick',0)} mode={rec.get('mode','')}"
            )
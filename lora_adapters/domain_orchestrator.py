# lora_adapters/retrieval/domain_orchestrator.py
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import glob, numpy as np, torch


@dataclass
class Domain:
    name: str
    lora_path: Path
    prototypes: np.ndarray  # ⭐ 修改：原来是 avg_embedding（1D），现在统一存 [K, C] 原型矩阵


class DomainOrchestrator:
    def __init__(
        self,
        domains: List[str],
        lora_db_path: str | Path = 'loradb',
        sim_metric: str = 'cosine',      # 相似度度量方式
        temperature: float = 0.07,       # 默认温度
        embedder_tag: str | None = None  # 用于选择 avg_embedding_<tag>.npy / prototypes_<tag>.npy
    ):
        """
        embedder_tag 例子：
          - 'dinov2_vitb14'
          - 'clip_vit-b-16'
          - 'fft_amp'
        如果为 None，则只会尝试读取 avg_embedding.npy（兼容旧逻辑）
        """
        self.lora_db_path = Path(lora_db_path)
        self.domains: Dict[str, Domain] = {}
        self.sim_metric = sim_metric.lower()
        self.temperature = float(temperature)
        self.embedder_tag = embedder_tag

        self._load(domains)

    def _load(self, domains: List[str]):
        for d in domains:
            dom_dir = self.lora_db_path / d

            emb_path: Path | None = None

            # ⭐ 新增优先级 1：多原型文件 prototypes_<embedder_tag>.npy
            if self.embedder_tag:
                tagged_proto = dom_dir / f"prototypes_{self.embedder_tag}.npy"
                if tagged_proto.exists():
                    emb_path = tagged_proto

            # 优先级 2：avg_embedding_<embedder_tag>.npy（兼容之前只建了均值原型的情况）
            if emb_path is None and self.embedder_tag:
                tagged_avg = dom_dir / f"avg_embedding_{self.embedder_tag}.npy"
                if tagged_avg.exists():
                    emb_path = tagged_avg

            # 优先级 3：老版本 avg_embedding.npy
            if emb_path is None:
                legacy = dom_dir / "avg_embedding.npy"
                if legacy.exists():
                    emb_path = legacy

            if emb_path is None:
                raise FileNotFoundError(
                    f"[DomainOrchestrator] 没有找到 {d} 的 prototype："
                    f"既没有 prototypes_{self.embedder_tag}.npy，"
                    f"也没有 avg_embedding_{self.embedder_tag}.npy，"
                    f"也没有 avg_embedding.npy，"
                    f"请先用相同 embedder 跑 build_prototypes.py"
                )

            arr = np.load(emb_path).astype(np.float32)
            # 统一为 [K, C]，K=1 时就是单原型
            if arr.ndim == 1:
                arr = arr[None, :]
            print(f"[load] 域 {d} 使用 prototype: {emb_path} | shape={arr.shape}")

            # 找 LoRA ckpt
            cand = dom_dir / f"{d}.pth"
            if not cand.exists():
                pths = glob.glob(str(dom_dir / "*.pth"))
                if not pths:
                    raise FileNotFoundError(f"No LoRA ckpt under {dom_dir}")
                cand = Path(pths[0])

            self.domains[d] = Domain(d, cand, arr)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray, eps=1e-6):
        a = a / (np.linalg.norm(a) + eps)
        b = b / (np.linalg.norm(b) + eps)
        return float(np.dot(a, b))

    @staticmethod
    def l2(a: np.ndarray, b: np.ndarray) -> float:
        """欧式距离 ||a-b||_2"""
        diff = a - b
        return float(np.sqrt(np.sum(diff * diff)))

    def select_topk(
        self,
        img_emb: np.ndarray,
        top_k: int = 3,
        temperature: float | None = None
    ) -> List[Tuple[str, float]]:
        """
        多原型检索 + 域级聚合版 select_topk：

        记每个域 d 有原型集合 C_d = {c_{d,1},...,c_{d,K_d}}。

        - cosine:
            1) 对所有原型 c_{d,j} 计算 s_{d,j} = cos(img_emb, c_{d,j})
            2) 在 (d,j) 维度上全局选原型级 top_k（相似度越大越好）
            3) 对每个域 d，将属于该域的原型相似度求和 S_d = Σ s_{d,j}
            4) 对 {S_d} / temperature 做 softmax 得到域权重 w_d

        - euclidean / l2:
            1) 对所有原型计算 L2 距离 d_{d,j}
            2) 在 (d,j) 维度上按距离从小到大选原型级 top_k
            3) 令 proximity p_{d,j} = 1 / d_{d,j}，对同一域求和 P_d = Σ p_{d,j}
            4) 对 {P_d} / temperature 做 softmax 得到域权重 w_d

        注意：
          - 如果每个域只有 1 个原型（K_d=1），则行为退化为“域级 top_k”，
            与旧版本在数值上等价（只是实现方式不同）。
        """
        if temperature is None:
            temperature = self.temperature
        temperature = max(float(temperature), 1e-6)

        metric = self.sim_metric

        # ===== 余弦相似度：越大越好，top_k 原型，域级聚合 =====
        if metric == 'cosine':
            entries: List[Tuple[str, int, float]] = []  # (domain, proto_idx, sim)

            for d, dom in self.domains.items():
                protos = dom.prototypes  # [K_d, C]
                for j in range(protos.shape[0]):
                    s = self.cosine(img_emb, protos[j])
                    entries.append((d, j, s))

            if not entries:
                return []

            # 按相似度从大到小排序，取原型级 top_k
            entries.sort(key=lambda x: x[2], reverse=True)
            top_entries = entries[:max(1, top_k)]

            # 对同一域聚合相似度
            domain_scores: Dict[str, float] = {}
            for d, j, s in top_entries:
                domain_scores[d] = domain_scores.get(d, 0.0) + float(s)

            doms = list(domain_scores.keys())
            scores = np.array([domain_scores[d] for d in doms], dtype=np.float32)

            print("[debug raw_scores]", list(zip(doms, scores)))  # ★ 额外看一下聚合前的原始分数

            # softmax 激活
            logits = scores / temperature
            logits = logits - logits.max()
            w = np.exp(logits)
            w_sum = w.sum()
            if w_sum <= 1e-8:
                w = np.ones_like(w) / len(w)
            else:
                w /= w_sum

            return [(name, float(weight)) for name, weight in zip(doms, w)]

        # ===== 欧式距离：越小越好，top_k 原型，域级聚合 =====
        elif metric in ('euclidean', 'l2'):
            entries: List[Tuple[str, int, float]] = []  # (domain, proto_idx, dist)

            for d, dom in self.domains.items():
                protos = dom.prototypes  # [K_d, C]
                for j in range(protos.shape[0]):
                    dist = self.l2(img_emb, protos[j])
                    entries.append((d, j, dist))

            if not entries:
                return []

            # 按距离从小到大排序，取原型级 top_k
            entries.sort(key=lambda x: x[2])
            top_entries = entries[:max(1, top_k)]

            # 用 1/d 当 proximity，再在域级聚合
            domain_scores: Dict[str, float] = {}
            for d, j, dist in top_entries:
                prox = 1.0 / max(float(dist), 1e-6)
                domain_scores[d] = domain_scores.get(d, 0.0) + prox

            doms = list(domain_scores.keys())
            scores = np.array([domain_scores[d] for d in doms], dtype=np.float32)

            print("[debug raw_scores]", list(zip(doms, scores)))  # ★ 额外看一下聚合前的原始分数
            
            logits = scores / temperature
            logits = logits - logits.max()
            w = np.exp(logits)
            w_sum = w.sum()
            if w_sum <= 1e-8:
                w = np.ones_like(w) / len(w)
            else:
                w /= w_sum

            return [(name, float(weight)) for name, weight in zip(doms, w)]

        else:
            raise ValueError(f"Unknown sim_metric: {metric}")

    def _normalize_keys(self, sd: dict) -> dict:
        """将可能带域名的键（...lora_down.<Domain>.weight）规范为域无关键（...lora_down.weight）"""
        out = {}
        for k, v in sd.items():
            if "lora_" not in k:
                continue
            parts = k.split(".")
            # 期望 [..., 'lora_down', '<maybeDomain>', 'weight']
            if (
                len(parts) >= 2
                and parts[-1] == 'weight'
                and parts[-3].startswith('lora_')
                and parts[-2] not in ('weight', 'bias')
            ):
                # drop domain token
                k2 = ".".join(parts[:-2] + parts[-1:])
                out[k2] = v
            elif parts[-1] == 'weight':
                out[k] = v
        return out

    def build_weighted_lora(self, selections: List[Tuple[str, float]]) -> Dict[str, torch.Tensor]:
        """
        根据选出的 (domain_name, weight) 列表，加载各自 LoRA，按权重加权融合。
        注意：这里仍然是域级加权，与“多原型检索”解耦。
        """
        merged: Dict[str, torch.Tensor] = {}
        for name, w in selections:
            sd = torch.load(self.domains[name].lora_path, map_location='cpu')
            if 'state_dict' in sd:
                sd = sd['state_dict']
            sd = self._normalize_keys(sd)
            for k, v in sd.items():
                if k not in merged:
                    merged[k] = v.float() * w
                else:
                    merged[k] += v.float() * w
        return merged

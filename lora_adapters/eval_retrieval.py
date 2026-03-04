# lora_adapters/eval_retrieval.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, csv, json, time, random, hashlib, argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image
import torch


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _cache_key_for_path(p: Path, extra: str = "") -> str:
    try:
        st = p.stat()
        key = f"{p.resolve().as_posix()}|{st.st_mtime_ns}|{st.st_size}|{extra}"
    except Exception:
        key = f"{p.as_posix()}|{extra}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def load_cached_embedding(cache_dir: Path, embedder_tag: str, img_path: Path) -> Optional[np.ndarray]:
    h = _cache_key_for_path(img_path, extra=embedder_tag)
    fp = cache_dir / f"{embedder_tag}__{h}.npy"
    if fp.exists():
        return np.load(fp).astype(np.float32).reshape(-1)
    return None


def save_cached_embedding(cache_dir: Path, embedder_tag: str, img_path: Path, vec: np.ndarray) -> None:
    h = _cache_key_for_path(img_path, extra=embedder_tag)
    fp = cache_dir / f"{embedder_tag}__{h}.npy"
    np.save(fp, vec.astype(np.float32).reshape(-1))


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def softmax_np(x: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 0:
        tau = 1e-6
    z = x / tau
    z = z - np.max(z)
    e = np.exp(z)
    s = np.sum(e)
    if s <= 0:
        return np.ones_like(x) / max(1, x.size)
    return e / s


def entropy_np(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def list_images_in_dir(d: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    out = []
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    out.sort()
    return out


def read_pair_list(txt_path: Path, data_root: Path) -> List[Path]:
    paths = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            lq = parts[0]
            p = Path(lq)
            if not p.is_absolute():
                p = (data_root / p).resolve()
            paths.append(p)
    return paths


def derive_embedder_tag(embedder: str, clip_model: str, embedder_tag: Optional[str]) -> str:
    if embedder_tag:
        return embedder_tag
    if embedder == "clip":
        return f"clip_{clip_model}".replace("/", "-").lower()
    if embedder == "dino_v2":
        return "dinov2_vitb14"
    if embedder == "fft":
        return "fft_amp"
    if embedder == "fft_enhanced":
        return "fft_enh"
    return embedder


def build_embedder(args, device: str):
    if args.embedder == "clip":
        from lora_adapters.embedding_clip import CLIPEmbedder
        return CLIPEmbedder(device=device, model_name=args.clip_model, pretrained=args.clip_pretrained)

    if args.embedder == "dino_v2":
        from lora_adapters.embedding_dinov2 import DINOv2Embedder
        if not args.dino_ckpt:
            raise ValueError("embedder=dino_v2 时必须提供 --dino_ckpt")
        return DINOv2Embedder(device=device, ckpt_path=args.dino_ckpt)

    if args.embedder == "fft":
        from lora_adapters.embedding_fft import FFTAmplitudeEmbedder
        return FFTAmplitudeEmbedder(device=device, resize=args.fft_resize, center_crop=args.fft_center_crop, out_size=args.fft_out_size)

    if args.embedder == "fft_enhanced":
        from lora_adapters.embedding_fft import FFTEnhancedEmbedder
        return FFTEnhancedEmbedder(device=device, resize=args.fft_resize, center_crop=args.fft_center_crop, out_size=args.fft_out_size, clean_proto_path=args.fft_clean_proto)

    raise ValueError(f"unknown embedder: {args.embedder}")


def embed_one_image(emb, img_path: Path) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    v = emb.embed_image(img)
    if isinstance(v, torch.Tensor):
        v = v.detach().float().cpu().numpy()
    v = np.asarray(v).astype(np.float32).reshape(-1)
    return v


def load_domain_prototypes(loradb_root: Path, domain: str, embedder_tag: str, force_avg: bool = False) -> Tuple[np.ndarray, str]:
    dom_dir = loradb_root / domain
    proto_fp = dom_dir / f"prototypes_{embedder_tag}.npy"
    avg_fp = dom_dir / f"avg_embedding_{embedder_tag}.npy"

    if (not force_avg) and proto_fp.exists():
        arr = np.load(proto_fp).astype(np.float32)
        src = str(proto_fp)
    elif avg_fp.exists():
        arr = np.load(avg_fp).astype(np.float32)
        src = str(avg_fp)
    else:
        legacy = dom_dir / "avg_embedding.npy"
        if legacy.exists():
            arr = np.load(legacy).astype(np.float32)
            src = str(legacy)
        else:
            raise FileNotFoundError(f"[prototype] domain={domain} missing: {proto_fp} / {avg_fp} / {legacy}")

    if arr.ndim == 1:
        arr = arr[None, :]
    return arr, src


def reduce_proto(values: np.ndarray, mode: str = "sum", topm: int = 2) -> float:
    if values.size == 0:
        return -1e9
    if mode == "sum":
        return float(values.sum())
    if mode == "mean":
        return float(values.mean())
    if mode == "max":
        return float(values.max())
    if mode == "topm_sum":
        m = min(topm, values.size)
        idx = np.argsort(values)[::-1][:m]
        return float(values[idx].sum())
    raise ValueError(f"unknown proto_reduce: {mode}")


def score_domains(query: np.ndarray, protos_by_domain: Dict[str, np.ndarray],
                 sim_metric: str, proto_reduce: str, proto_topm: int,
                 normalize: bool, euclidean_mode: str) -> Dict[str, float]:

    scores: Dict[str, float] = {}
    q = query.astype(np.float32)

    if sim_metric == "cosine":
        q0 = l2_normalize(q) if normalize else q
        for d, P in protos_by_domain.items():
            P0 = l2_normalize(P, axis=1) if normalize else P
            sims = P0 @ q0
            scores[d] = reduce_proto(sims, proto_reduce, proto_topm)

    elif sim_metric == "euclidean":
        q0 = l2_normalize(q) if normalize else q
        for d, P in protos_by_domain.items():
            P0 = l2_normalize(P, axis=1) if normalize else P
            dists = np.linalg.norm(P0 - q0[None, :], axis=1)

            if proto_reduce == "max":
                dist_agg = float(dists.min())
            elif proto_reduce == "sum":
                dist_agg = float(dists.mean())
            elif proto_reduce == "topm_sum":
                m = min(proto_topm, dists.size)
                idx = np.argsort(dists)[:m]
                dist_agg = float(dists[idx].mean())
            else:
                dist_agg = float(dists.mean())

            if euclidean_mode == "inv":
                scores[d] = 1.0 / (dist_agg + 1e-6)
            elif euclidean_mode == "neg":
                scores[d] = -dist_agg
            else:
                raise ValueError(f"unknown euclidean_mode: {euclidean_mode}")

    else:
        raise ValueError(f"unknown sim_metric: {sim_metric}")

    return scores


@dataclass
class RetrievalConfig:
    loradb_root: str
    domains: List[str]
    embedder: str
    embedder_tag: str
    sim_metric: str
    temperature: float
    topk: int
    proto_reduce: str
    proto_topm: int
    normalize: bool
    euclidean_mode: str
    force_avg: bool
    seed: int
    deterministic: bool
    device: str

    clip_model: str = "ViT-B-16"
    clip_pretrained: str = "openai"
    dino_ckpt: Optional[str] = None
    fft_resize: int = 256
    fft_center_crop: int = 0
    fft_out_size: int = 32
    fft_clean_proto: Optional[str] = None


@dataclass
class SetSpec:
    name: str
    true_domain: str
    true_domains: Optional[List[str]] = None  # <- 多标签（混合退化）
    input_dir: Optional[str] = None
    pair_list: Optional[str] = None
    data_root: str = "."

    def validate(self):
        if (self.input_dir is None) and (self.pair_list is None):
            raise ValueError(f"SetSpec({self.name}): 必须提供 input_dir 或 pair_list")
        if (self.input_dir is not None) and (self.pair_list is not None):
            raise ValueError(f"SetSpec({self.name}): input_dir 与 pair_list 二选一")
        if not self.true_domain:
            raise ValueError(f"SetSpec({self.name}): true_domain 不能为空")


def evaluate_one_set(cfg: RetrievalConfig, set_spec: SetSpec, out_dir: Path, cache_emb_dir: Optional[Path] = None) -> Dict[str, Any]:
    set_spec.validate()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if (cfg.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    # 1) prototypes
    loradb_root = Path(cfg.loradb_root)
    protos_by_domain: Dict[str, np.ndarray] = {}
    proto_manifest: Dict[str, Dict[str, Any]] = {}
    for d in cfg.domains:
        P, src = load_domain_prototypes(loradb_root, d, cfg.embedder_tag, force_avg=cfg.force_avg)
        protos_by_domain[d] = P
        proto_manifest[d] = {"path": src, "shape": list(P.shape)}

    # 2) embedder
    class _Args: pass
    a = _Args()
    a.embedder = cfg.embedder
    a.clip_model = cfg.clip_model
    a.clip_pretrained = cfg.clip_pretrained
    a.dino_ckpt = cfg.dino_ckpt
    a.fft_resize = cfg.fft_resize
    a.fft_center_crop = cfg.fft_center_crop
    a.fft_out_size = cfg.fft_out_size
    a.fft_clean_proto = cfg.fft_clean_proto
    emb = build_embedder(a, device=device)

    # 3) images
    if set_spec.input_dir is not None:
        imgs = list_images_in_dir(Path(set_spec.input_dir))
    else:
        imgs = read_pair_list(Path(set_spec.pair_list), Path(set_spec.data_root))

    gt_set = None
    if set_spec.true_domains:
        gt_set = [x for x in set_spec.true_domains if x]  # multi-label GT list

    per_rows: List[Dict[str, Any]] = []
    conf_top1 = {pd: 0 for pd in cfg.domains}
    freq_topk = {d: 0 for d in cfg.domains}
    wsum_topk = {d: 0.0 for d in cfg.domains}
    n_valid = 0

    t0 = time.time()
    for img_path in imgs:
        img_path = Path(img_path)
        if not img_path.exists():
            continue

        if cache_emb_dir is not None:
            cache_emb_dir.mkdir(parents=True, exist_ok=True)
            q = load_cached_embedding(cache_emb_dir, cfg.embedder_tag, img_path)
            if q is None:
                q = embed_one_image(emb, img_path)
                save_cached_embedding(cache_emb_dir, cfg.embedder_tag, img_path, q)
        else:
            q = embed_one_image(emb, img_path)

        raw_scores = score_domains(q, protos_by_domain, cfg.sim_metric, cfg.proto_reduce, cfg.proto_topm, cfg.normalize, cfg.euclidean_mode)
        sorted_all = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
        domains_sorted = [d for d, _ in sorted_all]
        scores_sorted = np.array([float(v) for _, v in sorted_all], dtype=np.float32)

        # single-label rank
        try:
            rank_true = domains_sorted.index(set_spec.true_domain) + 1
        except ValueError:
            rank_true = 999999

        # best rank among multi-label GTs
        rank_best_gt = None
        if gt_set:
            ranks = []
            for gd in gt_set:
                if gd in domains_sorted:
                    ranks.append(domains_sorted.index(gd) + 1)
            rank_best_gt = min(ranks) if ranks else 999999

        k = min(cfg.topk, len(sorted_all))
        top_names = domains_sorted[:k]
        top_vals = scores_sorted[:k]
        top_w = softmax_np(top_vals, cfg.temperature)

        # single-label hits
        hit1 = int(top_names[0] == set_spec.true_domain)
        hitk = int(set_spec.true_domain in top_names)

        # multi-label hits
        hit_any = None
        hit_all = None
        sum_weight_gt = None
        max_weight_gt = None
        if gt_set:
            top_set = set(top_names)
            gt_s = set(gt_set)
            hit_any = int(len(top_set.intersection(gt_s)) > 0)
            hit_all = int(gt_s.issubset(top_set))
            sw = 0.0
            mw = 0.0
            for dn, ww in zip(top_names, top_w.tolist()):
                if dn in gt_s:
                    sw += float(ww)
                    mw = max(mw, float(ww))
            sum_weight_gt = sw
            max_weight_gt = mw

        ent = entropy_np(top_w)
        if k >= 2:
            margin_score = float(top_vals[0] - top_vals[1])
            margin_w = float(top_w[0] - top_w[1])
        else:
            margin_score = 0.0
            margin_w = 0.0

        wt_true = 0.0
        if set_spec.true_domain in top_names:
            wt_true = float(top_w[top_names.index(set_spec.true_domain)])

        conf_top1[top_names[0]] += 1
        for dn, ww in zip(top_names, top_w.tolist()):
            freq_topk[dn] += 1
            wsum_topk[dn] += float(ww)

        per_rows.append({
            "set_name": set_spec.name,
            "true_domain": set_spec.true_domain,
            "true_domains": json.dumps(gt_set, ensure_ascii=False) if gt_set else "",
            "image": str(img_path),
            "pred1": top_names[0],

            "hit_top1": hit1,
            "hit_topk": hitk,
            "rank_true": rank_true,
            "weight_true_in_topk": wt_true,

            "hit_any": hit_any if hit_any is not None else "",
            "hit_all": hit_all if hit_all is not None else "",
            "rank_best_gt": rank_best_gt if rank_best_gt is not None else "",
            "sum_weight_gt": sum_weight_gt if sum_weight_gt is not None else "",
            "max_weight_gt": max_weight_gt if max_weight_gt is not None else "",

            "entropy_topk": ent,
            "margin_score_12": margin_score,
            "margin_weight_12": margin_w,

            "topk_domains": json.dumps(top_names, ensure_ascii=False),
            "topk_weights": json.dumps([float(x) for x in top_w], ensure_ascii=False),
            "topk_raw_scores": json.dumps([float(x) for x in top_vals], ensure_ascii=False),
            "raw_scores_all": json.dumps({k: float(v) for k, v in raw_scores.items()}, ensure_ascii=False),

            "embedder": cfg.embedder,
            "embedder_tag": cfg.embedder_tag,
            "sim_metric": cfg.sim_metric,
            "temperature": cfg.temperature,
            "topk": cfg.topk,
            "proto_reduce": cfg.proto_reduce,
            "proto_topm": cfg.proto_topm,
            "normalize": int(cfg.normalize),
            "euclidean_mode": cfg.euclidean_mode,
            "force_avg": int(cfg.force_avg),
            "seed": cfg.seed,
        })

        n_valid += 1

    dt = time.time() - t0

    # per_image.csv
    per_csv = out_dir / "per_image.csv"
    if per_rows:
        keys = list(per_rows[0].keys())
        with per_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in per_rows:
                w.writerow(r)

    def _mean(vals: List[float]) -> float:
        return float(sum(vals) / max(1, len(vals)))

    summary = {
        "set_name": set_spec.name,
        "true_domain": set_spec.true_domain,
        "true_domains": json.dumps(gt_set, ensure_ascii=False) if gt_set else "",

        "n_images_total": len(imgs),
        "n_images_used": n_valid,

        "top1_acc": _mean([float(r["hit_top1"]) for r in per_rows]) if per_rows else 0.0,
        "topk_acc": _mean([float(r["hit_topk"]) for r in per_rows]) if per_rows else 0.0,
        "mean_rank_true": _mean([float(r["rank_true"]) for r in per_rows]) if per_rows else 0.0,
        "mean_weight_true_in_topk": _mean([float(r["weight_true_in_topk"]) for r in per_rows]) if per_rows else 0.0,

        # multi-label aggregate（如果是单域则为空/0）
        "hit_any_acc": _mean([float(r["hit_any"]) for r in per_rows if r["hit_any"] != ""]) if gt_set else "",
        "hit_all_acc": _mean([float(r["hit_all"]) for r in per_rows if r["hit_all"] != ""]) if gt_set else "",
        "mean_rank_best_gt": _mean([float(r["rank_best_gt"]) for r in per_rows if r["rank_best_gt"] != ""]) if gt_set else "",
        "mean_sum_weight_gt": _mean([float(r["sum_weight_gt"]) for r in per_rows if r["sum_weight_gt"] != ""]) if gt_set else "",
        "mean_max_weight_gt": _mean([float(r["max_weight_gt"]) for r in per_rows if r["max_weight_gt"] != ""]) if gt_set else "",

        "mean_entropy_topk": _mean([float(r["entropy_topk"]) for r in per_rows]) if per_rows else 0.0,
        "mean_margin_weight_12": _mean([float(r["margin_weight_12"]) for r in per_rows]) if per_rows else 0.0,
        "mean_margin_score_12": _mean([float(r["margin_score_12"]) for r in per_rows]) if per_rows else 0.0,

        "time_sec": dt,

        "embedder": cfg.embedder,
        "embedder_tag": cfg.embedder_tag,
        "sim_metric": cfg.sim_metric,
        "temperature": cfg.temperature,
        "topk": cfg.topk,
        "proto_reduce": cfg.proto_reduce,
        "proto_topm": cfg.proto_topm,
        "normalize": int(cfg.normalize),
        "euclidean_mode": cfg.euclidean_mode,
        "force_avg": int(cfg.force_avg),
        "seed": cfg.seed,
    }

    # summary.csv
    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    # confusion_top1.csv
    conf_csv = out_dir / "confusion_top1.csv"
    with conf_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pred1", "count"])
        w.writeheader()
        for pd in cfg.domains:
            w.writerow({"pred1": pd, "count": conf_top1[pd]})

    # heatmaps
    heat_freq_csv = out_dir / "heatmap_freq_topk.csv"
    heat_w_csv = out_dir / "heatmap_mean_weight.csv"
    denom = max(1, n_valid)

    with heat_freq_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "freq_in_topk"])
        w.writeheader()
        for d in cfg.domains:
            w.writerow({"domain": d, "freq_in_topk": float(freq_topk[d]) / denom})

    with heat_w_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "mean_weight"])
        w.writeheader()
        for d in cfg.domains:
            w.writerow({"domain": d, "mean_weight": float(wsum_topk[d]) / denom})

    (out_dir / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "proto_manifest.json").write_text(json.dumps(proto_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary

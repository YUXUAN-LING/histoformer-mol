# -*- coding: utf-8 -*-
"""
PrototypeBank (robust)

- Auto-picks correct embedding/prototype file for each domain by:
  1) prefer files that match embedder_tag (fuzzy)
  2) MUST match expected_dim (e.g., CLIP ViT-B-16 -> 512)
  3) fallback to any file with matching dim
- Supports:
  - avg_embedding*.npy     (D,) or (1,D)
  - prototypes*.npy        (K,D)  -> reduced to mean (D,)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import glob
import numpy as np


@dataclass
class DomainProto:
    name: str
    proto: np.ndarray          # (D,)
    proto_path: Path
    lora_path: Path


def _norm_tag(s: str) -> str:
    s = s.lower().strip()
    # normalize common separators
    for ch in [" ", "-", "/", "\\", ":", ";", ",", "|", ".", "(", ")", "[", "]", "{", "}"]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _tag_tokens(tag: str) -> List[str]:
    tag = _norm_tag(tag)
    toks = [t for t in tag.split("_") if t]
    # keep only informative tokens
    drop = {"openai", "laion", "quickgelu", "clip", "vit", "b", "l", "h", "g"}
    keep = [t for t in toks if t not in drop and len(t) >= 2]
    return keep if keep else toks


def _load_to_vector(path: Path) -> Tuple[np.ndarray, int]:
    arr = np.load(path)
    if arr.ndim == 1:
        v = arr.astype(np.float32)
        return v.reshape(-1), v.shape[0]
    if arr.ndim == 2:
        # (K, D) prototypes OR (1, D)
        if arr.shape[0] == 1:
            v = arr[0].astype(np.float32)
            return v.reshape(-1), v.shape[-1]
        v = arr.mean(axis=0).astype(np.float32)  # reduce prototypes -> mean
        return v.reshape(-1), v.shape[-1]
    # rarely higher dims
    v = arr.reshape(-1).astype(np.float32)
    return v, v.shape[0]


class PrototypeBank:
    def __init__(
        self,
        domains: Iterable[str],
        loradb_root: str | Path,
        embedder_tag: Optional[str] = None,
        expected_dim: Optional[int] = None,
    ):
        self.loradb_root = Path(loradb_root)
        self.embedder_tag = embedder_tag
        self.expected_dim = int(expected_dim) if expected_dim is not None else None

        self.domains: Dict[str, DomainProto] = {}
        self._load_all(list(domains))

    def _pick_lora_path(self, dom_dir: Path, domain: str) -> Path:
        # prefer lora.pth then <domain>.pth then any *.pth
        for cand in [dom_dir / "lora.pth", dom_dir / f"{domain}.pth"]:
            if cand.exists():
                return cand
        pths = sorted(glob.glob(str(dom_dir / "*.pth")))
        if not pths:
            raise FileNotFoundError(f"[PrototypeBank] no .pth found under {dom_dir}")
        return Path(pths[0])

    def _candidate_npy(self, dom_dir: Path) -> List[Path]:
        # accept avg_embedding*.npy and prototypes*.npy
        cands = []
        cands += [Path(p) for p in glob.glob(str(dom_dir / "avg_embedding*.npy"))]
        cands += [Path(p) for p in glob.glob(str(dom_dir / "prototypes*.npy"))]
        # stable order
        cands = sorted(set(cands), key=lambda p: p.name)
        return cands

    def _score_name_match(self, fname: str) -> int:
        """higher is better"""
        if not self.embedder_tag:
            return 0
        fn = _norm_tag(fname)
        toks = _tag_tokens(self.embedder_tag)
        score = 0
        for t in toks:
            if t and t in fn:
                score += 1
        # bonus if contains 'clip' when tag hints clip
        if "clip" in _norm_tag(self.embedder_tag) and "clip" in fn:
            score += 2
        return score

    def _pick_proto_path(self, dom_dir: Path) -> Tuple[Path, np.ndarray]:
        cands = self._candidate_npy(dom_dir)
        if not cands:
            raise FileNotFoundError(
                f"[PrototypeBank] no avg_embedding*.npy / prototypes*.npy under {dom_dir}"
            )

        # pre-load dims and rank
        scored = []
        for p in cands:
            try:
                v, dim = _load_to_vector(p)
            except Exception:
                continue
            name_score = self._score_name_match(p.name)
            dim_ok = 1 if (self.expected_dim is None or dim == self.expected_dim) else 0
            # rank: dim_ok first, then name match, then prefer avg_embedding over prototypes for same dim
            prefer_avg = 1 if p.name.startswith("avg_embedding") else 0
            scored.append((dim_ok, name_score, prefer_avg, p, v, dim))

        if not scored:
            raise FileNotFoundError(f"[PrototypeBank] cannot load any npy under {dom_dir}")

        # filter by expected_dim if given
        if self.expected_dim is not None:
            dim_matched = [x for x in scored if x[-1] == self.expected_dim]
            if dim_matched:
                scored = dim_matched

        # choose best by (dim_ok, name_score, prefer_avg)
        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        best = scored[0]
        _, _, _, path, vec, dim = best
        return path, vec

    def _load_all(self, domains: List[str]):
        if not domains:
            raise ValueError("[PrototypeBank] domains is empty")

        for d in domains:
            dom_dir = self.loradb_root / d
            if not dom_dir.exists():
                raise FileNotFoundError(f"[PrototypeBank] domain folder not found: {dom_dir}")

            proto_path, proto = self._pick_proto_path(dom_dir)
            proto = proto.reshape(-1).astype(np.float32)

            if self.expected_dim is not None and proto.shape[0] != self.expected_dim:
                raise ValueError(
                    f"[PrototypeBank] still got dim mismatch for {d}: "
                    f"{proto.shape[0]} vs expected {self.expected_dim}. "
                    f"Picked {proto_path.name}"
                )

            lora_path = self._pick_lora_path(dom_dir, d)

            self.domains[d] = DomainProto(
                name=d,
                proto=proto,
                proto_path=proto_path,
                lora_path=lora_path,
            )

            print(
                f"[PrototypeBank] loaded {d}: proto={proto_path.name} dim={proto.shape[0]} "
                f"lora={lora_path.name}"
            )

    def names(self) -> List[str]:
        return list(self.domains.keys())

    def get(self, domain: str) -> DomainProto:
        return self.domains[domain]

    def proto_matrix(self) -> np.ndarray:
        names = self.names()
        return np.stack([self.domains[d].proto for d in names], axis=0)

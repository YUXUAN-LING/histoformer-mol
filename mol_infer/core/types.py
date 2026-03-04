# mol_infer/core/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------
# Stage-1: retrieval outputs
# ---------------------------
@dataclass
class RetrievalResult:
    """
    picks: top-k softmax weights (sorted desc), e.g. [("rain3", 0.41), ("snow1", 0.22), ...]
    raw_scores: optional full list before softmax, usually similarity or distance scores
    """
    picks: List[Tuple[str, float]]
    raw_scores: Optional[List[Tuple[str, float]]] = None


# ---------------------------
# Routing decision (single/mix)
# ---------------------------
@dataclass
class RoutingDecision:
    """
    mode:
      - "single": use top1 only
      - "mix": use local + global
    """
    mode: str  # "single" | "mix"

    top1: str
    top1_w: float
    top2_w: float
    margin: float

    # for mix
    local: Optional[str] = None
    local_w: Optional[float] = None
    global_: Optional[str] = None  # use global_ to avoid keyword 'global'
    global_w: Optional[float] = None

    reason: str = ""


# ---------------------------
# Configs
# ---------------------------
@dataclass
class IOConfig:
    # data
    input: str
    pair_list: Optional[str] = None
    gt_root: Optional[str] = None

    # output
    output_dir: str = "outputs"

    # save
    save_images: bool = True
    concat: bool = True
    save_singles: bool = False
    save_lq: bool = False
    annotate: bool = False

    # logs
    metrics_csv: Optional[str] = None
    routing_jsonl: Optional[str] = None
    summary_csv: Optional[str] = None


@dataclass
class ModelConfig:
    base_ckpt: str
    yaml: Optional[str]
    loradb_root: str
    domains: str  # comma-separated list (keep as str, runner splits)
    rank: int = 16
    alpha: float = 16.0
    enable_patch_lora: bool = False


@dataclass
class RetrievalConfig:
    embedder: str = "clip"  # "clip"
    clip_model: str = "ViT-B-16"
    clip_pretrained: Optional[str] = None  # can be local path or openai tag
    embedder_tag: Optional[str] = None  # must match avg_embedding_<tag>.npy if you use that convention

    sim_metric: str = "cosine"  # "cosine"|"euclidean"|"l2"
    temperature: float = 0.07
    topk: int = 5

    # convenience (not always needed, but helpful for orchestrator)
    loradb_root: Optional[str] = None
    domains: Optional[str] = None


@dataclass
class RoutingConfig:
    local_domains: str  # csv string
    global_domains: str  # csv string
    mix_topk: int = 5
    single_tau: float = 0.72
    single_margin: float = 0.10


@dataclass
class FusionConfig:
    """
    name:
      - "none": stage-1 only (base + retrieval + routing), no LoRA fusion applied
      - "kselect_static": two-choice local/global (static plan)
      - "kselect_activation": activation/hybrid (optional, if you wire it)
      - "kselect_none": three-choice local/global/none (base)
    """
    name: str = "none"

    # kselect common
    k_topk: int = 0
    k_score_mode: str = "topk_ratio"   # "topk_sum"|"mean"|"median"|"fro"|"topk_ratio"
    k_alpha: float = 1.0
    k_beta: float = 1.0
    k_ramp_mode: str = "sigmoid"       # "linear"|"sigmoid"
    use_gamma: bool = False
    gamma_mode: str = "per_layer"      # "none"|"global"|"per_layer"
    gamma_clip: float = 0.0

    # none-gating (3-choice) specific
    none_mode: str = "const"           # "const"|"ramp"
    none_alpha: float = 0.0
    none_beta: float = 0.15
    none_metric: str = "abs"           # "abs" or "ratio" (depends on your impl)


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    seed: int = 0
    deterministic: bool = False

    run_name: str = ""

    enable_lora: bool = False
    enable_fusion: bool = False

    print_full_scores: bool = False
    debug: bool = False


@dataclass
class RunConfig:
    io: IOConfig
    model: ModelConfig
    retrieval: RetrievalConfig
    routing: RoutingConfig
    fusion: FusionConfig
    runtime: RuntimeConfig


# ---------------------------
# Optional: generic dict-like record
# ---------------------------
Record = Dict[str, Any]

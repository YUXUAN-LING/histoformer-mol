# mol_infer/core/runner.py
from __future__ import annotations

import os
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from mol_infer.core.types import RunConfig, RetrievalResult
from mol_infer.io.dataset import build_pairs, load_image_tensor
from mol_infer.io.logging import JsonlLogger, CsvLogger, SummaryLogger
from mol_infer.io.image_save import save_triplet_images

from mol_infer.retrieval.embedder import build_embedder, Embedder
from mol_infer.retrieval.orchestrator import build_orchestrator, RetrievalOrchestrator
from mol_infer.routing.decision import decide_single_or_mix

from mol_infer.lora.adapter import HistoformerLoRAAdapter

# --- fusion (only import what we need) ---
from mol_infer.fusion.kselect_none import KSelectNoneConfig, KSelectNone
from mol_infer.fusion.kselect_static import KSelectStaticConfig
from mol_infer.fusion.kselect_static import KSelectStaticStrategy as KSelectStatic
from mol_infer.fusion.kselect_activation import KSelectActivationConfig
from mol_infer.fusion.kselect_activation import KSelectActivationStrategy as KSelectActivation


# --- metrics reuse from your existing repo ---
try:
    from lora_adapters.infer_data import tensor_psnr, tensor_ssim
except Exception as e:
    raise ImportError(f"Cannot import tensor_psnr/tensor_ssim from lora_adapters.infer_data: {e}")


def _ensure_dir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def _to_numpy_1d(x: Any) -> np.ndarray:
    """Normalize embedder output to shape [D] float32."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[0] == 1:
        x = x[0]
    return x.astype(np.float32).reshape(-1)


class Runner:
    """
    Stage-1/2 runner:
      - always runs base
      - retrieval + routing decides single vs mix
      - if enable_lora: run single/mix output
      - if enable_fusion and fusion==kselect_*: use fusion strategy on mix
    """

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.device = self._resolve_device(cfg.runtime.device)

        # parsed domain lists
        self.domains: List[str] = [d.strip() for d in str(cfg.retrieval.domains).split(",") if d.strip()]
        self.local_domains: List[str] = [d.strip() for d in str(cfg.routing.local_domains).split(",") if d.strip()]
        self.global_domains: List[str] = [d.strip() for d in str(cfg.routing.global_domains).split(",") if d.strip()]

        # loggers (created in setup)
        self.jsonl: Optional[JsonlLogger] = None
        self.metrics_logger: Optional[CsvLogger] = None
        self.summary_logger: Optional[SummaryLogger] = None

        # components
        self.adapter: Optional[HistoformerLoRAAdapter] = None
        self.embedder: Optional[Embedder] = None
        self.orch: Optional[RetrievalOrchestrator] = None
        self.fusion: Optional[Any] = None  # FusionStrategy-like

        # accumulators
        self.base_psnr: List[float] = []
        self.base_ssim: List[float] = []
        self.auto_psnr: List[float] = []
        self.auto_ssim: List[float] = []

    def _resolve_device(self, dev: str) -> str:
        dev = str(dev or "cuda")
        if dev.startswith("cuda") and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # --------------------------
    # setup
    # --------------------------
    def setup(self) -> None:
        cfg = self.cfg

        # output dirs
        _ensure_dir(cfg.io.output_dir)
        if cfg.io.save_images:
            _ensure_dir(os.path.join(cfg.io.output_dir, "triplets"))
            _ensure_dir(os.path.join(cfg.io.output_dir, "images"))

        # loggers
        self.jsonl = JsonlLogger(cfg.io.routing_jsonl) if cfg.io.routing_jsonl else None

        if cfg.io.metrics_csv:
            self.metrics_logger = CsvLogger(
                cfg.io.metrics_csv,
                header=[
                    "run_name", "name",
                    "mode", "top1", "top1_w", "top2", "top2_w", "margin",
                    "local", "local_w", "global", "global_w",
                    "psnr_base", "ssim_base", "psnr_auto", "ssim_auto",
                    "fusion",
                    "kselect_mode", "k_layers", "k_local", "k_global", "k_none",
                    "k_local_ratio", "k_global_ratio", "k_none_ratio",
                    "reason",
                    "topk_short_json",
                ],
            )

        if cfg.io.summary_csv:
            self.summary_logger = SummaryLogger(
                cfg.io.summary_csv,
                header=[
                    "run_name", "n",
                    "mean_psnr_base", "mean_ssim_base",
                    "mean_psnr_auto", "mean_ssim_auto",
                    "fusion",
                ],
            )

        # model adapter
        self.adapter = HistoformerLoRAAdapter(
            base_ckpt=str(cfg.model.base_ckpt),
            yaml_file=str(cfg.model.yaml),
            loradb_root=str(cfg.model.loradb_root),
            domains=self.domains,
            rank=int(cfg.model.rank),
            alpha=float(cfg.model.alpha),
            enable_patch_lora=bool(cfg.model.enable_patch_lora),
            device=self.device,
        )
        self.adapter.build()
        self.adapter.load_base()

        if cfg.runtime.enable_lora:
            self.adapter.inject_lora()
            # load all domain loras into memory (bank)
            self.adapter.load_all_domain_loras()

        # retrieval components
        # If embedder_tag is empty, try to infer a sane default for CLIP.
        # You said you have avg_embedding_clip_vit-b-16.npy => embedder_tag should be "clip_vit-b-16"
        if (cfg.retrieval.embedder_tag is None) or (str(cfg.retrieval.embedder_tag).strip() == ""):
            if str(cfg.retrieval.embedder).lower() == "clip":
                tag = f"clip_{str(cfg.retrieval.clip_model).lower()}"
                cfg.retrieval.embedder_tag = tag  # mutate in-place for consistency

        self.embedder = build_embedder(cfg.retrieval, device=self.device)
        self.orch = build_orchestrator(cfg.retrieval, domains=self.domains)

        # fusion strategy (optional)
        self.fusion = None
        if cfg.runtime.enable_fusion:
            fname = str(cfg.fusion.name or "").strip().lower()
            if fname == "kselect_none":
                self.fusion = KSelectNone(cfg.fusion.kselect_none, debug=cfg.runtime.debug)
            elif fname == "kselect_static":
                self.fusion = KSelectStatic(cfg.fusion.kselect_static, debug=cfg.runtime.debug)
            elif fname == "kselect_activation":
                self.fusion = KSelectActivation(cfg.fusion.kselect_activation, debug=cfg.runtime.debug)
            else:
                raise ValueError(f"Unknown fusion strategy: {cfg.fusion.name}")

    # --------------------------
    # run
    # --------------------------
    @torch.no_grad()
    def run(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if self.adapter is None or self.embedder is None or self.orch is None:
            self.setup()

        assert self.adapter is not None
        assert self.embedder is not None
        assert self.orch is not None

        cfg = self.cfg

        pairs = build_pairs(cfg.io.input, cfg.io.pair_list, cfg.io.gt_root)
        n = 0
        records: List[Dict[str, Any]] = []

        for pair in pairs:
            n += 1
            name = pair.name
            stem = pair.stem

            # load lq/gt
            lq = load_image_tensor(pair.lq_path).to(self.device)  # [1,3,H,W]
            gt = None
            if pair.gt_path:
                gt = load_image_tensor(pair.gt_path).to(self.device)

            # ---------------- base forward ----------------
            y_base = self.adapter.forward_base(lq).clamp(0, 1)

            # metrics for base
            psnr_base = float("nan")
            ssim_base = float("nan")
            if gt is not None:
                psnr_base = float(tensor_psnr(y_base, gt))
                ssim_base = float(tensor_ssim(y_base, gt))
                self.base_psnr.append(psnr_base)
                self.base_ssim.append(ssim_base)

            # ---------------- retrieval + routing ----------------
            emb = _to_numpy_1d(self.embedder.encode(lq))  # [D]
            ret: RetrievalResult = self.orch.retrieve(
                img_emb=emb,
                topk=int(cfg.retrieval.topk),
                temperature=float(cfg.retrieval.temperature),
                return_raw_scores=bool(cfg.runtime.print_full_scores),
            )

            if cfg.runtime.print_full_scores and ret.raw_scores is not None:
                print("[debug raw_scores]", ret.raw_scores[: min(10, len(ret.raw_scores))])

            dec = decide_single_or_mix(
                picks=ret.picks,
                local_domains=self.local_domains,
                global_domains=self.global_domains,
                mix_topk=int(cfg.routing.mix_topk),
                single_tau=float(cfg.routing.single_tau),
                single_margin=float(cfg.routing.single_margin),
            )

            # default: auto==base
            y_auto = y_base
            fusion_used = "none"
            kstats: Dict[str, Any] = {
                "kselect_mode": "",
                "k_layers": 0,
                "k_local": 0,
                "k_global": 0,
                "k_none": 0,
                "k_local_ratio": 0.0,
                "k_global_ratio": 0.0,
                "k_none_ratio": 0.0,
            }

            # ---------------- apply LoRA ----------------
            if cfg.runtime.enable_lora:
                if dec["mode"] == "single":
                    y_auto = self.adapter.forward_single(lq, domain=str(dec["top1"])).clamp(0, 1)
                    fusion_used = "single"
                else:
                    local = str(dec["local"])
                    global_ = str(dec["global"])

                    if cfg.runtime.enable_fusion and (self.fusion is not None):
                        fusion_used = str(cfg.fusion.name)
                        # KSelect mix (layer-wise local/global/none)
                        y_auto, ks = self.fusion.run(
                            adapter=self.adapter,
                            x=lq,
                            local_domain=local,
                            global_domain=global_,
                        )
                        y_auto = y_auto.clamp(0, 1)

                        # normalize kstats fields
                        if isinstance(ks, dict):
                            kstats.update({
                                "kselect_mode": str(getattr(cfg.fusion.kselect_none, "kselect_mode", "")) if fusion_used == "kselect_none" else str(getattr(cfg.fusion.kselect_static, "kselect_mode", "")),
                                "k_layers": int(ks.get("n_total", ks.get("k_layers", 0))),
                                "k_local": int(ks.get("n_local", ks.get("k_local", 0))),
                                "k_global": int(ks.get("n_global", ks.get("k_global", 0))),
                                "k_none": int(ks.get("n_none", ks.get("k_none", 0))),
                                "k_local_ratio": float(ks.get("local_ratio", ks.get("k_local_ratio", 0.0))),
                                "k_global_ratio": float(ks.get("global_ratio", ks.get("k_global_ratio", 0.0))),
                                "k_none_ratio": float(ks.get("none_ratio", ks.get("k_none_ratio", 0.0))),
                            })
                    else:
                        # simple soft weighted fusion (no layerwise)
                        w_local = float(dec.get("local_w", 0.5))
                        w_global = float(dec.get("global_w", 0.5))
                        self.adapter.set_domain_weights({local: w_local, global_: w_global})
                        y_auto = self.adapter.forward_padded(lq).clamp(0, 1)
                        fusion_used = "linear_mix"

            # ---------------- metrics for auto ----------------
            psnr_auto = float("nan")
            ssim_auto = float("nan")
            if gt is not None:
                psnr_auto = float(tensor_psnr(y_auto, gt))
                ssim_auto = float(tensor_ssim(y_auto, gt))
                self.auto_psnr.append(psnr_auto)
                self.auto_ssim.append(ssim_auto)

            # ---------------- save images ----------------
            if cfg.io.save_images:
                meta = {
                    "run_name": cfg.runtime.run_name,
                    "mode": dec["mode"],
                    "top1": dec.get("top1"),
                    "top2": dec.get("top2"),
                    "local": dec.get("local"),
                    "global": dec.get("global"),
                    "fusion": fusion_used,
                    "kstats": kstats,
                }
                save_triplet_images(
                    output_dir=cfg.io.output_dir,
                    stem=stem,
                    lq=lq,
                    base=y_base,
                    mix=y_auto,
                    save_concat=bool(cfg.io.concat),
                    save_singles=bool(cfg.io.save_singles),
                    save_lq=bool(cfg.io.save_lq),
                    annotate=bool(cfg.io.annotate),
                    meta=meta,
                )

            # ---------------- logging ----------------
            topk_short_json = json.dumps(ret.picks[: min(8, len(ret.picks))], ensure_ascii=False)

            rec = {
                "run_name": cfg.runtime.run_name,
                "name": name,
                "mode": dec["mode"],
                "top1": dec.get("top1", ""),
                "top1_w": float(dec.get("top1_w", 0.0)),
                "top2": dec.get("top2", ""),
                "top2_w": float(dec.get("top2_w", 0.0)),
                "margin": float(dec.get("margin", 0.0)),
                "local": dec.get("local", ""),
                "local_w": float(dec.get("local_w", 0.0)),
                "global": dec.get("global", ""),
                "global_w": float(dec.get("global_w", 0.0)),
                "psnr_base": psnr_base,
                "ssim_base": ssim_base,
                "psnr_auto": psnr_auto,
                "ssim_auto": ssim_auto,
                "fusion": fusion_used,
                "kselect_mode": kstats["kselect_mode"],
                "k_layers": kstats["k_layers"],
                "k_local": kstats["k_local"],
                "k_global": kstats["k_global"],
                "k_none": kstats["k_none"],
                "k_local_ratio": kstats["k_local_ratio"],
                "k_global_ratio": kstats["k_global_ratio"],
                "k_none_ratio": kstats["k_none_ratio"],
                "reason": dec.get("reason", ""),
                "topk_short_json": topk_short_json,
            }
            records.append(rec)

            if self.metrics_logger is not None:
                self.metrics_logger.write([
                    rec["run_name"], rec["name"],
                    rec["mode"], rec["top1"], rec["top1_w"], rec["top2"], rec["top2_w"], rec["margin"],
                    rec["local"], rec["local_w"], rec["global"], rec["global_w"],
                    rec["psnr_base"], rec["ssim_base"], rec["psnr_auto"], rec["ssim_auto"],
                    rec["fusion"],
                    rec["kselect_mode"], rec["k_layers"], rec["k_local"], rec["k_global"], rec["k_none"],
                    rec["k_local_ratio"], rec["k_global_ratio"], rec["k_none_ratio"],
                    rec["reason"],
                    rec["topk_short_json"],
                ])

            if self.jsonl is not None:
                self.jsonl.write({
                    "name": name,
                    "lq_path": pair.lq_path,
                    "gt_path": pair.gt_path,
                    "decision": dec,
                    "picks": ret.picks,
                    "fusion": fusion_used,
                    "kstats": kstats,
                })

        # ---------------- summary ----------------
        mean_psnr_base = float(np.mean(self.base_psnr)) if self.base_psnr else float("nan")
        mean_ssim_base = float(np.mean(self.base_ssim)) if self.base_ssim else float("nan")
        mean_psnr_auto = float(np.mean(self.auto_psnr)) if self.auto_psnr else float("nan")
        mean_ssim_auto = float(np.mean(self.auto_ssim)) if self.auto_ssim else float("nan")

        summary = {
            "run_name": cfg.runtime.run_name,
            "n": int(n),
            "mean_psnr_base": mean_psnr_base,
            "mean_ssim_base": mean_ssim_base,
            "mean_psnr_auto": mean_psnr_auto,
            "mean_ssim_auto": mean_ssim_auto,
            "fusion": str(cfg.fusion.name if cfg.runtime.enable_fusion else "none"),
        }

        if self.summary_logger is not None:
            self.summary_logger.write([
                cfg.runtime.run_name, n,
                mean_psnr_base, mean_ssim_base,
                mean_psnr_auto, mean_ssim_auto,
                summary["fusion"],
            ])

        print(f"[DONE] stage-1/2 finished [SUMMARY] n={n}")
        return records, summary

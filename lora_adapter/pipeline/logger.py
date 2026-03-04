# -*- coding: utf-8 -*-
"""lora_adapters.pipeline.logger

Logging utilities:
- routes.jsonl (per-image routing/decision/mix summary)
- metrics.csv  (per-image PSNR/SSIM)
- summary.csv  (mean PSNR/SSIM)

This file is small by design. Replace with your own experiment tracker later.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class RunLogger:
    out_dir: Path
    routes_jsonl: bool = True
    metrics_csv: bool = True
    summary_csv: bool = True

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._routes_fp = None
        self._csv_fp = None
        self._csv_writer = None

        self._sum_psnr = 0.0
        self._sum_ssim = 0.0
        self._count = 0

        if self.routes_jsonl:
            p = self.out_dir / "routes.jsonl"
            _ensure_parent(p)
            self._routes_fp = open(p, "w", encoding="utf-8")

        if self.metrics_csv:
            p = self.out_dir / "metrics.csv"
            _ensure_parent(p)
            self._csv_fp = open(p, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_fp, fieldnames=["name", "psnr", "ssim"])
            self._csv_writer.writeheader()

    def log_item(
        self,
        name: str,
        psnr: Optional[float],
        ssim: Optional[float],
        payload: Optional[Dict[str, Any]] = None,
    ):
        if psnr is not None and ssim is not None:
            self._sum_psnr += float(psnr)
            self._sum_ssim += float(ssim)
            self._count += 1

        if self._csv_writer is not None and psnr is not None and ssim is not None:
            self._csv_writer.writerow({"name": name, "psnr": f"{psnr:.4f}", "ssim": f"{ssim:.6f}"})
            self._csv_fp.flush()

        if self._routes_fp is not None and payload is not None:
            obj = {"name": name, **payload}
            self._routes_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._routes_fp.flush()

    def finalize(self):
        if self.summary_csv:
            p = self.out_dir / "summary.csv"
            _ensure_parent(p)
            mean_psnr = self._sum_psnr / max(1, self._count)
            mean_ssim = self._sum_ssim / max(1, self._count)
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["count", "mean_psnr", "mean_ssim"])
                w.writeheader()
                w.writerow({"count": self._count, "mean_psnr": f"{mean_psnr:.4f}", "mean_ssim": f"{mean_ssim:.6f}"})

        if self._routes_fp is not None:
            self._routes_fp.close()
            self._routes_fp = None

        if self._csv_fp is not None:
            self._csv_fp.close()
            self._csv_fp = None
            self._csv_writer = None

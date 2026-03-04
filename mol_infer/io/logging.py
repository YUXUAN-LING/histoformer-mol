# mol_infer/io/logging.py
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


class JsonlLogger:
    def __init__(self, path: Optional[str]):
        self.path = path
        if self.path:
            os.makedirs(str(Path(self.path).parent), exist_ok=True)

    def log(self, obj: Dict[str, Any]):
        self.write(obj)

    def write(self, obj: Dict[str, Any]):
        if not self.path:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # compatibility aliases
    def append(self, obj: Dict[str, Any]):
        self.write(obj)

    def __call__(self, obj: Dict[str, Any]):
        self.write(obj)


class CsvLogger:
    """
    一个简单但稳定的 CSV logger：
      - 第一次写会自动写 header
      - 后续写只按 header 顺序填充，没有的字段写 None
    """
    def __init__(self, path: Optional[str], header: List[str]):
        self.path = path
        self.header = list(header)
        if self.path:
            _ensure_parent(self.path)
            self._ensure_header()

    def _ensure_header(self):
        if not self.path:
            return
        if os.path.isfile(self.path) and os.path.getsize(self.path) > 0:
            return
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.header)

    def write(self, row: Dict[str, Any]):
        if not self.path:
            return
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([row.get(k, None) for k in self.header])


@dataclass
class SummaryLogger:
    """
    用于写 summary.csv：每次 run 写一行
    """
    path: Optional[str]
    header: List[str]

    def __post_init__(self):
        if self.path:
            _ensure_parent(self.path)
            if not os.path.isfile(self.path) or os.path.getsize(self.path) == 0:
                with open(self.path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(self.header)

    def write(self, row: Dict[str, Any]):
        if not self.path:
            return
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(k, None) for k in self.header])

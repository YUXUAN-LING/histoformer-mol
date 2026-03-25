from __future__ import annotations

import csv
import json
import os
from typing import Iterable, List, Tuple


def list_images(folder: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    if os.path.isfile(folder):
        return [folder]
    return [os.path.join(folder, fn) for fn in sorted(os.listdir(folder)) if fn.lower().endswith(exts)]


def parse_pair_list(path: str) -> List[Tuple[str, str | None]]:
    out: List[Tuple[str, str | None]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            out.append((parts[0], parts[1] if len(parts) > 1 else None))
    return out


def write_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: Iterable[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[dict]):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

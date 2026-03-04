# lora_adapters/make_eval_sets.py
# -*- coding: utf-8 -*-
"""
从 data/txt_lists 下的 val_list_*.txt 自动生成 eval_sets.json

默认规则：
- 文件名: val_list_<name>.txt
- name 会作为 set_name
- true_domain 默认等于 name
- 如果 name 里含 '_'（如 haze_rain），则额外写入 true_domains 多标签列表 ['haze','rain']
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt_root", type=str, required=True, help="例如 data/txt_lists")
    ap.add_argument("--pattern", type=str, default="val_list_*.txt")
    ap.add_argument("--out", type=str, default="eval_sets.json")
    ap.add_argument("--data_root", type=str, default=".", help="pair_list 里相对路径的根")
    args = ap.parse_args()

    txt_root = Path(args.txt_root)
    files = sorted(txt_root.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"在 {txt_root} 下找不到 {args.pattern}")

    sets: List[Dict] = []
    for fp in files:
        stem = fp.stem  # val_list_haze1
        if not stem.startswith("val_list_"):
            continue
        name = stem[len("val_list_"):]  # haze1 / haze_rain / ...
        item = {
            "name": name,
            "pair_list": str(fp.as_posix()),
            "data_root": args.data_root,
        }

        # 单标签：默认 true_domain = name
        item["true_domain"] = name

        # 多标签：如果 name 里带 '_'，解析成多个域
        if "_" in name:
            parts = [p for p in name.split("_") if p]
            item["true_domains"] = parts  # 用于 multi-label 评估

        sets.append(item)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(sets, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVE] {out_path} | num_sets={len(sets)}")
    print("example item:", sets[0])


if __name__ == "__main__":
    main()

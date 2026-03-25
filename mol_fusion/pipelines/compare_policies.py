from __future__ import annotations

import argparse
import subprocess


def main():
    ap = argparse.ArgumentParser("compare-policies")
    ap.add_argument("--commands_file", required=True, help="Text file, one infer_pairwise command per line")
    args = ap.parse_args()

    with open(args.commands_file, "r", encoding="utf-8") as f:
        commands = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    for cmd in commands:
        print(f"[RUN] {cmd}")
        subprocess.run(cmd, shell=True, check=False)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run exactly the five ablations reported in Figures 3 and 5."""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/litd_benchmark.json")
    parser.add_argument("--manifest", default="configs/ablation_manifest.csv")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    manifest = ROOT / args.manifest
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    selected = [row for row in rows if not args.only or row["ablation_name"] in args.only]
    for row in selected:
        command = [sys.executable, str(ROOT / "workflows/ablation/run_one_ablation.py"),
                   "--config", args.config, "--ablation-name", row["ablation_name"],
                   "--input-mode", row["input_mode"], "--lr-mode", row["lr_mode"],
                   "--pair-mode", row["pair_mode"]]
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

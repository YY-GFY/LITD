#!/usr/bin/env python3
"""End-to-end post-hoc workflow for Figure 6 and the distinct-TF relay analysis."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/litd_biological_case.json")
    parser.add_argument("--figure6-config", default="configs/figure6_fixed_path.json")
    parser.add_argument("--relay-config", default="configs/relay_internal_validation.json")
    parser.add_argument(
        "--stage", choices=["windows", "pathscore", "figure6", "relay", "relay-validation", "all"],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    py = sys.executable
    commands = {
        "windows": [py, str(HERE / "export_window_artifacts.py"), "--config", args.model_config],
        "pathscore": [
            py, str(HERE / "run_weighted_pathscore.py"), "--config", args.model_config,
            "--out-dir", "outputs/biological_case/pathscore", "--device", "cpu",
        ],
        "figure6": [py, str(HERE / "run_fixed_path_validation.py"), "--config", args.figure6_config],
        "relay": [
            py, str(HERE / "run_full_structural_relay_distinct_tf.py"), "--config", args.model_config,
            "--out-dir", "outputs/biological_case/relay", "--device", "cpu",
        ],
        "relay-validation": [py, str(HERE / "evaluate_internal_validation.py"), "--config", args.relay_config],
    }
    stages = ["windows", "pathscore", "figure6", "relay", "relay-validation"] if args.stage == "all" else [args.stage]
    for stage in stages:
        run(commands[stage], args.dry_run)


if __name__ == "__main__":
    main()

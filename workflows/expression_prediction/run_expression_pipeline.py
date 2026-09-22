#!/usr/bin/env python3
"""End-to-end driver for the six expression-prediction methods reported in Figure S1."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_driver_config(path_str: str) -> dict:
    """Load the small command-plan config without importing data dependencies."""
    path = Path(path_str)
    path = path if path.is_absolute() else ROOT / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    base_name = payload.get("base_config")
    if not base_name:
        return payload
    base_path = path.parent / base_name
    base = json.loads(base_path.read_text(encoding="utf-8"))
    base.update({key: value for key, value in payload.items() if key != "base_config"})
    return base


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/expression_prediction.json")
    parser.add_argument("--stage", choices=["prepare", "methods", "collect", "all"], default="all")
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--skip-external", action="store_true", help="Skip PRESCIENT, whose upstream source is installed separately.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command plan without reading data or using a GPU.")
    args = parser.parse_args()
    cfg = load_driver_config(args.config)
    datasets = args.datasets or cfg["datasets"]
    methods = args.methods or cfg["reported_methods"]
    python = sys.executable

    if args.stage in {"prepare", "all"}:
        run([python, str(HERE / "prepare_expression_data.py"), "--config", args.config, "--datasets", *datasets], args.dry_run)
    if args.stage in {"methods", "all"}:
        for method in methods:
            if method == "PRESCIENT":
                if args.skip_external:
                    continue
                runner = Path(cfg["prescient"]["runner"])
                runner = runner if runner.is_absolute() else ROOT / runner
                if not runner.is_file() and not args.dry_run:
                    raise FileNotFoundError(
                        f"PRESCIENT adapter not found: {runner}. Install the upstream implementation as documented, "
                        "or use --skip-external when regenerating only the repository-native methods."
                    )
                for dataset in datasets:
                    run([python, str(runner), "--config", args.config, "--dataset", dataset], args.dry_run)
                continue
            for dataset in datasets:
                run([
                    python, str(HERE / "run_reported_method.py"), "--config", args.config,
                    "--method", method, "--dataset", dataset,
                ], args.dry_run)
    if args.stage in {"collect", "all"}:
        command = [python, str(HERE / "collect_results.py"), "--config", args.config]
        if args.skip_external:
            command.append("--allow-missing")
        run(command, args.dry_run)


if __name__ == "__main__":
    main()

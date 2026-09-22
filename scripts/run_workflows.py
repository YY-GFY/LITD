#!/usr/bin/env python3
"""Unified entry point for the manuscript analysis workflows."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def latent_paths(config: Path) -> tuple[Path, Path]:
    payload = json.loads(config.read_text(encoding="utf-8"))
    base = json.loads((config.parent / payload["base_config"]).read_text(encoding="utf-8"))
    base.update({key: value for key, value in payload.items() if key != "base_config"})
    root = Path(base["root"])
    root = root if root.is_absolute() else ROOT / root
    analysis = root / "analysis" / base["analysis_name"]
    return analysis / "export", analysis / "coordinates" / "cell_umap_coordinates_full_24425.csv"


def workflow_commands(name: str, py: str) -> list[list[str]]:
    if name == "expression":
        return [[py, "workflows/expression_prediction/run_expression_pipeline.py", "--config", "configs/expression_prediction.json"]]
    if name == "biological-case":
        return [[py, "workflows/biological_case/run_biological_case_pipeline.py"]]
    if name == "weak-correlation":
        return [[py, "workflows/weak_correlation/run_weak_correlation_pipeline.py", "--config", "configs/weak_correlation.json"]]
    if name == "latent-interpretability":
        config = ROOT / "configs" / "latent_interpretability.json"
        export_dir, coordinate_file = latent_paths(config)
        return [
            [py, "workflows/latent_interpretability/export_hierarchical_embeddings.py", "--config", str(config)],
            [py, "workflows/latent_interpretability/analyze_latent_spaces.py", "--config", str(config)],
            [py, "workflows/latent_interpretability/compute_full_cell_umap_s5.py", "--export-dir", str(export_dir), "--output", str(coordinate_file), "--seed", "42", "--n-neighbors", "30", "--min-dist", "0.3"],
            [py, "workflows/latent_interpretability/validate_outputs.py", "--config", str(config)],
        ]
    if name == "figures":
        return [[py, "figures/build_paper_figures.py"]]
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choices = ["expression", "biological-case", "weak-correlation", "latent-interpretability", "figures", "all"]
    parser.add_argument("--workflow", choices=choices, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running data, training or inference.")
    parser.add_argument(
        "--skip-external", action="store_true",
        help="Skip the separately installed PRESCIENT method in the expression workflow.",
    )
    args = parser.parse_args()
    selected = choices[:-1] if args.workflow == "all" else [args.workflow]
    for name in selected:
        print(f"[{name}]", flush=True)
        for command in workflow_commands(name, sys.executable):
            if name == "expression" and args.skip_external:
                command = [*command, "--skip-external"]
            if args.dry_run and name in {"expression", "biological-case", "weak-correlation"}:
                run([*command, "--dry-run"], dry_run=False)
            else:
                run(command, args.dry_run)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the fixed LITD benchmark configuration reported in the paper."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _config import ROOT, benchmark_paths, load_config, run_args
from litd import benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/litd_benchmark.json")
    parser.add_argument("--stage", choices=("prepare", "train", "infer", "all"), default="all")
    parser.add_argument("--datasets", nargs="*", help="Defaults to the datasets in the config.")
    parser.add_argument("--model", help="Checkpoint used with --stage infer.")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    cfg = load_config(cli.config)
    paths = benchmark_paths(cfg)
    args = run_args(cfg)
    datasets = cli.datasets or cfg["datasets"]
    records: list[dict] = []

    for dataset in datasets:
        if cli.stage in {"prepare", "all"}:
            benchmark.prepare_dataset(paths, dataset)
            benchmark.ensure_embedding(paths, dataset, pre_normalized="T", tgthighres="t4")
        if cli.stage == "prepare":
            continue

        model_path = cli.model
        if cli.stage in {"train", "all"}:
            model_path = benchmark.train_one(paths, dataset, args)
        if cli.stage == "train":
            records.append({"dataset": dataset, "model": model_path})
            continue
        if not model_path:
            raise ValueError("--model is required when --stage infer is used")

        predictions = benchmark.infer_targets(
            paths, dataset, args, model_path,
            tuple(cfg["inference"]["target_gene_counts"]),
        )
        for target_count, prediction in predictions.items():
            for row in benchmark.evaluate_one(paths, dataset, prediction, target_count):
                row.update({"run_name": args.run_name, "target_genes": target_count,
                            "pred_path": prediction})
                records.append(row)

    if records:
        output = Path(paths.workdir) / "paper_metrics.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()

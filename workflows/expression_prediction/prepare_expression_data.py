#!/usr/bin/env python3
"""Prepare shared hESC/hHep data and the fixed 90/10 pair split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from expression_common import benchmark_paths, create_split, get_benchmark, load_config, split_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/expression_prediction.json")
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--skip-embedding", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    benchmark = get_benchmark()
    paths = benchmark_paths(cfg)
    datasets = args.datasets or cfg["datasets"]
    rows = []
    for dataset in datasets:
        benchmark.prepare_dataset(paths, dataset)
        if not args.skip_embedding:
            benchmark.ensure_embedding(paths, dataset, pre_normalized="T", tgthighres="t4")
        runtime = benchmark.load_runtime_paths(paths, dataset)
        pairs = pd.read_csv(runtime["pairs_csv"])
        train_idx, test_idx = create_split(len(pairs), float(cfg["train_fraction"]), int(cfg["split_seed"]))
        train_path, test_path = split_paths(cfg, dataset)
        train_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(train_path, train_idx)
        np.save(test_path, test_idx)
        pairs.iloc[train_idx].to_csv(train_path.with_name(f"{dataset}_train_pairs.csv"), index=False)
        pairs.iloc[test_idx].to_csv(test_path.with_name(f"{dataset}_test_pairs.csv"), index=False)
        rows.append({
            "dataset": dataset, "n_pairs": len(pairs), "n_train": len(train_idx),
            "n_test": len(test_idx), "split_seed": int(cfg["split_seed"]),
            "train_fraction": float(cfg["train_fraction"]),
        })
    out = Path(cfg["root"]) / "shared" / "splits" / "split_summary.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

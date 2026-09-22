#!/usr/bin/env python3
"""Run one of the five paper ablations using the fixed LITD benchmark configuration."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from _config import benchmark_paths, load_config, run_args  # noqa: E402
from litd import benchmark  # noqa: E402

SUPPORTED = {
    ("scfoundation", "masked", "shuffled_temporal"),
    ("scfoundation", "masked", "same_time"),
    ("scfoundation", "off", "temporal"),
    ("scfoundation", "dense", "temporal"),
    ("raw_expr", "masked", "temporal"),
}


def shuffled_pairs(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    groups = []
    columns = ["experiment", "cell_line", "day_source", "day_target"]
    for _, group in frame.groupby(columns, dropna=False, sort=False):
        group = group.copy()
        targets = group["target_index"].astype(str).tolist()
        if len(targets) > 1:
            shuffled = targets[:]
            rng.shuffle(shuffled)
            if shuffled == targets:
                offset = rng.randint(1, len(targets) - 1)
                shuffled = shuffled[offset:] + shuffled[:offset]
            group["target_index"] = shuffled
        groups.append(group)
    return pd.concat(groups, ignore_index=True)


def rewrite_pairs(path: Path, mode: str, seed: int) -> None:
    frame = pd.read_csv(path)
    if mode == "shuffled_temporal":
        frame = shuffled_pairs(frame, seed)
    elif mode == "same_time":
        frame["target_index"] = frame["source_index"].astype(str)
        frame["probability"] = 1.0
        frame["day_target"] = frame["day_source"]
    elif mode != "temporal":
        raise ValueError(f"unsupported pair mode: {mode}")
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/litd_benchmark.json")
    parser.add_argument("--ablation-name", required=True)
    parser.add_argument("--input-mode", required=True)
    parser.add_argument("--lr-mode", required=True)
    parser.add_argument("--pair-mode", required=True)
    parser.add_argument("--datasets", nargs="*")
    cli = parser.parse_args()
    if (cli.input_mode, cli.lr_mode, cli.pair_mode) not in SUPPORTED:
        raise ValueError("configuration is not one of the five paper ablations")

    cfg = load_config(cli.config)
    paths = benchmark_paths(cfg)
    paths = benchmark.Paths(**{**paths.__dict__,
        "workdir": str(ROOT / "outputs" / "ablation" / cli.ablation_name)})
    args = run_args(cfg, run_name=f'{cfg["run_name"]}__{cli.ablation_name}',
                    input_mode=cli.input_mode, lr_mode=cli.lr_mode)
    records = []
    for dataset in cli.datasets or cfg["datasets"]:
        benchmark.prepare_dataset(paths, dataset)
        benchmark.ensure_embedding(paths, dataset, pre_normalized="T", tgthighres="t4")
        pair_path = Path(paths.workdir) / "datasets" / dataset / "data" / "cell_pairs_train.csv"
        rewrite_pairs(pair_path, cli.pair_mode, cfg["seed"])
        model_path = benchmark.train_one(paths, dataset, args)
        predictions = benchmark.infer_targets(paths, dataset, args, model_path,
                                               tuple(cfg["inference"]["target_gene_counts"]))
        for target_count, prediction in predictions.items():
            for row in benchmark.evaluate_one(paths, dataset, prediction, target_count):
                row.update({"ablation_name": cli.ablation_name, "dataset": dataset,
                            "target_genes": target_count, "pred_path": prediction})
                records.append(row)
    output = Path(paths.workdir) / "ablation_metrics.json"
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

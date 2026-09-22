#!/usr/bin/env python3
"""Evaluate LITD edges with the paper benchmark or a prefiltered candidate space."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _config import benchmark_paths, load_config  # noqa: E402
from litd import benchmark  # noqa: E402


def evaluate(true_df: pd.DataFrame, pred_df: pd.DataFrame, n_possible: int) -> dict[str, float | int]:
    true_df = true_df[true_df.Gene1 != true_df.Gene2].drop_duplicates(["Gene1", "Gene2"])
    pred_df = pred_df[pred_df.Gene1 != pred_df.Gene2].drop_duplicates(["Gene1", "Gene2"]).copy()
    pred_df["EdgeWeight"] = pred_df.EdgeWeight.abs()
    truth = set(zip(true_df.Gene1.astype(str), true_df.Gene2.astype(str)))
    k = min(len(truth), len(pred_df))
    if not truth or not k or n_possible <= 0:
        return {"Eprec": 0.0, "Erec": 0.0, "EPR": 0.0, "selected_edges": 0}
    ranked = pred_df.sort_values(["EdgeWeight", "Gene1", "Gene2"], ascending=[False, True, True])
    cutoff = ranked.iloc[k - 1].EdgeWeight
    selected = set(zip(ranked.loc[ranked.EdgeWeight >= cutoff, "Gene1"].astype(str),
                       ranked.loc[ranked.EdgeWeight >= cutoff, "Gene2"].astype(str)))
    hits = len(selected & truth)
    eprec = hits / len(selected)
    erec = hits / len(truth)
    return {"Eprec": eprec, "Erec": erec,
            "EPR": eprec / (len(truth) / n_possible), "selected_edges": len(selected)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-benchmark",
        action="store_true",
        help="Reconstruct the paper's dataset-specific candidate space and evaluate all reference networks.",
    )
    parser.add_argument("--config", default="configs/litd_benchmark.json")
    parser.add_argument("--dataset", choices=("hesc", "hhep"))
    parser.add_argument("--ground-truth")
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--tf-count", type=int)
    parser.add_argument("--target-count", type=int)
    parser.add_argument("--overlap-count", type=int, default=0)
    parser.add_argument("--output", default="outputs/epr_metrics.json")
    args = parser.parse_args()

    if args.paper_benchmark:
        if args.dataset is None or args.target_count is None:
            parser.error("--paper-benchmark requires --dataset and --target-count")
        if args.target_count not in (500, 1000):
            parser.error("--paper-benchmark supports --target-count 500 or 1000")
        config = load_config(args.config)
        paths = benchmark_paths(config)
        result = benchmark.evaluate_one(
            paths,
            args.dataset,
            args.prediction,
            args.target_count,
        )
    else:
        if args.ground_truth is None or args.tf_count is None or args.target_count is None:
            parser.error(
                "prefiltered evaluation requires --ground-truth, --tf-count and --target-count"
            )
        n_possible = args.tf_count * args.target_count - args.overlap_count
        result = evaluate(pd.read_csv(args.ground_truth), pd.read_csv(args.prediction), n_possible)

    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

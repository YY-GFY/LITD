#!/usr/bin/env python3
"""Validate and combine the six reported expression-prediction outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_metrics import compute_metrics
from expression_common import load_config, method_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/expression_prediction.json")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows, errors = [], []
    for method in cfg["reported_methods"]:
        for dataset in cfg["datasets"]:
            root = method_dir(cfg, method, dataset)
            pred_path = root / "predictions" / "test_predictions.npy"
            true_path = root / "predictions" / "test_targets.npy"
            index_path = root / "predictions" / "test_indices.npy"
            if not all(path.is_file() for path in (pred_path, true_path, index_path)):
                errors.append(f"missing {method}/{dataset} prediction contract under {root}")
                continue
            pred, true = np.load(pred_path), np.load(true_path)
            metrics = compute_metrics(true, pred)
            rows.append({"dataset": dataset, "method": method, **metrics})
    if errors and not args.allow_missing:
        raise FileNotFoundError("\n".join(errors))
    out_dir = Path(cfg["root"]) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["dataset", "method"]) if rows else pd.DataFrame()
    frame.to_csv(out_dir / "expression_prediction_results.csv", index=False)
    report = {
        "status": "ok" if not errors else "incomplete",
        "expected_rows": len(cfg["datasets"]) * len(cfg["reported_methods"]),
        "observed_rows": len(rows),
        "errors": errors,
        "paper_snapshot": cfg["paper_snapshot"],
        "note": "The included paper snapshot is immutable provenance; newly computed results are written separately.",
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

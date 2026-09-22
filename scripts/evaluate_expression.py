#!/usr/bin/env python3
"""Evaluate one saved expression-prediction matrix."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows" / "expression_prediction"
if str(WORKFLOW) not in sys.path:
    sys.path.insert(0, str(WORKFLOW))

from evaluate_metrics import compute_metrics, load_array  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, help="Ground-truth .npy or .csv matrix [cells, genes].")
    parser.add_argument("--prediction", required=True, help="Prediction .npy or .csv matrix [cells, genes].")
    parser.add_argument("--output", default="outputs/expression_metrics.json")
    args = parser.parse_args()
    result = compute_metrics(load_array(Path(args.truth)), load_array(Path(args.prediction)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

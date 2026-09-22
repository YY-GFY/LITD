#!/usr/bin/env python3
"""Acceptance checks for the hierarchical latent-representation workflow."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/latent_interpretability.json")
    args = parser.parse_args()
    from interpretability_common import load_config, output_paths

    cfg = load_config(args.config)
    paths = output_paths(cfg)
    required = [
        paths["export"] / "cell_metadata.parquet",
        paths["export"] / "cell_embeddings.npz",
        paths["export"] / "token_means.npz",
        paths["metrics"] / "cell_representation_metrics.csv",
        paths["coordinates"] / "cell_umap_coordinates_full_24425.csv",
    ]
    errors = [f"missing output: {path}" for path in required if not path.is_file()]
    checks: dict[str, bool] = {"required_outputs_present": not errors}
    if not errors:
        meta = pd.read_parquet(required[0])
        arrays = np.load(required[1], allow_pickle=False)
        metrics = pd.read_csv(required[3])
        coords = pd.read_csv(required[4])
        checks.update({
            "unique_source_cells": not meta["source_index"].astype(str).duplicated().any(),
            "source_cell_count": len(meta) == 24425,
            "coordinate_cell_count": coords["source_index"].astype(str).nunique() == 24425,
            "finite_embeddings": all(
                np.isfinite(arrays[key]).all()
                for key in arrays.files
                if arrays[key].dtype.kind in "fc"
            ),
            "finite_metrics": np.isfinite(
                metrics.select_dtypes(include=[np.number]).to_numpy()
            ).all(),
        })
        if not all(checks.values()):
            errors.append("one or more latent-output acceptance checks failed")
    report = {"status": "ok" if not errors else "error", "checks": checks, "errors": errors}
    out = paths["analysis"] / "validation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

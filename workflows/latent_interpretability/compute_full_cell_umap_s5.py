#!/usr/bin/env python3
"""Compute full-cell UMAP coordinates for Supplementary Figure S5.

The script reads the completed interpretability export and fits an independent
UMAP for each requested cell-level representation using all unique source cells.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import umap


LAYERS = ["raw_gene", "projected_gene", "gene_cross", "gene_final"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = pd.read_parquet(args.export_dir / "cell_metadata.parquet")
    arrays = np.load(args.export_dir / "cell_embeddings.npz", allow_pickle=False)
    if meta["source_index"].astype(str).duplicated().any():
        raise ValueError("source_index is not unique")
    if len(meta) != 24425:
        raise ValueError(f"Expected 24,425 source cells, found {len(meta):,}")

    frames = []
    for offset, layer in enumerate(LAYERS):
        key = f"real__{layer}"
        if key not in arrays.files:
            raise KeyError(f"Missing array: {key}")
        values = arrays[key]
        if values.shape[0] != len(meta) or not np.isfinite(values).all():
            raise ValueError(f"Invalid values for {key}: {values.shape}")
        print(f"[full-umap] fitting {layer}: {values.shape}", flush=True)
        xy = umap.UMAP(
            n_components=2,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric="cosine",
            random_state=args.seed + offset,
            init="spectral",
            low_memory=True,
        ).fit_transform(values)
        frame = meta.copy()
        frame["layer"] = layer
        frame["UMAP1"] = xy[:, 0]
        frame["UMAP2"] = xy[:, 1]
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    if len(result) != 24425 * len(LAYERS):
        raise RuntimeError(f"Unexpected output rows: {len(result):,}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"[full-umap] wrote {args.output}: {len(result):,} rows", flush=True)


if __name__ == "__main__":
    main()

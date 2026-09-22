#!/usr/bin/env python3
"""Shared metric helpers for expression prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def _safe_mean_corr(x: np.ndarray, y: np.ndarray, rank: bool = False) -> float:
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    if rank:
        x = np.apply_along_axis(rankdata, 0, x)
        y = np.apply_along_axis(rankdata, 0, y)
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)
    denom = np.sqrt((x_centered**2).sum(axis=0) * (y_centered**2).sum(axis=0))
    valid = denom > 1e-12
    if not np.any(valid):
        return 0.0
    corr = np.zeros_like(denom, dtype=np.float64)
    corr[valid] = (x_centered[:, valid] * y_centered[:, valid]).sum(axis=0) / denom[valid]
    return float(np.mean(corr[valid]))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    residual = y_true - y_pred
    mse = float(np.mean(residual**2))

    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    flat_true_centered = y_true_flat - y_true_flat.mean()
    flat_pred_centered = y_pred_flat - y_pred_flat.mean()
    flat_denom = np.sqrt((flat_true_centered**2).sum() * (flat_pred_centered**2).sum())
    pearson = float((flat_true_centered * flat_pred_centered).sum() / flat_denom) if flat_denom > 1e-12 else 0.0

    true_rank = rankdata(y_true_flat)
    pred_rank = rankdata(y_pred_flat)
    true_rank_centered = true_rank - true_rank.mean()
    pred_rank_centered = pred_rank - pred_rank.mean()
    rank_denom = np.sqrt((true_rank_centered**2).sum() * (pred_rank_centered**2).sum())
    spearman = float((true_rank_centered * pred_rank_centered).sum() / rank_denom) if rank_denom > 1e-12 else 0.0

    ss_res = float((residual**2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return {
        "mse": mse,
        "pearson": pearson,
        "spearman": spearman,
        "r2": r2,
        "genewise_pearson": _safe_mean_corr(y_true, y_pred, rank=False),
        "genewise_spearman": _safe_mean_corr(y_true, y_pred, rank=True),
        "cellwise_pearson": _safe_mean_corr(y_true.T, y_pred.T, rank=False),
        "cellwise_spearman": _safe_mean_corr(y_true.T, y_pred.T, rank=True),
        "n_samples": int(y_true.shape[0]),
        "n_genes": int(y_true.shape[1]),
    }


def load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".csv":
        return np.loadtxt(path, delimiter=",")
    raise ValueError(f"Unsupported array format: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate expression prediction metrics.")
    parser.add_argument("--true", required=True, help="Path to ground-truth array (.npy or .csv).")
    parser.add_argument("--pred", required=True, help="Path to prediction array (.npy or .csv).")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    args = parser.parse_args()

    y_true = load_array(Path(args.true))
    y_pred = load_array(Path(args.pred))
    metrics = compute_metrics(y_true, y_pred)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

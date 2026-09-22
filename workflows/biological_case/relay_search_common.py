#!/usr/bin/env python3
"""Numerical Top-K helpers shared by the reported distinct-TF relay search."""
from __future__ import annotations

from typing import Any

import numpy as np


def positive_log(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, -np.inf, dtype=np.float64)
    mask = values > 0.0
    out[mask] = np.log(values[mask])
    return out


def topk_flat(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    flat = matrix.reshape(-1)
    finite = np.flatnonzero(np.isfinite(flat))
    if len(finite) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    if len(finite) <= k:
        chosen = finite
    else:
        chosen = finite[np.argpartition(flat[finite], -k)[-k:]]
    chosen = chosen[np.argsort(-flat[chosen], kind="stable")]
    return chosen.astype(np.int64, copy=False), flat[chosen]


def retain_global_base_topk(
    current: list[dict[str, Any]], additions: list[dict[str, Any]], k: int,
) -> list[dict[str, Any]]:
    pooled = current + additions
    if len(pooled) <= k:
        return sorted(pooled, key=lambda item: float(item["best_log_score"]), reverse=True)
    scores = np.fromiter((float(item["best_log_score"]) for item in pooled), dtype=np.float64)
    selected = np.argpartition(scores, -k)[-k:]
    return sorted(
        (pooled[int(index)] for index in selected),
        key=lambda item: float(item["best_log_score"]), reverse=True,
    )

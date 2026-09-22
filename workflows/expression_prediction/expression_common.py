#!/usr/bin/env python3
"""Shared paths, splits and output contracts for expression prediction."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    base_name = payload.pop("base_config", None)
    if base_name:
        base = json.loads((config_path.parent / base_name).read_text(encoding="utf-8"))
        base.update(payload)
        payload = base
    payload["config_path"] = str(config_path)
    root = Path(payload["root"])
    payload["root"] = str(root if root.is_absolute() else ROOT / root)
    snapshot = Path(payload["paper_snapshot"])
    payload["paper_snapshot"] = str(snapshot if snapshot.is_absolute() else ROOT / snapshot)
    prescient = payload.get("prescient", {})
    for key in ("source_dir", "runner"):
        if key in prescient:
            value = Path(prescient[key])
            prescient[key] = str(value if value.is_absolute() else ROOT / value)
    return payload


def get_benchmark():
    from litd import benchmark
    return benchmark


def benchmark_paths(cfg: dict[str, Any]):
    benchmark = get_benchmark()
    values = {}
    for key, value in cfg["paths"].items():
        path = Path(value)
        values[key] = str(path if path.is_absolute() else ROOT / path)
    # Expression prediction uses one prepared dataset tree shared by all methods.
    values["workdir"] = str(Path(cfg["root"]) / "shared")
    return benchmark.Paths(**values)


def split_paths(cfg: dict[str, Any], dataset: str) -> tuple[Path, Path]:
    split_dir = Path(cfg["root"]) / "shared" / "splits"
    return split_dir / f"{dataset}_train_indices.npy", split_dir / f"{dataset}_test_indices.npy"


def create_split(n_items: int, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    order = torch.randperm(n_items, generator=torch.Generator().manual_seed(seed)).numpy()
    n_train = int(fraction * n_items)
    return order[:n_train], order[n_train:]


def load_split(cfg: dict[str, Any], dataset: str, n_items: int) -> tuple[np.ndarray, np.ndarray]:
    train_path, test_path = split_paths(cfg, dataset)
    train_idx = np.load(train_path)
    test_idx = np.load(test_path)
    combined = np.concatenate([train_idx, test_idx])
    if (
        len(combined) != n_items
        or len(np.unique(combined)) != n_items
        or not np.array_equal(np.sort(combined), np.arange(n_items))
    ):
        raise ValueError(f"Invalid shared split for {dataset}: expected a partition of 0..{n_items - 1}")
    return train_idx.astype(int), test_idx.astype(int)


def loaders(dataset, train_idx: np.ndarray, test_idx: np.ndarray, batch_size: int, workers: int):
    train = DataLoader(
        Subset(dataset, train_idx.tolist()), batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True,
    )
    test = DataLoader(
        Subset(dataset, test_idx.tolist()), batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
    )
    return train, test


def method_dir(cfg: dict[str, Any], method: str, dataset: str) -> Path:
    slug = method.lower().replace("foundation-", "foundation_").replace(" ", "_")
    return Path(cfg["root"]) / "results" / slug / dataset


def save_result(
    cfg: dict[str, Any], method: str, dataset: str, prediction: np.ndarray,
    target: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
    pairs: pd.DataFrame, metrics: dict[str, Any],
) -> None:
    root = method_dir(cfg, method, dataset)
    pred_dir, metric_dir = root / "predictions", root / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "test_predictions.npy", prediction)
    np.save(pred_dir / "test_targets.npy", target)
    np.save(pred_dir / "train_indices.npy", train_idx)
    np.save(pred_dir / "test_indices.npy", test_idx)
    pairs.iloc[test_idx].copy().to_csv(pred_dir / "test_pairs.csv", index=False)
    pd.DataFrame([metrics]).to_csv(metric_dir / "expression_metrics.csv", index=False)
    (metric_dir / "expression_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

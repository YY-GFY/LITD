from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else ROOT / path)


def benchmark_paths(config: dict):
    from litd.benchmark import Paths

    values = {key: resolve_path(value) for key, value in config["paths"].items()}
    return Paths(**values)


def run_args(config: dict, *, run_name: str | None = None, input_mode: str | None = None,
             lr_mode: str | None = None) -> argparse.Namespace:
    training = config["training"]
    model = config["model"]
    inference = config["inference"]
    return argparse.Namespace(
        run_name=run_name or config["run_name"],
        batch_size=training["batch_size"], epochs=training["epochs"],
        lr=training["learning_rate"], weight_decay=training["weight_decay"],
        entropy_lambda=training["entropy_lambda"], hidden_dim=model["hidden_dim"],
        n_heads=model["n_heads"], tf_depth=model["tf_depth"],
        gene_depth=model["gene_depth"], mlp_ratio=model["mlp_ratio"],
        dropout=model["dropout"], input_mode=input_mode or model["input_mode"],
        lr_mode=lr_mode or model["lr_mode"], device=config["device"],
        seed=config["seed"], inf_batch_size=inference["batch_size"],
        inf_n_samples=inference["n_samples"],
        edge_threshold=inference["edge_threshold_top1000"],
    )

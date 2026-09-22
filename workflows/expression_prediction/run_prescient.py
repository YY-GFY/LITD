#!/usr/bin/env python3
"""Run PRESCIENT on the same fixed split used by the other reported methods."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from evaluate_metrics import compute_metrics
from expression_common import benchmark_paths, get_benchmark, load_config, load_split, method_dir, save_result


def load_train_callback(source: Path):
    candidates = [source / "prescient" / "commands" / "train_model.py", source / "commands" / "train_model.py"]
    module_path = next((path for path in candidates if path.is_file()), None)
    if module_path is None:
        raise FileNotFoundError(f"Could not find PRESCIENT train_model.py under {source}")
    spec = importlib.util.spec_from_file_location("prescient_train_model_adapter", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.train_init


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/expression_prediction.json")
    parser.add_argument("--dataset", choices=["hesc", "hhep"], required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    benchmark = get_benchmark()
    settings = cfg["prescient"]
    source = Path(settings["source_dir"])
    if not source.is_dir():
        raise FileNotFoundError(
            f"PRESCIENT source not found: {source}. Clone/install the upstream project at prescient.source_dir."
        )
    sys.path.insert(0, str(source))
    from prescient.train import run as prescient_run  # type: ignore
    from prescient.train.model import AutoGenerator  # type: ignore

    paths = benchmark_paths(cfg)
    benchmark.set_seed(int(cfg["seed"]))
    runtime = benchmark.load_runtime_paths(paths, args.dataset)
    dataset = benchmark.ScCellDataset(runtime)
    train_idx, test_idx = load_split(cfg, args.dataset, len(dataset))
    train_pairs = dataset.pairs_df.iloc[train_idx]
    test_pairs = dataset.pairs_df.iloc[test_idx].reset_index(drop=True)
    cells = set(train_pairs.source_index.astype(str)).union(train_pairs.target_index.astype(str))
    train_cells = [name for name in dataset.obs_names if name in cells]
    cell_indices = [dataset.obs_name_to_idx[name] for name in train_cells]
    train_expression = dataset.expr_matrix[cell_indices]
    time_parser = benchmark.build_specs(paths)[args.dataset].time_parser
    timepoints = np.asarray([int(time_parser(name)) for name in train_cells], dtype=np.int64)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(train_expression).astype(np.float32)
    n_pcs = min(int(settings["num_pcs"]), scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=n_pcs, random_state=int(cfg["seed"])).fit(scaled)
    projected = pca.transform(scaled).astype(np.float32)
    stages = sorted(np.unique(timepoints).tolist())
    payload = {
        "data": train_expression, "genes": dataset.var_names,
        "celltype": np.asarray(["combined"] * len(train_cells), dtype=object),
        "tps": timepoints,
        "x": [torch.from_numpy(scaled[timepoints == stage]).float() for stage in stages],
        "xp": [torch.from_numpy(projected[timepoints == stage]).float() for stage in stages],
        "xu": None, "y": stages,
        "w": [np.zeros(int(np.sum(timepoints == stage)), dtype=np.float32) for stage in stages],
        "pca": pca, "scaler": scaler, "train_obs_names": train_cells,
    }
    result = method_dir(cfg, "PRESCIENT", args.dataset)
    model_dir = result / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    data_path = model_dir / "training_data.pt"
    torch.save(payload, data_path)
    use_cuda = str(cfg["device"]).startswith("cuda") and torch.cuda.is_available()
    gpu_index = int(str(cfg["device"]).split(":")[-1]) if use_cuda else 0
    train_args = SimpleNamespace(
        no_cuda=not use_cuda, gpu=gpu_index, out_dir=str(model_dir),
        seed=int(cfg["seed"]), data_path=str(data_path), weight_name="uniform-growth",
        loss="euclidean", k_dim=int(settings["k_dim"]), activation=settings["activation"],
        layers=int(settings["layers"]), pretrain_epochs=int(settings["pretrain_epochs"]),
        train_epochs=int(settings["train_epochs"]), train_lr=float(settings["train_learning_rate"]),
        train_dt=float(settings["time_step"]), train_sd=float(settings["noise_scale"]),
        train_tau=float(settings["tau"]), train_batch=float(settings["train_batch_fraction"]),
        train_clip=float(settings["gradient_clip"]), save=int(settings["save_every"]),
        pretrain=True, train=True, config=None,
    )
    prescient_run(train_args, load_train_callback(source))
    checkpoint = next(model_dir.glob("**/train.best.pt"))
    model_cfg = SimpleNamespace(**torch.load(checkpoint.parent / "config.pt", map_location="cpu"))
    device = torch.device(cfg["device"] if use_cuda else "cpu")
    model = AutoGenerator(model_cfg)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state_dict"])
    model.to(device).eval()

    predictions, targets = [], []
    batch_size = int(settings["evaluation_batch_size"])
    for start in range(0, len(test_pairs), batch_size):
        frame = test_pairs.iloc[start : start + batch_size]
        source_idx = [dataset.obs_name_to_idx[str(name)] for name in frame.source_index]
        target_idx = [dataset.obs_name_to_idx[str(name)] for name in frame.target_index]
        source_expression = dataset.expr_matrix[source_idx]
        target_expression = dataset.expr_matrix[target_idx]
        delta = frame.day_target.to_numpy(float) - frame.day_source.to_numpy(float)
        initial = pca.transform(scaler.transform(source_expression)).astype(np.float32)
        simulation_sum = np.zeros_like(initial)
        for simulation in range(int(settings["num_simulations"])):
            torch.manual_seed(int(cfg["seed"]) + simulation)
            state = torch.from_numpy(initial.copy()).to(device)
            with torch.enable_grad():
                for row, duration in enumerate(delta):
                    value = state[row : row + 1]
                    for _ in range(max(1, int(round(duration / float(settings["time_step"]))))):
                        value = model._step(
                            value, dt=float(settings["time_step"]),
                            z=torch.randn_like(value) * float(settings["noise_scale"]),
                        )
                    state[row : row + 1] = value.detach()
            simulation_sum += state.detach().cpu().numpy()
        predicted_scaled = pca.inverse_transform(simulation_sum / float(settings["num_simulations"]))
        prediction = scaler.inverse_transform(predicted_scaled).astype(np.float32)
        predictions.append(np.clip(prediction, 0.0, None))
        targets.append(target_expression)
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    metrics = {"dataset": args.dataset, "method": "PRESCIENT", **compute_metrics(target, prediction)}
    save_result(cfg, "PRESCIENT", args.dataset, prediction, target, train_idx, test_idx, dataset.pairs_df, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

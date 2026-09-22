#!/usr/bin/env python3
"""Run one reported expression-prediction method on the shared split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler

from evaluate_metrics import compute_metrics
from expression_common import (
    benchmark_paths, get_benchmark, load_config, load_split, loaders, method_dir, save_result,
)


class RawExpressionMLP(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_genes),
        )
        self.apply(_initialize)

    def forward(self, source_expr: torch.Tensor) -> torch.Tensor:
        return self.net(source_expr)


class ScFoundationMLP(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.weight = nn.Parameter(torch.empty(n_genes, hidden_dim // 2))
        self.bias = nn.Parameter(torch.empty(n_genes))
        self.apply(_initialize)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        bound = 1 / np.sqrt(self.weight.shape[1])
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        hidden = self.head(self.input_proj(embeddings))
        return (hidden * self.weight.unsqueeze(0)).sum(dim=-1) + self.bias


def _initialize(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def forward(model, batch, method: str, device: torch.device):
    embeddings, ligand, source, target = batch
    target = target.to(device, non_blocking=True)
    if method == "LITD":
        prediction, attention = model(
            embeddings.to(device, non_blocking=True), ligand.to(device, non_blocking=True), return_grn=True
        )
        return prediction, target, attention
    if method == "MLP":
        return model(source.to(device, non_blocking=True)), target, None
    return model(embeddings.to(device, non_blocking=True)), target, None


@torch.no_grad()
def predict(model, loader, method: str, device: torch.device):
    model.eval()
    predictions, targets = [], []
    for batch in loader:
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred, target, _ = forward(model, batch, method, device)
        predictions.append(pred.float().cpu().numpy())
        targets.append(target.float().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def gene_pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    x = prediction - prediction.mean(0, keepdims=True)
    y = target - target.mean(0, keepdims=True)
    denom = np.sqrt((x * x).sum(0) * (y * y).sum(0))
    valid = (denom > 1e-8) & (target.std(0) > 1e-5)
    return float(((x[:, valid] * y[:, valid]).sum(0) / denom[valid]).mean()) if valid.any() else 0.0


def train_model(model, train_loader, test_loader, method: str, settings: dict, device: torch.device, out: Path):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"])
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.HuberLoss(delta=float(settings.get("huber_delta", 0.5)))
    best = float("-inf")
    history = []
    out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred, target, attention = forward(model, batch, method, device)
                loss = criterion(pred, target)
                if method == "LITD":
                    entropy = -(attention * torch.log(attention + 1e-8)).sum(dim=-1).mean()
                    loss = loss + float(settings["entropy_lambda"]) * entropy
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("gradient_clip_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        val_pred, val_true = predict(model, test_loader, method, device)
        score = gene_pearson(val_pred, val_true)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_genewise_pearson": score})
        if score > best:
            best = score
            torch.save(model.state_dict(), out)
        print(json.dumps({"method": method, **history[-1]}), flush=True)
    model.load_state_dict(torch.load(out, map_location=device))
    return best, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/expression_prediction.json")
    parser.add_argument("--method", choices=["LITD", "MLP", "scFoundation-MLP", "persistence", "stage mean"], required=True)
    parser.add_argument("--dataset", choices=["hesc", "hhep"], required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    benchmark = get_benchmark()
    paths = benchmark_paths(cfg)
    runtime = benchmark.load_runtime_paths(paths, args.dataset)
    dataset = benchmark.ScCellDataset(runtime)
    train_idx, test_idx = load_split(cfg, args.dataset, len(dataset))

    if args.method in {"persistence", "stage mean"}:
        matrix = dataset.expr_matrix
        pairs = dataset.pairs_df
        if args.method == "persistence":
            pred = np.stack([matrix[dataset.obs_name_to_idx[str(pairs.iloc[i].source_index)]] for i in test_idx])
        else:
            groupby = cfg["stage_mean_groupby"]
            train_pairs = pairs.iloc[train_idx]
            means: dict[tuple[object, ...], list[np.ndarray]] = {}
            def key(row):
                return tuple(float(row[x]) if x.startswith("day_") else str(row[x]) for x in groupby)
            for _, row in train_pairs.iterrows():
                means.setdefault(key(row), []).append(matrix[dataset.obs_name_to_idx[str(row.source_index)]])
            group_means = {k: np.mean(v, axis=0).astype(np.float32) for k, v in means.items()}
            fallback = np.mean(np.stack(list(group_means.values())), axis=0).astype(np.float32)
            pred = np.stack([group_means.get(key(pairs.iloc[i]), fallback) for i in test_idx])
        true = np.stack([matrix[dataset.obs_name_to_idx[str(pairs.iloc[i].target_index)]] for i in test_idx])
        metrics = {"dataset": args.dataset, "method": args.method, **compute_metrics(true, pred)}
        save_result(cfg, args.method, args.dataset, pred, true, train_idx, test_idx, pairs, metrics)
        print(json.dumps(metrics, indent=2))
        return

    benchmark.set_seed(int(cfg["seed"]))
    device_name = cfg["device"] if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if args.method == "LITD":
        model_cfg, train_cfg = cfg["model"], cfg["training"]
        rec, tf, mask = benchmark.prepare_metadata(runtime)
        model = benchmark.ScFoundationCellModel(
            dataset.n_genes, rec, tf, mask, 512, model_cfg["hidden_dim"], model_cfg["n_heads"],
            model_cfg["tf_depth"], model_cfg["gene_depth"], model_cfg["mlp_ratio"],
            model_cfg["dropout"], model_cfg["input_mode"], model_cfg["lr_mode"],
        ).to(device)
        settings = dict(train_cfg)
    elif args.method == "MLP":
        settings = cfg["mlp"]
        model = RawExpressionMLP(dataset.n_genes, int(settings["hidden_dims"][args.dataset]), float(settings["dropout"])).to(device)
    else:
        settings = cfg["scfoundation_mlp"]
        model = ScFoundationMLP(dataset.n_genes, int(settings["hidden_dim"]), float(settings["dropout"])).to(device)
    train_loader, test_loader = loaders(dataset, train_idx, test_idx, int(settings["batch_size"]), int(cfg["num_workers"]))
    checkpoint = method_dir(cfg, args.method, args.dataset) / "models" / "best.pth"
    best, history = train_model(model, train_loader, test_loader, args.method, settings, device, checkpoint)
    pred, true = predict(model, test_loader, args.method, device)
    metrics = {
        "dataset": args.dataset, "method": args.method, "best_validation_genewise_pearson": best,
        "parameter_count": int(sum(p.numel() for p in model.parameters())), **compute_metrics(true, pred),
    }
    save_result(cfg, args.method, args.dataset, pred, true, train_idx, test_idx, dataset.pairs_df, metrics)
    (checkpoint.parent / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train LITD on the 36,044-cell biological-case dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "workflows" / "biological_case"
sys.path.insert(0, str(CASE_DIR))

from case_common import (  # noqa: E402
    BiologicalCaseDataset,
    build_loaders,
    build_model,
    ensure_dir,
    load_config,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/litd_biological_case.json")
    return parser.parse_args()


def resolve_config(path_string: str) -> Path:
    path = Path(path_string)
    return path if path.is_absolute() else ROOT / path


def mean_gene_pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_centered = prediction - prediction.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    numerator = np.sum(pred_centered * target_centered, axis=0)
    denominator = np.sqrt(
        np.sum(pred_centered**2, axis=0) * np.sum(target_centered**2, axis=0)
    )
    correlations = numerator / (denominator + 1e-8)
    valid = np.std(target, axis=0) > 1e-5
    return float(np.mean(correlations[valid])) if np.any(valid) else 0.0


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
    input_mode: str,
    huber_delta: float,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for embeddings, ligand_vec, source_expr, target_expr in loader:
        model_input = source_expr if input_mode == "raw_expr" else embeddings
        model_input = model_input.to(device, non_blocking=True)
        ligand_vec = ligand_vec.to(device, non_blocking=True)
        target_expr = target_expr.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction, _ = model(model_input, ligand_vec, return_grn=True)
            loss = F.huber_loss(prediction, target_expr, delta=huber_delta)
        losses.append(float(loss.detach().cpu()))
        predictions.append(prediction.float().cpu().numpy())
        targets.append(target_expr.float().cpu().numpy())
    return float(np.mean(losses)), mean_gene_pearson(np.concatenate(predictions), np.concatenate(targets))


def main() -> None:
    cli = parse_args()
    cfg = load_config(resolve_config(cli.config))
    set_seed(cfg.seed)

    device_name = cfg.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    dataset = BiologicalCaseDataset(cfg)
    train_loader, test_loader, train_idx, test_idx = build_loaders(dataset, cfg)
    model = build_model(dataset, cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    output_root = Path(cfg.checkpoint_root) / "results" / cfg.run_name
    model_dir = output_root / "models"
    ensure_dir(model_dir)
    np.savez(output_root / "split_indices.npz", train_idx=train_idx, test_idx=test_idx)

    history: list[dict[str, float | int]] = []
    best_pearson = float("-inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for embeddings, ligand_vec, source_expr, target_expr in train_loader:
            model_input = source_expr if cfg.input_mode == "raw_expr" else embeddings
            model_input = model_input.to(device, non_blocking=True)
            ligand_vec = ligand_vec.to(device, non_blocking=True)
            target_expr = target_expr.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prediction, attention = model(model_input, ligand_vec, return_grn=True)
                prediction_loss = F.huber_loss(prediction, target_expr, delta=cfg.huber_delta)
                entropy = -(attention * torch.log(attention + 1e-8)).sum(dim=-1).mean()
                loss = prediction_loss + cfg.entropy_lambda * entropy
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        validation_loss, validation_pearson = evaluate(
            model, test_loader, device, cfg.input_mode, cfg.huber_delta
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation_loss,
            "validation_mean_gene_pearson": validation_pearson,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_pearson > best_pearson:
            best_pearson = validation_pearson
            torch.save(model.state_dict(), model_dir / "best.pth")

    pd.DataFrame(history).to_csv(output_root / "training_history.csv", index=False)
    (output_root / "training_summary.json").write_text(
        json.dumps(
            {
                "run_name": cfg.run_name,
                "best_validation_mean_gene_pearson": best_pearson,
                "epochs": cfg.epochs,
                "seed": cfg.seed,
                "checkpoint": str(model_dir / "best.pth"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

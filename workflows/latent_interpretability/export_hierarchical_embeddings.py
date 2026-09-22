#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from interpretability_common import (
    WINDOWS, InterpretabilityDataset, build_model, condition_ligands, ensure_output_dirs,
    git_revision, global_env_by_day, load_config, load_tables, output_paths, safe_key,
    select_pair_rows, sha256_file, validate_input_paths, write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def checkpoint_summary(cfg):
    state = torch.load(cfg["checkpoint_path"], map_location="cpu")
    required = {
        "input_proj.0.weight": (cfg["hidden_dim"], cfg["embed_dim"]),
        "gene_head_weight": (cfg["n_genes"], cfg["hidden_dim"] // 2),
    }
    errors = []
    for key, shape in required.items():
        actual = tuple(state[key].shape) if key in state else None
        if actual != tuple(shape):
            errors.append({"key": key, "expected": shape, "actual": actual})
    if errors:
        raise ValueError(f"Checkpoint shape validation failed: {errors}")
    return {"n_state_keys": len(state), "validated_shapes": required}


def pooled_hidden(hidden):
    result = {
        "raw_gene": hidden["raw_gene"].mean(dim=1),
        "projected_gene": hidden["projected_gene"].mean(dim=1),
        "rec_pre_film": hidden["rec_pre_film"].mean(dim=1),
        "rec_post_film": hidden["rec_post_film"].mean(dim=1),
        "tf_cross": hidden["tf_cross"].mean(dim=1),
        "tf_final": hidden["tf_final"].mean(dim=1),
        "gene_cross": hidden["gene_cross"].mean(dim=1),
        "gene_final": hidden["gene_final"].mean(dim=1),
    }
    for idx, tensor in enumerate(hidden["tf_layers"][:-1], start=1):
        result[f"tf_layer{idx}"] = tensor.mean(dim=1)
    for idx, tensor in enumerate(hidden["gene_layers"][:-1], start=1):
        result[f"gene_layer{idx}"] = tensor.mean(dim=1)
    return result


def add_sum(store, key, tensor):
    value = tensor.detach().float().sum(dim=0).cpu().numpy().astype(np.float64, copy=False)
    store[key] = value if key not in store else store[key] + value


def main():
    args = parse_args()
    cfg = load_config(args.config)
    validate_input_paths(cfg)
    ligand_df, receptor_df, tf_df, _ = load_tables(cfg)
    pair_rows = select_pair_rows(cfg)
    dry = {
        "status": "ok",
        "config": cfg["config_path"],
        "analysis_name": cfg["analysis_name"],
        "selected_pairs": int(len(pair_rows)),
        "conditions": cfg["conditions"],
        "n_ligands": int(len(ligand_df)),
        "n_receptors": int(len(receptor_df)),
        "n_tfs": int(len(tf_df)),
        "checkpoint": checkpoint_summary(cfg),
    }
    if args.dry_run:
        print(json.dumps(dry, indent=2, default=str))
        return

    paths = ensure_output_dirs(cfg)
    shutil.copy2(cfg["config_path"], paths["analysis"] / "config_snapshot.json")
    device_name = cfg["device"]
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device_name}")
    device = torch.device(device_name)
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    dataset = InterpretabilityDataset(cfg, pair_rows)
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=False,
                        num_workers=int(cfg["num_workers"]), pin_memory=device.type == "cuda")
    model, ligand_df, receptor_df, tf_df = build_model(cfg, device)
    global_env = global_env_by_day(dataset.env_dict)

    cell_store = defaultdict(list)
    token_sums = {}
    token_counts = defaultdict(int)
    expr_sums = {}
    metadata_rows = []

    for batch_index, batch in enumerate(loader):
        embedding = batch["embedding"].to(device, non_blocking=True)
        real_ligand = batch["ligand"]
        conditioned = condition_ligands(real_ligand, cfg["conditions"], list(batch["experiment"]),
                                        batch["day_source"], dataset.env_dict, global_env)
        if not metadata_rows or len(metadata_rows) < len(dataset):
            for i in range(len(batch["source_index"])):
                metadata_rows.append({
                    "row_order": len(metadata_rows),
                    "pair_index": int(batch["pair_index"][i]),
                    "source_index": batch["source_index"][i],
                    "experiment": batch["experiment"][i],
                    "cell_line": batch["cell_line"][i],
                    "day_source": float(batch["day_source"][i]),
                    "day_target": float(batch["day_target"][i]),
                    "window": batch["window"][i],
                })

        for condition in cfg["conditions"]:
            ligand = conditioned[condition].to(device, non_blocking=True)
            with torch.inference_mode():
                _prediction, hidden = model(embedding, ligand, return_hidden=True)
            for layer, values in pooled_hidden(hidden).items():
                cell_store[safe_key(condition, layer)].append(values.detach().float().cpu().numpy())

            windows = list(batch["window"])
            for window in sorted(set(windows)):
                mask_np = np.asarray([item == window for item in windows])
                mask = torch.as_tensor(mask_np, device=device)
                count = int(mask.sum().item())
                token_counts[safe_key(condition, window)] += count
                add_sum(token_sums, safe_key(condition, window, "rec_pre_film"), hidden["rec_pre_film"][mask])
                add_sum(token_sums, safe_key(condition, window, "rec_post_film"), hidden["rec_post_film"][mask])
                add_sum(token_sums, safe_key(condition, window, "tf_final"), hidden["tf_final"][mask])
                add_sum(token_sums, safe_key(condition, window, "gene_final"), hidden["gene_final"][mask])
                if condition == "real":
                    expr = batch["source_expr"][mask_np].double().sum(dim=0).numpy()
                    expr_sums[window] = expr if window not in expr_sums else expr_sums[window] + expr
            del hidden

        if (batch_index + 1) % 25 == 0:
            print(f"[export] batches={batch_index + 1}/{len(loader)} cells={min((batch_index + 1) * cfg['batch_size'], len(dataset))}", flush=True)

    cell_arrays = {key: np.concatenate(chunks, axis=0).astype(np.float32) for key, chunks in cell_store.items()}
    cell_arrays["selected_pair_rows"] = pair_rows.astype(np.int64)
    np.savez_compressed(paths["export"] / "cell_embeddings.npz", **cell_arrays)
    pd.DataFrame(metadata_rows).to_parquet(paths["export"] / "cell_metadata.parquet", index=False)

    token_arrays = {}
    for key, total in token_sums.items():
        condition, window, _layer = key.split("__", 2)
        token_arrays[key] = (total / token_counts[safe_key(condition, window)]).astype(np.float32)
    for window, total in expr_sums.items():
        count = token_counts[safe_key("real", window)]
        token_arrays[safe_key("real", window, "source_expression_mean")] = (total / count).astype(np.float32)
    token_arrays["receptor_names"] = np.asarray(receptor_df["gene_name"].astype(str).tolist(), dtype=str)
    token_arrays["tf_names"] = np.asarray(tf_df["gene_name"].astype(str).tolist(), dtype=str)
    token_arrays["gene_names"] = np.asarray(dataset.var_names, dtype=str)
    np.savez_compressed(paths["export"] / "token_means.npz", **token_arrays)

    count_table = pd.DataFrame([
        {"condition": key.split("__")[0], "window": key.split("__")[1], "n_cells": value}
        for key, value in sorted(token_counts.items())
    ])
    count_table.to_csv(paths["export"] / "condition_window_counts.csv", index=False)
    metadata = {
        **dry,
        "device": str(device),
        "checkpoint_sha256": sha256_file(cfg["checkpoint_path"]),
        "scaffold_git_revision": git_revision(cfg["root"]),
        "script_sha256": {
            path.name: sha256_file(path)
            for path in sorted((Path(cfg["root"]) / "scripts").glob("*"))
            if path.is_file() and path.suffix in {".py", ".sh"}
        },
        "cell_embedding_keys": sorted(cell_arrays),
        "token_mean_keys": sorted(token_arrays),
        "interpretation": "Learned representation organization and model-internal evidence; not causal proof.",
    }
    write_json(paths["export"] / "export_metadata.json", metadata)
    print(f"[export] completed: {paths['export']}")


if __name__ == "__main__":
    main()

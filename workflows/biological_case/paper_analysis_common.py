#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from case_common import (
    BiologicalCaseDataset,
    ExperimentConfig,
    build_model,
    ensure_dir,
    load_config,
    load_metadata_frames,
    prepare_metadata,
)


@dataclass(frozen=True)
class WindowSpec:
    day_source: float
    day_target: float
    label: str


WINDOWS = (
    WindowSpec(0.0, 1.0, "day0_to_day1"),
    WindowSpec(1.0, 2.0, "day1_to_day2"),
    WindowSpec(2.0, 3.0, "day2_to_day3"),
)
WINDOW_BY_LABEL = {window.label: window for window in WINDOWS}
MISMATCH_DAY_SOURCE = {
    0.0: 2.0,
    1.0: 0.0,
    2.0: 1.0,
}


class WindowConditionDataset(Dataset):
    def __init__(
        self,
        base_dataset: BiologicalCaseDataset,
        indices: list[int],
        condition_name: str,
        ligand_index_map: dict[str, int],
        global_env_by_day: dict[float, torch.Tensor],
        zero_ligands: list[str] | None = None,
        mask_ligands: list[str] | None = None,
        stage_mismatch: bool = False,
    ):
        self.base_dataset = base_dataset
        self.indices = [int(i) for i in indices]
        self.condition_name = condition_name
        self.ligand_index_map = ligand_index_map
        self.global_env_by_day = global_env_by_day
        self.zero_ligands = list(zero_ligands or [])
        self.mask_ligands = list(mask_ligands or [])
        self.stage_mismatch = stage_mismatch
        meta = base_dataset.pairs_df.iloc[self.indices][["experiment", "day_source", "day_target"]].copy()
        self.experiments = meta["experiment"].astype(str).tolist()
        self.day_sources = meta["day_source"].astype(float).tolist()
        self.day_targets = meta["day_target"].astype(float).tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def _build_mismatched_ligand(self, experiment: str, day_source: float, current_ligand: torch.Tensor) -> torch.Tensor:
        wrong_day = MISMATCH_DAY_SOURCE.get(float(day_source))
        if wrong_day is None:
            return current_ligand
        replacement = self.base_dataset.env_dict.get((experiment, float(wrong_day)))
        if replacement is None:
            replacement = self.global_env_by_day.get(float(wrong_day))
        if replacement is None:
            return current_ligand
        if not isinstance(replacement, torch.Tensor):
            replacement = torch.tensor(replacement)
        return replacement.detach().clone().float()

    def _apply_condition(self, ligand_vec: torch.Tensor, experiment: str, day_source: float) -> torch.Tensor:
        conditioned = ligand_vec.detach().clone().float()
        if self.stage_mismatch:
            conditioned = self._build_mismatched_ligand(experiment, day_source, conditioned)
        for gene in self.zero_ligands + self.mask_ligands:
            idx = self.ligand_index_map.get(gene)
            if idx is not None:
                conditioned[idx] = 0.0
        return conditioned

    def __getitem__(self, position: int):
        base_idx = self.indices[position]
        emb_t, ligand_vec, src_expr, tgt_expr = self.base_dataset[base_idx]
        experiment = self.experiments[position]
        day_source = self.day_sources[position]
        day_target = self.day_targets[position]
        conditioned_ligand = self._apply_condition(ligand_vec, experiment, day_source)
        return emb_t, conditioned_ligand, src_expr, tgt_expr, experiment, day_source, day_target


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def load_checkpointed_model(
    cfg: ExperimentConfig,
    device_override: str | None = None,
) -> tuple[BiologicalCaseDataset, torch.nn.Module, torch.device]:
    device_str = device_override or cfg.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    dataset = BiologicalCaseDataset(cfg)
    model = build_model(dataset, cfg, device)
    model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device))
    model.eval()
    return dataset, model, device


def build_positive_lr_table(model, cfg: ExperimentConfig) -> pd.DataFrame:
    ligand_df, receptor_df, _tf_df, _lr_pairs_df = load_metadata_frames(cfg)
    _rec_indices, _tf_indices, lr_mask = prepare_metadata(cfg)
    weights = model.state_dict()["lr_layer.weights"].detach().cpu().numpy()
    masked = weights * lr_mask.detach().cpu().numpy()
    positive = np.clip(masked, a_min=0.0, a_max=None)
    return pd.DataFrame(
        positive,
        index=ligand_df["gene_name"].astype(str).tolist(),
        columns=receptor_df["gene_name"].astype(str).tolist(),
    )


def build_global_env_by_day(env_dict: dict[Any, Any]) -> dict[float, torch.Tensor]:
    grouped: dict[float, list[torch.Tensor]] = defaultdict(list)
    for key, value in env_dict.items():
        _experiment, day = key
        tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
        grouped[float(day)].append(tensor.detach().clone().float())
    return {day: torch.stack(vectors, dim=0).mean(dim=0) for day, vectors in grouped.items()}


def source_window_root(cfg: ExperimentConfig) -> Path:
    return Path(cfg.source_window_artifact_root)


def load_window_bundle(cfg: ExperimentConfig, window_label: str, condition: str = "real") -> dict[str, Any]:
    root = source_window_root(cfg) / window_label / condition
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    ligand_context = pd.read_csv(root / "ligand_context_mean.csv").set_index("gene_name")["mean_expr"]
    receptor_to_tf = pd.read_csv(root / "receptor_to_tf_attention.csv").set_index("receptor")
    tf_to_gene = pd.read_csv(root / "tf_to_gene_attention.csv").set_index("gene_name")
    predicted_expression = pd.read_csv(root / "predicted_expression_mean.csv").set_index("gene_name")["pred_mean"]
    target_expression = pd.read_csv(root / "target_expression_mean.csv").set_index("gene_name")["target_mean"]
    return {
        "condition": condition,
        "window_label": window_label,
        "day_source": float(summary["day_source"]),
        "day_target": float(summary["day_target"]),
        "n_samples": int(summary["n_samples"]),
        "ligand_context": ligand_context,
        "receptor_to_tf": receptor_to_tf,
        "tf_to_gene": tf_to_gene,
        "predicted_expression": predicted_expression,
        "target_expression": target_expression,
    }


def load_real_window_bundles(cfg: ExperimentConfig) -> dict[str, dict[str, Any]]:
    return {window.label: load_window_bundle(cfg, window.label, condition="real") for window in WINDOWS}


def summarize_window_condition(
    dataset: BiologicalCaseDataset,
    model,
    cfg: ExperimentConfig,
    device: torch.device,
    window: WindowSpec,
    condition_name: str,
    ligand_index_map: dict[str, int],
    ligand_names: list[str],
    receptor_names: list[str],
    tf_names: list[str],
    gene_names: list[str],
    global_env_by_day: dict[float, torch.Tensor],
    batch_size: int,
    zero_ligands: list[str] | None = None,
    mask_ligands: list[str] | None = None,
    stage_mismatch: bool = False,
    max_samples_per_window: int = 0,
) -> dict[str, Any]:
    pair_df = dataset.pairs_df
    mask = (pair_df["day_source"].astype(float) == window.day_source) & (pair_df["day_target"].astype(float) == window.day_target)
    indices = pair_df.index[mask].tolist()
    if max_samples_per_window > 0:
        indices = indices[:max_samples_per_window]
    wrapper = WindowConditionDataset(
        dataset,
        indices,
        condition_name=condition_name,
        ligand_index_map=ligand_index_map,
        global_env_by_day=global_env_by_day,
        zero_ligands=zero_ligands,
        mask_ligands=mask_ligands,
        stage_mismatch=stage_mismatch,
    )
    loader = DataLoader(wrapper, batch_size=batch_size, shuffle=False)

    ligand_sum = np.zeros(len(ligand_names), dtype=np.float64)
    rt_sum = np.zeros((len(tf_names), len(receptor_names)), dtype=np.float64)
    tg_sum = np.zeros((len(gene_names), len(tf_names)), dtype=np.float64)
    pred_sum = np.zeros(len(gene_names), dtype=np.float64)
    target_sum = np.zeros(len(gene_names), dtype=np.float64)
    sample_count = 0

    with torch.no_grad():
        for emb_t, ligand_vec, src_expr, tgt_expr, _experiments, _day_source, _day_target in loader:
            emb_t = emb_t.to(device)
            ligand_vec = ligand_vec.to(device)
            src_expr = src_expr.to(device)
            tgt_expr = tgt_expr.to(device)
            model_input = emb_t if cfg.input_mode == "scfoundation" else src_expr
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=device.type == "cuda"):
                pred, artifacts = model(model_input, ligand_vec, return_artifacts=True)

            ligand_np = to_numpy(ligand_vec)
            rt_np = to_numpy(artifacts["rec_to_tf_attn"])
            tg_np = to_numpy(artifacts["tf_to_gene_attn"])
            pred_np = to_numpy(pred)
            target_np = to_numpy(tgt_expr)

            ligand_sum += ligand_np.sum(axis=0)
            rt_sum += rt_np.sum(axis=0)
            tg_sum += tg_np.sum(axis=0)
            pred_sum += pred_np.sum(axis=0)
            target_sum += target_np.sum(axis=0)
            sample_count += ligand_np.shape[0]

    if sample_count == 0:
        raise ValueError(f"No pairs found for window {window.label} condition {condition_name}")

    return {
        "condition": condition_name,
        "window_label": window.label,
        "day_source": window.day_source,
        "day_target": window.day_target,
        "n_samples": sample_count,
        "ligand_context": pd.Series(ligand_sum / sample_count, index=ligand_names, name="mean_expr"),
        "receptor_to_tf": pd.DataFrame((rt_sum / sample_count).T, index=receptor_names, columns=tf_names),
        "tf_to_gene": pd.DataFrame(tg_sum / sample_count, index=gene_names, columns=tf_names),
        "predicted_expression": pd.Series(pred_sum / sample_count, index=gene_names, name="pred_mean"),
        "target_expression": pd.Series(target_sum / sample_count, index=gene_names, name="target_mean"),
    }


def write_bundle(bundle: dict[str, Any], out_dir: Path) -> None:
    ensure_dir(out_dir)
    bundle["ligand_context"].rename("mean_expr").reset_index().rename(columns={"index": "gene_name"}).to_csv(
        out_dir / "ligand_context_mean.csv",
        index=False,
    )
    bundle["receptor_to_tf"].rename_axis("receptor").to_csv(out_dir / "receptor_to_tf_attention.csv")
    bundle["tf_to_gene"].rename_axis("gene_name").to_csv(out_dir / "tf_to_gene_attention.csv")
    bundle["predicted_expression"].rename("pred_mean").reset_index().rename(columns={"index": "gene_name"}).to_csv(
        out_dir / "predicted_expression_mean.csv",
        index=False,
    )
    bundle["target_expression"].rename("target_mean").reset_index().rename(columns={"index": "gene_name"}).to_csv(
        out_dir / "target_expression_mean.csv",
        index=False,
    )
    metadata = {
        "window": bundle["window_label"],
        "condition": bundle["condition"],
        "day_source": bundle["day_source"],
        "day_target": bundle["day_target"],
        "n_samples": bundle["n_samples"],
    }
    (out_dir / "summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

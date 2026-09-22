#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import Dataset

from litd_model_interpretable import ScFoundationCellModel


WINDOWS = ((0.0, 1.0, "day0_to_day1"), (1.0, 2.0, "day1_to_day2"), (2.0, 3.0, "day2_to_day3"))
WINDOW_LABEL = {(a, b): label for a, b, label in WINDOWS}
MISMATCH_DAY = {0.0: 2.0, 1.0: 0.0, 2.0: 1.0}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "base_config" in payload:
        base = json.loads((path.parent / payload["base_config"]).read_text(encoding="utf-8"))
        base.update({k: v for k, v in payload.items() if k != "base_config"})
        payload = base
    repo_root = path.parents[1]
    for key in ("root", "source_model_root", "checkpoint_path", "h5ad_file", "embed_bin",
                "pairs_csv", "env_dict", "ligand_list", "receptor_list", "tf_list", "lr_pairs"):
        if key not in payload:
            continue
        value = Path(payload[key])
        payload[key] = str(value if value.is_absolute() else repo_root / value)
    payload["config_path"] = str(path)
    return payload


def output_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    root = Path(cfg["root"])
    name = cfg["analysis_name"]
    return {
        "root": root,
        "analysis": root / "analysis" / name,
        "export": root / "analysis" / name / "export",
        "metrics": root / "analysis" / name / "metrics",
        "coordinates": root / "analysis" / name / "coordinates",
        "figures": root / "figures" / name,
        "logs": root / "logs" / name,
        "state": root / "run_state" / name,
    }


def ensure_output_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = output_paths(cfg)
    for key in ("analysis", "export", "metrics", "coordinates", "figures", "logs", "state"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: str | Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def validate_input_paths(cfg: dict[str, Any]) -> None:
    required = ("checkpoint_path", "h5ad_file", "embed_bin", "pairs_csv", "env_dict",
                "ligand_list", "receptor_list", "tf_list", "lr_pairs")
    missing = [key for key in required if not Path(cfg[key]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing configured inputs: {missing}")
    expected_bytes = int(cfg["n_cells"]) * int(cfg["n_genes"]) * int(cfg["embed_dim"]) * 2
    actual_bytes = Path(cfg["embed_bin"]).stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"Embedding memmap size mismatch: expected={expected_bytes}, actual={actual_bytes}")


def load_tables(cfg: dict[str, Any]):
    ligand = pd.read_csv(cfg["ligand_list"], sep="\t")
    receptor = pd.read_csv(cfg["receptor_list"], sep="\t")
    tf = pd.read_csv(cfg["tf_list"], sep="\t")
    lr = pd.read_csv(cfg["lr_pairs"], sep="\t")
    return ligand, receptor, tf, lr


def prepare_indices(cfg: dict[str, Any]):
    ligand, receptor, tf, lr = load_tables(cfg)
    ligand_key = "new_index" if "new_index" in ligand else "list_index"
    receptor_key = "new_index" if "new_index" in receptor else "list_index"
    ligand_map = dict(zip(ligand["gene_name"].astype(str), ligand[ligand_key].astype(int)))
    receptor_map = dict(zip(receptor["gene_name"].astype(str), receptor[receptor_key].astype(int)))
    mask = np.zeros((len(ligand), len(receptor)), dtype=np.float32)
    for row in lr.itertuples(index=False):
        lig = str(getattr(row, "ligand_gene_symbol"))
        rec = str(getattr(row, "receptor_gene_symbol"))
        if lig in ligand_map and rec in receptor_map:
            mask[ligand_map[lig], receptor_map[rec]] = 1.0
    return (torch.tensor(receptor["original_index"].to_numpy(), dtype=torch.long),
            torch.tensor(tf["original_index"].to_numpy(), dtype=torch.long),
            torch.tensor(mask), ligand, receptor, tf)


class InterpretabilityDataset(Dataset):
    """Lazy source-cell reader that avoids densifying the full h5ad matrix."""

    def __init__(self, cfg: dict[str, Any], selected_rows: np.ndarray):
        self.cfg = cfg
        self.pairs = pd.read_csv(cfg["pairs_csv"]).iloc[selected_rows].reset_index(drop=False).rename(columns={"index": "pair_index"})
        self.embedding = np.memmap(cfg["embed_bin"], dtype="float16", mode="r",
                                   shape=(cfg["n_cells"], cfg["n_genes"], cfg["embed_dim"]))
        self.adata = anndata.read_h5ad(cfg["h5ad_file"], backed="r")
        self.cell_to_index = {str(cell): i for i, cell in enumerate(self.adata.obs_names)}
        self.var_names = self.adata.var_names.astype(str).tolist()
        self.env_dict = torch.load(cfg["env_dict"], map_location="cpu")
        self.n_ligands = len(pd.read_csv(cfg["ligand_list"], sep="\t"))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        cell_id = str(row["source_index"])
        cell_idx = self.cell_to_index[cell_id]
        emb = torch.from_numpy(self.embedding[cell_idx].copy()).float()
        env = self.env_dict.get((row["experiment"], float(row["day_source"])), torch.zeros(self.n_ligands))
        env = env.detach().clone().float() if isinstance(env, torch.Tensor) else torch.tensor(env, dtype=torch.float32)
        x = self.adata.X[cell_idx]
        if sparse.issparse(x):
            x = x.toarray()
        expr = np.asarray(x, dtype=np.float32).reshape(-1)
        return {
            "embedding": emb,
            "ligand": env,
            "source_expr": torch.from_numpy(expr),
            "pair_index": int(row["pair_index"]),
            "source_index": cell_id,
            "experiment": str(row["experiment"]),
            "cell_line": str(row.get("cell_line", "NA")),
            "day_source": float(row["day_source"]),
            "day_target": float(row["day_target"]),
            "window": WINDOW_LABEL[(float(row["day_source"]), float(row["day_target"]))],
        }


def select_pair_rows(cfg: dict[str, Any]) -> np.ndarray:
    pairs = pd.read_csv(cfg["pairs_csv"])
    if pairs["source_index"].astype(str).duplicated().any():
        raise ValueError("source_index is not unique in the configured pair table")
    valid = pairs.apply(lambda r: (float(r.day_source), float(r.day_target)) in WINDOW_LABEL, axis=1)
    pairs = pairs.loc[valid]
    limit = int(cfg.get("max_cells_per_window", 0))
    if limit <= 0:
        return pairs.index.to_numpy(dtype=int)
    rng = np.random.default_rng(int(cfg["seed"]))
    selected = []
    for _, _, label in WINDOWS:
        candidates = pairs.index[(pairs.day_source.astype(float).map(lambda x: f"day{int(x)}") + "_to_" +
                                  pairs.day_target.astype(float).map(lambda x: f"day{int(x)}")) == label].to_numpy()
        rng.shuffle(candidates)
        selected.extend(sorted(candidates[:limit].tolist()))
    return np.asarray(selected, dtype=int)


def global_env_by_day(env_dict: dict[Any, Any]) -> dict[float, torch.Tensor]:
    grouped: dict[float, list[torch.Tensor]] = {}
    for (_experiment, day), value in env_dict.items():
        tensor = value.detach().clone().float() if isinstance(value, torch.Tensor) else torch.tensor(value).float()
        grouped.setdefault(float(day), []).append(tensor)
    return {day: torch.stack(values).mean(0) for day, values in grouped.items()}


def condition_ligands(ligand: torch.Tensor, conditions: list[str], experiments: list[str],
                      days: torch.Tensor, env_dict: dict[Any, Any], global_env: dict[float, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for condition in conditions:
        if condition == "real":
            out[condition] = ligand.clone()
        elif condition == "zero_all_ligands":
            out[condition] = torch.zeros_like(ligand)
        elif condition == "stage_mismatch":
            rows = []
            for i, (experiment, day) in enumerate(zip(experiments, days.tolist())):
                wrong_day = MISMATCH_DAY[float(day)]
                value = env_dict.get((experiment, wrong_day), global_env[wrong_day])
                rows.append(value.detach().clone().float() if isinstance(value, torch.Tensor) else torch.tensor(value).float())
            out[condition] = torch.stack(rows)
        else:
            raise ValueError(f"Unknown condition: {condition}")
    return out


def build_model(cfg: dict[str, Any], device: torch.device):
    rec_idx, tf_idx, lr_mask, ligand, receptor, tf = prepare_indices(cfg)
    model = ScFoundationCellModel(
        n_genes=cfg["n_genes"], rec_indices=rec_idx, tf_indices=tf_idx, lr_mask=lr_mask,
        input_embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"], n_heads=cfg["n_heads"],
        tf_depth=cfg["tf_depth"], gene_depth=cfg["gene_depth"], mlp_ratio=cfg["mlp_ratio"],
        dropout=cfg["dropout"], input_mode=cfg["input_mode"], lr_mode=cfg["lr_mode"])
    state = torch.load(cfg["checkpoint_path"], map_location=device)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {result}")
    model.to(device).eval()
    return model, ligand, receptor, tf


def safe_key(*parts: str) -> str:
    return "__".join(str(part).replace("/", "_") for part in parts)


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

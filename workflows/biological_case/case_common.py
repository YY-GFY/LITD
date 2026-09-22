#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import DataLoader, Dataset, Subset

from case_model import ScFoundationCellModel

try:
    import anndata
except ImportError:  # pragma: no cover
    anndata = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "litd_biological_case.json"


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    analysis_name: str
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    huber_delta: float
    entropy_lambda: float
    gradient_clip_norm: float
    hidden_dim: int
    n_heads: int
    tf_depth: int
    gene_depth: int
    mlp_ratio: float
    dropout: float
    device: str
    seed: int
    input_mode: str
    lr_mode: str
    num_workers: int
    train_fraction: float
    split_seed: int
    n_cells: int
    n_genes: int
    embed_dim: int
    h5ad_file: str
    expr_csv: str
    embed_bin: str
    pairs_csv: str
    env_dict: str
    ligand_list: str
    receptor_list: str
    tf_list: str
    lr_pairs: str
    tf_db: str
    human_tfs: str
    target_genes_tsv: str
    checkpoint_root: str
    checkpoint_path: str
    source_window_artifact_root: str
    source_case_root: str
    chip_network_path: str
    string_network_path: str


def load_config(config_path: Path) -> ExperimentConfig:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    path_keys = {
        "h5ad_file", "expr_csv", "embed_bin", "pairs_csv", "env_dict",
        "ligand_list", "receptor_list", "tf_list", "lr_pairs", "tf_db",
        "human_tfs", "target_genes_tsv", "checkpoint_root", "checkpoint_path",
        "source_window_artifact_root", "source_case_root", "chip_network_path",
        "string_network_path",
    }
    for key in path_keys:
        path = Path(payload[key])
        payload[key] = str(path if path.is_absolute() else ROOT / path)
    return ExperimentConfig(**payload)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_metadata_frames(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ligand_df = pd.read_csv(cfg.ligand_list, sep="\t")
    receptor_df = pd.read_csv(cfg.receptor_list, sep="\t")
    tf_df = pd.read_csv(cfg.tf_list, sep="\t")
    lr_pairs_df = pd.read_csv(cfg.lr_pairs, sep="\t")
    return ligand_df, receptor_df, tf_df, lr_pairs_df


def prepare_metadata(cfg: ExperimentConfig):
    ligand_df, receptor_df, tf_df, lr_pairs_df = load_metadata_frames(cfg)

    rec_indices = receptor_df["original_index"].to_numpy()
    tf_indices = tf_df["original_index"].to_numpy()

    ligand_key = "new_index" if "new_index" in ligand_df.columns else "list_index"
    receptor_key = "new_index" if "new_index" in receptor_df.columns else "list_index"
    ligand_map = dict(zip(ligand_df["gene_name"], ligand_df[ligand_key]))
    receptor_map = dict(zip(receptor_df["gene_name"], receptor_df[receptor_key]))

    lr_mask = np.zeros((len(ligand_df), len(receptor_df)), dtype=np.float32)
    for _, row in lr_pairs_df.iterrows():
        lig = row["ligand_gene_symbol"]
        rec = row["receptor_gene_symbol"]
        if lig in ligand_map and rec in receptor_map:
            lr_mask[ligand_map[lig], receptor_map[rec]] = 1.0

    return (
        torch.tensor(rec_indices, dtype=torch.long),
        torch.tensor(tf_indices, dtype=torch.long),
        torch.tensor(lr_mask, dtype=torch.float32),
    )


def build_cell_id_mapping(h5ad_file: str) -> dict[str, int]:
    if anndata is None:
        raise ImportError("anndata is required to build cell ID mappings.")
    adata = anndata.read_h5ad(h5ad_file, backed="r")
    return {name: i for i, name in enumerate(adata.obs_names)}


class BiologicalCaseDataset(Dataset):
    def __init__(self, cfg: ExperimentConfig):
        if anndata is None:
            raise ImportError("anndata is required to construct BiologicalCaseDataset.")
        self.cfg = cfg
        self.sc_embeddings = np.memmap(
            cfg.embed_bin,
            dtype="float16",
            mode="r",
            shape=(cfg.n_cells, cfg.n_genes, cfg.embed_dim),
        )
        self.pairs_df = pd.read_csv(cfg.pairs_csv)
        self.env_dict = torch.load(cfg.env_dict)
        self.adata = anndata.read_h5ad(cfg.h5ad_file)
        if sparse.issparse(self.adata.X):
            self.expr_matrix = self.adata.X.toarray().astype(np.float32)
        else:
            self.expr_matrix = np.asarray(self.adata.X, dtype=np.float32)
        self.obs_name_to_idx = {name: i for i, name in enumerate(self.adata.obs_names)}
        self.cell_id_map = self.obs_name_to_idx
        self.var_names = self.adata.var_names.astype(str).tolist()
        self.n_genes = self.expr_matrix.shape[1]
        self.n_ligand = len(pd.read_csv(cfg.ligand_list, sep="\t"))

    def __len__(self) -> int:
        return len(self.pairs_df)

    def __getitem__(self, idx: int):
        row = self.pairs_df.iloc[idx]
        source_cell_id = row["source_index"]
        target_cell_id = row["target_index"]

        source_idx_emb = self.cell_id_map[source_cell_id]
        emb_t = torch.from_numpy(self.sc_embeddings[source_idx_emb].copy()).float()

        env_key = (row["experiment"], float(row["day_source"]))
        ligand_vec = self.env_dict.get(env_key, torch.zeros(self.n_ligand))
        if not isinstance(ligand_vec, torch.Tensor):
            ligand_vec = torch.tensor(ligand_vec)

        s_idx_expr = self.obs_name_to_idx[source_cell_id]
        t_idx_expr = self.obs_name_to_idx[target_cell_id]
        src_expr = torch.from_numpy(self.expr_matrix[s_idx_expr]).float()
        tgt_expr = torch.from_numpy(self.expr_matrix[t_idx_expr]).float()
        return emb_t, ligand_vec.float(), src_expr, tgt_expr


def build_index_split(n_items: int, train_fraction: float, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(split_seed)
    indices = np.arange(n_items)
    rng.shuffle(indices)
    train_size = int(n_items * train_fraction)
    train_idx = np.sort(indices[:train_size])
    test_idx = np.sort(indices[train_size:])
    return train_idx, test_idx


def build_loaders(dataset: Dataset, cfg: ExperimentConfig):
    train_idx, test_idx = build_index_split(len(dataset), cfg.train_fraction, cfg.split_seed)
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx.tolist()),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, train_idx, test_idx


def build_model(dataset: BiologicalCaseDataset, cfg: ExperimentConfig, device: torch.device) -> ScFoundationCellModel:
    rec_indices, tf_indices, lr_mask = prepare_metadata(cfg)
    return ScFoundationCellModel(
        n_genes=dataset.n_genes,
        rec_indices=rec_indices,
        tf_indices=tf_indices,
        lr_mask=lr_mask,
        input_embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        n_heads=cfg.n_heads,
        tf_depth=cfg.tf_depth,
        gene_depth=cfg.gene_depth,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        input_mode=cfg.input_mode,
        lr_mode=cfg.lr_mode,
    ).to(device)

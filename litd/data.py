from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    expr_path: str
    gene_ordering_path: str
    time_parser: Callable[[str], float]
    default_experiment: str = "exp_1"
    default_cell_line: str = "combined"
    existing_embedding: str | None = None


def parse_hhep_time(cell_name: str) -> float:
    suffix = str(cell_name).split("_")[-1]
    mapping = {
        "ipsc": 0.0,
        "de": 1.0,
        "he1": 2.0,
        "he2": 2.0,
        "ih1": 3.0,
        "mh1": 4.0,
        "mh2": 4.0,
    }
    if suffix not in mapping:
        raise ValueError(f"Unknown hHEP cell suffix: {cell_name}")
    return mapping[suffix]


def parse_hesc_time(cell_name: str) -> float:
    token = str(cell_name).split("_")[1]
    mapping = {
        "00hb4s": 0.0,
        "12h": 1.0,
        "24h": 2.0,
        "36h": 3.0,
        "72h": 4.0,
        "96h": 5.0,
    }
    if token not in mapping:
        raise ValueError(f"Unknown hESC cell token: {cell_name}")
    return mapping[token]


def load_expression_as_gene_by_cell(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    if not df.index.is_unique:
        raise ValueError(f"Gene index is not unique: {path}")

    cell_patterns = (
        lambda s: bool(re.search(r"_(ipsc|de|he1|he2|ih1|mh1|mh2)$", s)),
        lambda s: bool(re.match(r"^H9_(00hb4s|12h|24h|36h|72h|96h)_", s)),
    )

    index_cell_hits = sum(any(fn(str(x)) for fn in cell_patterns) for x in df.index[:50])
    col_cell_hits = sum(any(fn(str(x)) for fn in cell_patterns) for x in df.columns[:50])

    if col_cell_hits >= index_cell_hits:
        return df
    return df.T


def write_gene_by_cell(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def filter_to_scf_genes(expr_gene_by_cell: pd.DataFrame, scf_gene_list_path: str) -> pd.DataFrame:
    scf = pd.read_csv(scf_gene_list_path, sep="\t")
    gene_list = scf["gene_name"].astype(str).tolist()
    keep = [g for g in expr_gene_by_cell.index.astype(str) if g in set(gene_list)]
    return expr_gene_by_cell.loc[keep].copy()


def build_anndata_from_gene_by_cell(expr_gene_by_cell: pd.DataFrame) -> ad.AnnData:
    cell_by_gene = expr_gene_by_cell.T
    adata = ad.AnnData(cell_by_gene)
    adata.obs_names = cell_by_gene.index.astype(str)
    adata.var_names = cell_by_gene.columns.astype(str)
    return adata


def attach_basic_obs(adata: ad.AnnData, spec: DatasetSpec) -> ad.AnnData:
    adata = adata.copy()
    adata.obs["time_point"] = [spec.time_parser(name) for name in adata.obs_names]
    adata.obs["experiment"] = spec.default_experiment
    adata.obs["cell_line"] = spec.default_cell_line
    return adata


def normalized_log1p_if_needed(adata: ad.AnnData) -> ad.AnnData:
    adata = adata.copy()
    max_val = float(np.max(adata.X))
    if max_val > 20:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return adata


def torch_save(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

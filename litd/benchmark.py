from __future__ import annotations

import itertools
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import anndata as ad
import numpy as np
import ot
import pandas as pd
import scanpy as sc
import torch
from scipy.spatial.distance import cdist
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .data import (
    DatasetSpec,
    attach_basic_obs,
    build_anndata_from_gene_by_cell,
    ensure_dir,
    filter_to_scf_genes,
    load_expression_as_gene_by_cell,
    normalized_log1p_if_needed,
    parse_hesc_time,
    parse_hhep_time,
    torch_save,
    write_gene_by_cell,
)
from .model import ScFoundationCellModel


@dataclass(frozen=True)
class Paths:
    workdir: str
    scf_gene_list: str
    lr_db: str
    tf_db: str
    human_tfs: str
    string_gt: str
    hepg2_gt: str
    hesc_gt: str
    nonspecific_gt: str
    scf_model_dir: str
    scf_ckpt: str
    hhep_expr: str
    hhep_gene_ordering: str
    hhep_existing_embedding: str
    hesc_expr: str
    hesc_gene_ordering: str


DATASET_SPECS: dict[str, DatasetSpec] = {}


def build_specs(paths: Paths) -> dict[str, DatasetSpec]:
    return {
        "hhep": DatasetSpec(
            name="hhep",
            expr_path=paths.hhep_expr,
            gene_ordering_path=paths.hhep_gene_ordering,
            time_parser=parse_hhep_time,
            existing_embedding=paths.hhep_existing_embedding,
        ),
        "hesc": DatasetSpec(
            name="hesc",
            expr_path=paths.hesc_expr,
            gene_ordering_path=paths.hesc_gene_ordering,
            time_parser=parse_hesc_time,
        ),
    }


def dataset_dir(paths: Paths, dataset_name: str) -> Path:
    return Path(paths.workdir) / "datasets" / dataset_name


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_metadata_from_expr(expr_filtered: pd.DataFrame, lr_db_path: str, tf_db_path: str):
    gene_names = expr_filtered.index.astype(str).tolist()
    gene_map = {name: i for i, name in enumerate(gene_names)}

    def create_full_metadata_df(gene_subset):
        data = []
        seen = set()
        for gene in gene_subset:
            if gene in gene_map and gene not in seen:
                data.append({"gene_name": gene, "original_index": gene_map[gene]})
                seen.add(gene)
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values("original_index").reset_index(drop=True)
            df["list_index"] = df.index
        return df

    lr_df = pd.read_csv(lr_db_path, sep="\t")
    if "ligand_gene_symbol" not in lr_df.columns and len(lr_df.columns) >= 2:
        lr_df = lr_df.rename(columns={lr_df.columns[0]: "ligand_gene_symbol", lr_df.columns[1]: "receptor_gene_symbol"})

    lr_mask = lr_df["ligand_gene_symbol"].isin(gene_map) & lr_df["receptor_gene_symbol"].isin(gene_map)
    filtered_lr_df = lr_df.loc[lr_mask].copy()
    ligand_df = create_full_metadata_df(filtered_lr_df["ligand_gene_symbol"].unique())
    receptor_df = create_full_metadata_df(filtered_lr_df["receptor_gene_symbol"].unique())

    tf_source_df = pd.read_csv(tf_db_path, index_col=0)
    tf_col = "TF"
    for col in tf_source_df.columns:
        if "TF" in col or "tf" in col:
            tf_col = col
            break
    tf_df = create_full_metadata_df(tf_source_df[tf_col].astype(str).unique())
    return filtered_lr_df, ligand_df, receptor_df, tf_df


def compute_env_context(adata: ad.AnnData, ligand_df: pd.DataFrame):
    ligand_indices = ligand_df["original_index"].to_numpy()
    env_context = {}
    groups = adata.obs.groupby(["experiment", "time_point"], observed=True)
    for (experiment, time_point), group_df in groups:
        sub = adata[group_df.index]
        ligand_expr = sub.X[:, ligand_indices]
        mean_expr = ligand_expr.toarray().mean(axis=0) if hasattr(ligand_expr, "toarray") else np.asarray(ligand_expr).mean(axis=0)
        env_context[(str(experiment), float(time_point))] = torch.tensor(np.asarray(mean_expr).flatten(), dtype=torch.float32)
    return env_context


def compute_cell_pairs(adata: ad.AnnData, output_csv: str) -> pd.DataFrame:
    work = normalized_log1p_if_needed(adata)
    sc.tl.pca(work, n_comps=50, svd_solver="arpack")
    all_x_pca = np.array(work.obsm["X_pca"], copy=True)
    cell_name_to_idx = {name: i for i, name in enumerate(work.obs_names)}
    pairs_list = []
    groups = work.obs.groupby(["experiment", "cell_line"], observed=False)
    for (experiment, cell_line), group_df in groups:
        group_cell_names = group_df.index.tolist()
        group_times = group_df["time_point"].to_numpy()
        unique_times = sorted(np.unique(group_times))
        for i in range(len(unique_times) - 1):
            t_start = unique_times[i]
            t_end = unique_times[i + 1]
            if t_end - t_start > 1.5:
                continue
            src_names = np.array(group_cell_names)[group_times == t_start]
            tgt_names = np.array(group_cell_names)[group_times == t_end]
            if len(src_names) < 5 or len(tgt_names) < 5:
                continue
            src_idx = [cell_name_to_idx[name] for name in src_names]
            tgt_idx = [cell_name_to_idx[name] for name in tgt_names]
            src_pca = all_x_pca[src_idx]
            tgt_pca = all_x_pca[tgt_idx]
            distance = cdist(src_pca, tgt_pca, metric="sqeuclidean")
            distance /= max(distance.max(), 1e-8)
            transport = ot.sinkhorn(
                np.ones(len(src_pca)) / len(src_pca),
                np.ones(len(tgt_pca)) / len(tgt_pca),
                distance,
                reg=0.05,
            )
            target_relative_indices = np.argmax(transport, axis=1)
            for src_rel_i, tgt_rel_i in enumerate(target_relative_indices):
                probability = float(transport[src_rel_i, tgt_rel_i])
                if probability <= 1e-9:
                    continue
                pairs_list.append(
                    {
                        "source_index": src_names[src_rel_i],
                        "target_index": tgt_names[tgt_rel_i],
                        "probability": probability,
                        "experiment": experiment,
                        "cell_line": cell_line,
                        "day_source": float(t_start),
                        "day_target": float(t_end),
                    }
                )
    pairs_df = pd.DataFrame(pairs_list)
    pairs_df.to_csv(output_csv, index=False)
    return pairs_df


def prepare_dataset(paths: Paths, dataset_name: str) -> None:
    spec = build_specs(paths)[dataset_name]
    ds_dir = dataset_dir(paths, dataset_name)
    ensure_dir(str(ds_dir / "data"))
    expr = load_expression_as_gene_by_cell(spec.expr_path)
    expr_filtered = filter_to_scf_genes(expr, paths.scf_gene_list)
    write_gene_by_cell(expr, str(ds_dir / "data" / "ExpressionData.csv"))
    write_gene_by_cell(expr_filtered, str(ds_dir / "data" / "ExpressionData_filtered.csv"))
    expr.T.to_csv(ds_dir / "data" / "scfoundation_input.csv")

    filtered_lr_df, ligand_df, receptor_df, tf_df = prepare_metadata_from_expr(expr_filtered, paths.lr_db, paths.tf_db)
    filtered_lr_df.to_csv(ds_dir / "data" / "filtered_lr_pairs.tsv", sep="\t", index=False)
    ligand_df.to_csv(ds_dir / "data" / "ligand_list.tsv", sep="\t", index=False)
    receptor_df.to_csv(ds_dir / "data" / "receptor_list.tsv", sep="\t", index=False)
    tf_df.to_csv(ds_dir / "data" / "tf_list.tsv", sep="\t", index=False)

    adata = build_anndata_from_gene_by_cell(expr_filtered)
    adata = attach_basic_obs(adata, spec)
    adata.write_h5ad(ds_dir / "data" / "ExpressionData_filtered.h5ad")
    env_context = compute_env_context(adata, ligand_df)
    torch_save(env_context, str(ds_dir / "data" / "env_context_dict.pt"))
    compute_cell_pairs(adata, str(ds_dir / "data" / "cell_pairs_train.csv"))


def ensure_embedding(paths: Paths, dataset_name: str, pre_normalized: str = "T", tgthighres: str = "t4") -> None:
    spec = build_specs(paths)[dataset_name]
    ds_dir = dataset_dir(paths, dataset_name)
    target_path = ds_dir / "data" / f"{dataset_name}_01B-resolution_singlecell_gene_embedding_{tgthighres}_resolution.npy"
    if target_path.exists():
        return
    if spec.existing_embedding and Path(spec.existing_embedding).is_file():
        if target_path.is_symlink():
            target_path.unlink()
        os.symlink(spec.existing_embedding, target_path)
        return
    cmd = [
        sys.executable,
        str(Path(paths.scf_model_dir) / "get_embedding.py"),
        "--task_name",
        dataset_name,
        "--input_type",
        "singlecell",
        "--output_type",
        "gene",
        "--tgthighres",
        tgthighres,
        "--data_path",
        str(ds_dir / "data" / "scfoundation_input.csv"),
        "--save_path",
        str(ds_dir / "data"),
        "--pre_normalized",
        pre_normalized,
        "--version",
        "ce",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{paths.scf_model_dir}:{env.get('PYTHONPATH', '')}"
    subprocess.run(cmd, cwd=paths.scf_model_dir, env=env, check=True)
    generated = ds_dir / "data" / f"{dataset_name}_01B-resolution_singlecell_gene_embedding_{tgthighres}_resolution.npy"
    if generated != target_path and generated.exists():
        generated.rename(target_path)


def load_runtime_paths(paths: Paths, dataset_name: str):
    ds_dir = dataset_dir(paths, dataset_name)
    return {
        "embed_file": str(next(ds_dir.glob("data/*_singlecell_gene_embedding_t4_resolution.npy"))),
        "embed_gene_list": paths.scf_gene_list,
        "expr_file": str(ds_dir / "data" / "ExpressionData_filtered.csv"),
        "pairs_csv": str(ds_dir / "data" / "cell_pairs_train.csv"),
        "env_dict": str(ds_dir / "data" / "env_context_dict.pt"),
        "receptor_list": str(ds_dir / "data" / "receptor_list.tsv"),
        "ligand_list": str(ds_dir / "data" / "ligand_list.tsv"),
        "tf_list": str(ds_dir / "data" / "tf_list.tsv"),
        "lr_pairs": str(ds_dir / "data" / "filtered_lr_pairs.tsv"),
        "gene_ordering": build_specs(paths)[dataset_name].gene_ordering_path,
    }


def prepare_metadata(paths_map):
    ligand_df = pd.read_csv(paths_map["ligand_list"], sep="\t")
    receptor_df = pd.read_csv(paths_map["receptor_list"], sep="\t")
    tf_df = pd.read_csv(paths_map["tf_list"], sep="\t")
    pair_df = pd.read_csv(paths_map["lr_pairs"], sep="\t")
    ligand_map = dict(zip(ligand_df["gene_name"], ligand_df["list_index"]))
    receptor_map = dict(zip(receptor_df["gene_name"], receptor_df["list_index"]))
    mask = np.zeros((len(ligand_df), len(receptor_df)))
    for _, row in pair_df.iterrows():
        lig = row["ligand_gene_symbol"]
        rec = row["receptor_gene_symbol"]
        if lig in ligand_map and rec in receptor_map:
            mask[ligand_map[lig], receptor_map[rec]] = 1
    return (
        torch.tensor(receptor_df["original_index"].to_numpy(), dtype=torch.long),
        torch.tensor(tf_df["original_index"].to_numpy(), dtype=torch.long),
        torch.tensor(mask, dtype=torch.float32),
    )


class ScCellDataset(Dataset):
    def __init__(self, paths_map):
        expr = load_expression_as_gene_by_cell(paths_map["expr_file"])
        self.expr_matrix = expr.T.to_numpy(dtype=np.float32)
        self.obs_names = expr.columns.astype(str).tolist()
        self.var_names = expr.index.astype(str).tolist()
        emb_gene_list = pd.read_csv(paths_map["embed_gene_list"], sep="\t")["gene_name"].astype(str).tolist()
        emb_gene_map = {gene: i for i, gene in enumerate(emb_gene_list)}
        valid_emb_indices = [emb_gene_map.get(gene, emb_gene_map.get(gene.upper(), 0)) for gene in self.var_names]
        raw_embeddings = np.load(paths_map["embed_file"])
        self.sc_embeddings = raw_embeddings[:, valid_emb_indices, :]
        self.pairs_df = pd.read_csv(paths_map["pairs_csv"])
        self.env_dict = torch.load(paths_map["env_dict"])
        self.obs_name_to_idx = {name: i for i, name in enumerate(self.obs_names)}
        self.n_ligand = len(pd.read_csv(paths_map["ligand_list"], sep="\t"))
        self.n_genes = len(self.var_names)

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        s_idx = self.obs_name_to_idx.get(str(row["source_index"]), 0)
        t_idx = self.obs_name_to_idx.get(str(row["target_index"]), 0)
        emb_t = torch.from_numpy(self.sc_embeddings[s_idx].copy()).float()
        env_key = (str(row["experiment"]), float(row["day_source"]))
        ligand_vec = self.env_dict.get(env_key, torch.zeros(self.n_ligand))
        if not isinstance(ligand_vec, torch.Tensor):
            ligand_vec = torch.tensor(ligand_vec)
        src_expr = torch.from_numpy(self.expr_matrix[s_idx]).float()
        tgt_expr = torch.from_numpy(self.expr_matrix[t_idx]).float()
        return emb_t, ligand_vec.float(), src_expr, tgt_expr


def evaluate_gene_pearson(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    input_mode = getattr(model, "input_mode", "scfoundation")
    with torch.no_grad():
        for emb_t, ligand_vec, src_expr, tgt_expr in loader:
            emb_t = emb_t.to(device)
            ligand_vec = ligand_vec.to(device)
            src_expr = src_expr.to(device)
            tgt_expr = tgt_expr.to(device)
            model_input = emb_t if input_mode == "scfoundation" else src_expr
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=device.type == "cuda"):
                final_pred, _ = model(model_input, ligand_vec, return_grn=True)
            all_preds.append(final_pred.float().cpu().numpy())
            all_targets.append(tgt_expr.float().cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    epsilon = 1e-8
    vx_g = all_preds - np.mean(all_preds, axis=0)
    vy_g = all_targets - np.mean(all_targets, axis=0)
    cost_g = np.sum(vx_g * vy_g, axis=0) / (np.sqrt(np.sum(vx_g**2, axis=0)) * np.sqrt(np.sum(vy_g**2, axis=0)) + epsilon)
    valid_mask_g = np.std(all_targets, axis=0) > 1e-5
    return float(np.mean(cost_g[valid_mask_g])) if np.sum(valid_mask_g) > 0 else 0.0


def train_one(paths: Paths, dataset_name: str, args) -> str:
    seed = getattr(args, "seed", 42)
    set_seed(seed)
    paths_map = load_runtime_paths(paths, dataset_name)
    rec_indices, tf_indices, lr_mask = prepare_metadata(paths_map)
    full_dataset = ScCellDataset(paths_map)
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    input_mode = getattr(args, "input_mode", "scfoundation")
    lr_mode = getattr(args, "lr_mode", "masked")
    model = ScFoundationCellModel(
        n_genes=full_dataset.n_genes,
        rec_indices=rec_indices,
        tf_indices=tf_indices,
        lr_mask=lr_mask,
        input_embed_dim=512,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        tf_depth=args.tf_depth,
        gene_depth=args.gene_depth,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        input_mode=input_mode,
        lr_mode=lr_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.HuberLoss(delta=0.5)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    best_score = -1e9
    out_dir = dataset_dir(paths, dataset_name) / "runs" / args.run_name
    ensure_dir(str(out_dir / "models"))
    model_path = out_dir / "models" / f"{dataset_name}.pth"
    for _ in range(args.epochs):
        model.train()
        for emb_t, ligand_vec, src_expr, tgt_expr in train_loader:
            optimizer.zero_grad(set_to_none=True)
            emb_t = emb_t.to(device)
            ligand_vec = ligand_vec.to(device)
            src_expr = src_expr.to(device)
            tgt_expr = tgt_expr.to(device)
            model_input = emb_t if input_mode == "scfoundation" else src_expr
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=device.type == "cuda"):
                pred, grn_matrix = model(model_input, ligand_vec, return_grn=True)
                loss_main = criterion(pred, tgt_expr)
                entropy = -(grn_matrix * torch.log(grn_matrix + 1e-8)).sum(dim=-1).mean()
                loss = loss_main + args.entropy_lambda * entropy
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        val_score = evaluate_gene_pearson(model, test_loader, device)
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), model_path)
    return str(model_path)


def prepare_gene_subset(all_model_genes, num_genes, include_tfs, gene_ordering_path, tf_list_path):
    gene_df = pd.read_csv(gene_ordering_path, header=0, index_col=0)
    tfs = []
    if include_tfs:
        tfs_df = pd.read_csv(tf_list_path, header=0)
        tfs = tfs_df[tfs_df.columns[0]]
        tfs = tfs[tfs.isin(all_model_genes)]
    if len(gene_df.columns) >= 2:
        gene_df.sort_values(by=gene_df.columns[1], inplace=True, ascending=False)
    variable_genes = gene_df.index.values[: min(num_genes, len(gene_df))]
    variable_tfs = set(tfs[tfs.isin(gene_df.index)]) if include_tfs and len(tfs) > 0 else set()
    return set(variable_genes) | variable_tfs


def infer_targets(paths: Paths, dataset_name: str, args, model_path: str, target_gene_counts=(500, 1000), *, all_candidates=False, output_dir=None, score_accumulation="native_mean") -> dict[int, str]:
    paths_map = load_runtime_paths(paths, dataset_name)
    rec_indices, tf_indices, lr_mask = prepare_metadata(paths_map)
    dataset = ScCellDataset(paths_map)
    all_gene_names = np.array(dataset.var_names)
    model_tf_names = all_gene_names[tf_indices.cpu().numpy()]
    target_subsets = {
        int(target_gene_count): prepare_gene_subset(
            all_gene_names,
            int(target_gene_count),
            True,
            paths_map["gene_ordering"],
            paths.human_tfs,
        )
        for target_gene_count in target_gene_counts
    }
    if all_candidates:
        # The weak-correlation experiment uses the complete original candidate
        # universe, including scores below the ordinary benchmark threshold.
        target_subsets = {int(n): get_tf_target_space(paths, dataset_name, int(n))[1]
                          for n in target_gene_counts}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    input_mode = getattr(args, "input_mode", "scfoundation")
    lr_mode = getattr(args, "lr_mode", "masked")
    model = ScFoundationCellModel(
        n_genes=dataset.n_genes,
        rec_indices=rec_indices,
        tf_indices=tf_indices,
        lr_mask=lr_mask,
        input_embed_dim=512,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        tf_depth=args.tf_depth,
        gene_depth=args.gene_depth,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        input_mode=input_mode,
        lr_mode=lr_mode,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    subset_indices = list(range(min(args.inf_n_samples, len(dataset))))
    loader = DataLoader(Subset(dataset, subset_indices), batch_size=args.inf_batch_size, shuffle=False)
    if score_accumulation not in {"native_mean", "float32_sum"}:
        raise ValueError(f"Unknown score accumulation: {score_accumulation}")
    grn_batches = []
    score_sum, seen = None, 0
    with torch.no_grad():
        for emb_t, ligand_vec, src_expr, _ in loader:
            emb_t = emb_t.to(device)
            ligand_vec = ligand_vec.to(device)
            src_expr = src_expr.to(device)
            model_input = emb_t if input_mode == "scfoundation" else src_expr
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=device.type == "cuda"):
                _, grn_matrix = model(model_input, ligand_vec, return_grn=True)
            if score_accumulation == "float32_sum":
                batch_sum = grn_matrix.detach().float().sum(dim=0).cpu()
                score_sum = batch_sum if score_sum is None else score_sum + batch_sum
                seen += int(grn_matrix.shape[0])
            else:
                grn_batches.append(grn_matrix.detach().cpu())
    if score_accumulation == "float32_sum":
        if seen == 0:
            raise ValueError("No inference samples")
        avg_attn_matrix = (score_sum / seen).numpy()
    else:
        if not grn_batches:
            raise ValueError("No inference samples")
        avg_attn_matrix = torch.cat(grn_batches, dim=0).mean(dim=0).numpy()
    df_full = pd.DataFrame(avg_attn_matrix, index=all_gene_names, columns=model_tf_names)
    out_dir = Path(output_dir) if output_dir is not None else dataset_dir(paths, dataset_name) / "runs" / args.run_name / "results"
    ensure_dir(str(out_dir))
    out_paths = {}
    for target_gene_count, target_gene_subset in target_subsets.items():
        valid_subset = [g for g in target_gene_subset if g in df_full.index]
        df_filtered = df_full.loc[valid_subset]
        df_long = df_filtered.reset_index().melt(id_vars="index", var_name="Gene1", value_name="EdgeWeight")
        df_long.rename(columns={"index": "Gene2"}, inplace=True)
        df_long = df_long[["Gene1", "Gene2", "EdgeWeight"]]
        threshold = args.edge_threshold * 2.0 if int(target_gene_count) == 500 else args.edge_threshold
        if all_candidates:
            df_long = df_long[df_long["Gene1"] != df_long["Gene2"]].copy()
            df_long["EdgeWeight"] = df_long["EdgeWeight"].abs()
            if not np.isfinite(df_long["EdgeWeight"]).all():
                raise ValueError("Non-finite complete candidate scores")
            df_long = df_long.sort_values(["EdgeWeight", "Gene1", "Gene2"],
                                         ascending=[False, True, True], kind="mergesort")
            out_path = out_dir / f"{dataset_name}_tgt{target_gene_count}_all_candidate_edges.csv.gz"
        else:
            df_long = df_long[df_long["EdgeWeight"] > threshold]
            out_path = out_dir / f"{dataset_name}_tgt{target_gene_count}.csv"
        df_long.to_csv(out_path, index=False)
        out_paths[int(target_gene_count)] = str(out_path)
    return out_paths


def infer_one(paths: Paths, dataset_name: str, args, model_path: str, target_gene_count: int) -> str:
    return infer_targets(paths, dataset_name, args, model_path, (target_gene_count,))[int(target_gene_count)]


def early_precision(gt_path: str, pred_path: str):
    true_df = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)
    return early_precision_from_frames(true_df, pred_df)


def early_precision_from_frames(true_df: pd.DataFrame, pred_df: pd.DataFrame):
    true_df = true_df[true_df["Gene1"] != true_df["Gene2"]].drop_duplicates()
    true_edges = set(true_df["Gene1"] + "|" + true_df["Gene2"])
    pred_df = pred_df[pred_df["Gene1"] != pred_df["Gene2"]].drop_duplicates()
    pred_df = pred_df.copy()
    pred_df["Edges"] = pred_df["Gene1"] + "|" + pred_df["Gene2"]
    pred_df["EdgeWeight"] = pred_df["EdgeWeight"].abs()
    pred_df.sort_values(by="EdgeWeight", ascending=False, inplace=True)
    unique_nodes = np.unique(true_df[["Gene1", "Gene2"]])
    tf_nodes = set(true_df["Gene1"])
    possible_edges = {"|".join(edge) for edge in itertools.product(tf_nodes, unique_nodes)}
    true_edges = true_edges.intersection(possible_edges)
    pred_df = pred_df[pred_df["Edges"].isin(possible_edges)]
    num_edges = len(true_edges)
    if pred_df.empty or num_edges == 0:
        return 0.0, 0.0, 0.0
    maxk = min(pred_df.shape[0], num_edges)
    threshold = pred_df.iloc[maxk - 1]["EdgeWeight"]
    rank_set = set(pred_df[pred_df["EdgeWeight"] >= threshold]["Edges"])
    intersection = len(rank_set.intersection(true_edges))
    eprec = intersection / len(rank_set) if rank_set else 0.0
    erec = intersection / num_edges if num_edges else 0.0
    random_eprec = num_edges / len(possible_edges) if possible_edges else 1e-9
    return eprec, erec, eprec / random_eprec if random_eprec > 0 else 0.0


def _evaluate_ranked_edges(true_df: pd.DataFrame, pred_df: pd.DataFrame, possible_edges_count: int):
    true_edges = set(true_df["Gene1"] + "|" + true_df["Gene2"])
    num_edges = len(true_edges)
    if pred_df.empty or num_edges == 0 or possible_edges_count <= 0:
        return 0.0, 0.0, 0.0
    pred_df = pred_df.copy()
    pred_df["Edges"] = pred_df["Gene1"] + "|" + pred_df["Gene2"]
    pred_df["EdgeWeight"] = pred_df["EdgeWeight"].abs()
    pred_df.sort_values(by="EdgeWeight", ascending=False, inplace=True)
    maxk = min(pred_df.shape[0], num_edges)
    threshold = pred_df.iloc[maxk - 1]["EdgeWeight"]
    rank_set = set(pred_df[pred_df["EdgeWeight"] >= threshold]["Edges"])
    intersection = len(rank_set.intersection(true_edges))
    eprec = intersection / len(rank_set) if rank_set else 0.0
    erec = intersection / num_edges if num_edges else 0.0
    random_eprec = num_edges / possible_edges_count
    return eprec, erec, eprec / random_eprec if random_eprec > 0 else 0.0


@lru_cache(maxsize=None)
def get_tf_target_space(paths: Paths, dataset_name: str, target_gene_count: int):
    paths_map = load_runtime_paths(paths, dataset_name)
    expr = load_expression_as_gene_by_cell(paths_map["expr_file"])
    all_gene_names = expr.index.astype(str).tolist()
    all_gene_set = set(all_gene_names)
    tf_df = pd.read_csv(paths_map["tf_list"], sep="\t")
    tf_candidates = tuple(g for g in tf_df["gene_name"].astype(str).tolist() if g in all_gene_set)
    target_subset = prepare_gene_subset(
        np.array(all_gene_names),
        target_gene_count,
        True,
        paths_map["gene_ordering"],
        paths.human_tfs,
    )
    target_candidates = tuple(g for g in all_gene_names if g in target_subset)
    return tf_candidates, target_candidates


def early_precision_tf_target(gt_path: str, pred_path: str, tf_candidates, target_candidates):
    tf_candidates = set(tf_candidates)
    target_candidates = set(target_candidates)
    true_df = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)
    return early_precision_tf_target_from_frames(true_df, pred_df, tf_candidates, target_candidates)


def early_precision_tf_target_from_frames(true_df: pd.DataFrame, pred_df: pd.DataFrame, tf_candidates, target_candidates):
    tf_candidates = set(tf_candidates)
    target_candidates = set(target_candidates)
    true_df = true_df[
        (true_df["Gene1"] != true_df["Gene2"])
        & true_df["Gene1"].isin(tf_candidates)
        & true_df["Gene2"].isin(target_candidates)
    ].drop_duplicates()
    pred_df = pred_df[
        (pred_df["Gene1"] != pred_df["Gene2"])
        & pred_df["Gene1"].isin(tf_candidates)
        & pred_df["Gene2"].isin(target_candidates)
    ].drop_duplicates()
    possible_edges_count = len(tf_candidates) * len(target_candidates)
    return _evaluate_ranked_edges(true_df, pred_df, possible_edges_count)


def early_precision_beeline(paths: Paths, dataset_name: str, target_gene_count: int, gt_path: str, pred_path: str):
    tf_candidates, target_candidates = get_tf_target_space(paths, dataset_name, target_gene_count)
    tf_candidates = set(tf_candidates)
    target_candidates = set(target_candidates)
    true_df = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)
    return early_precision_beeline_from_frames(true_df, pred_df, tf_candidates, target_candidates)


def early_precision_beeline_from_frames(true_df: pd.DataFrame, pred_df: pd.DataFrame, tf_candidates, target_candidates):
    tf_candidates = set(tf_candidates)
    target_candidates = set(target_candidates)
    true_df = true_df[
        (true_df["Gene1"] != true_df["Gene2"])
        & true_df["Gene1"].isin(tf_candidates)
        & true_df["Gene2"].isin(target_candidates)
    ].drop_duplicates()
    pred_df = pred_df[
        (pred_df["Gene1"] != pred_df["Gene2"])
        & pred_df["Gene1"].isin(tf_candidates)
        & pred_df["Gene2"].isin(target_candidates)
    ].drop_duplicates()
    overlap_count = len(tf_candidates.intersection(target_candidates))
    possible_edges_count = len(tf_candidates) * len(target_candidates) - overlap_count
    return _evaluate_ranked_edges(true_df, pred_df, possible_edges_count)


def get_ground_truth_path(paths: Paths, dataset_name: str, network_label: str) -> str:
    if network_label == "STRING":
        return paths.string_gt
    if network_label in {"ChIP", "HepG2", "hESC-ChIP"}:
        if network_label == "HepG2":
            return paths.hepg2_gt
        if network_label == "hESC-ChIP":
            return paths.hesc_gt
        return paths.hepg2_gt if dataset_name == "hhep" else paths.hesc_gt if dataset_name == "hesc" else paths.hepg2_gt
    if network_label == "NonSpecific":
        return paths.nonspecific_gt
    raise ValueError(f"Unknown network label: {network_label}")


def evaluate_one(paths: Paths, dataset_name: str, pred_path: str, target_gene_count: int, gt_df_cache=None, pred_df_cache=None):
    chip_gt = paths.hepg2_gt if dataset_name == "hhep" else paths.hesc_gt if dataset_name == "hesc" else paths.hepg2_gt
    rows = []
    gt_df_cache = gt_df_cache if gt_df_cache is not None else {}
    pred_df_cache = pred_df_cache if pred_df_cache is not None else {}
    if pred_path not in pred_df_cache:
        pred_df_cache[pred_path] = pd.read_csv(pred_path)
    pred_df = pred_df_cache[pred_path]
    for label, gt_path in [
        ("STRING", paths.string_gt),
        ("ChIP", chip_gt),
        ("NonSpecific", paths.nonspecific_gt),
    ]:
        if gt_path not in gt_df_cache:
            gt_df_cache[gt_path] = pd.read_csv(gt_path)
        true_df = gt_df_cache[gt_path]
        eprec, erec, epr = early_precision_from_frames(true_df, pred_df)
        rows.append(
            {
                "dataset": dataset_name,
                "network": label,
                "Eprec": eprec,
                "Erec": erec,
                "EPR": epr,
            }
        )
    return rows


def save_json(path: str, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def reevaluate_existing_results(paths: Paths, args) -> None:
    summary_path = Path(args.summary_csv)
    df = pd.read_csv(summary_path)
    out_rows = []
    tf_space_cache: dict[tuple[str, int], tuple[tuple[str, ...], tuple[str, ...]]] = {}
    metric_cache: dict[tuple[str, int, str, str], tuple[float, float, float, float, float, float, float, float, float]] = {}
    gt_df_cache: dict[str, pd.DataFrame] = {}
    pred_df_cache: dict[str, pd.DataFrame] = {}

    for row in df.to_dict("records"):
        dataset_name = str(row["dataset"])
        network_label = str(row["network"])
        pred_path = str(row["pred_path"])
        target_gene_count = int(row["target_genes"])
        gt_path = get_ground_truth_path(paths, dataset_name, network_label)
        metric_key = (dataset_name, target_gene_count, network_label, pred_path)

        if metric_key not in metric_cache:
            if gt_path not in gt_df_cache:
                gt_df_cache[gt_path] = pd.read_csv(gt_path)
            if pred_path not in pred_df_cache:
                pred_df_cache[pred_path] = pd.read_csv(pred_path)
            true_df = gt_df_cache[gt_path]
            pred_df = pred_df_cache[pred_path]

            eprec, erec, epr = early_precision_from_frames(true_df, pred_df)
            space_key = (dataset_name, target_gene_count)
            if space_key not in tf_space_cache:
                tf_space_cache[space_key] = get_tf_target_space(paths, dataset_name, target_gene_count)
            tf_candidates, target_candidates = tf_space_cache[space_key]
            tf_eprec, tf_erec, tf_epr = early_precision_tf_target_from_frames(
                true_df,
                pred_df,
                tf_candidates,
                target_candidates,
            )
            beeline_eprec, beeline_erec, beeline_epr = early_precision_beeline_from_frames(
                true_df,
                pred_df,
                tf_candidates,
                target_candidates,
            )
            metric_cache[metric_key] = (
                eprec,
                erec,
                epr,
                tf_eprec,
                tf_erec,
                tf_epr,
                beeline_eprec,
                beeline_erec,
                beeline_epr,
            )
        (
            eprec,
            erec,
            epr,
            tf_eprec,
            tf_erec,
            tf_epr,
            beeline_eprec,
            beeline_erec,
            beeline_epr,
        ) = metric_cache[metric_key]

        row.update(
            {
                "Eprec": eprec,
                "Erec": erec,
                "EPR": epr,
                "TFTargetEprec": tf_eprec,
                "TFTargetErec": tf_erec,
                "TFTargetEPR": tf_epr,
                "BEELINEEprec": beeline_eprec,
                "BEELINEErec": beeline_erec,
                "BEELINEEPR": beeline_epr,
            }
        )
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    param_cols = [
        col
        for col in [
            "run_name",
            "batch_size",
            "epochs",
            "entropy_lambda",
            "hidden_dim",
            "n_heads",
            "mlp_ratio",
            "inf_n_samples",
            "inf_batch_size",
            "lr",
            "weight_decay",
            "tf_depth",
            "gene_depth",
            "dropout",
            "edge_threshold",
            "device",
        ]
        if col in out_df.columns
    ]

    for metric_col, mean_col in [
        ("EPR", "mean_across_datasets"),
        ("TFTargetEPR", "mean_across_datasets_tftarget"),
        ("BEELINEEPR", "mean_across_datasets_beeline"),
    ]:
        if param_cols:
            combo_means = (
                out_df.groupby(param_cols, dropna=False)[metric_col]
                .mean()
                .reset_index()
                .rename(columns={metric_col: mean_col})
            )
            out_df = out_df.drop(columns=[mean_col], errors="ignore").merge(combo_means, on=param_cols, how="left")

    output_csv = Path(args.output_csv) if args.output_csv else summary_path.with_name("search_summary_three_eprs.csv")
    output_json = output_csv.with_suffix(".json")
    out_df.to_csv(output_csv, index=False)

    payload = {}
    if "run_name" in out_df.columns:
        for metric_col in ("EPR", "TFTargetEPR", "BEELINEEPR"):
            mean_df = out_df.groupby("run_name", as_index=False)[metric_col].mean().sort_values(metric_col, ascending=False)
            if not mean_df.empty:
                best_run_name = mean_df.iloc[0]["run_name"]
                payload[metric_col] = {
                    "best_run_name": best_run_name,
                    "mean_value": float(mean_df.iloc[0][metric_col]),
                    "rows": out_df[out_df["run_name"] == best_run_name].to_dict("records"),
                }
    save_json(str(output_json), payload)
    print(f"[reevaluate] wrote {output_csv}", flush=True)
    print(f"[reevaluate] wrote {output_json}", flush=True)

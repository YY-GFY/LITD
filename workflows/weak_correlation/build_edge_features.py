#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", choices=["hesc", "hhep"], required=True)
    p.add_argument("--target-gene-count", type=int, choices=[500, 1000], required=True)
    return p.parse_args()


def normalized_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x -= x.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.square(x).sum(axis=1, keepdims=True))
    np.divide(x, denom, out=x, where=denom > 0)
    x[~np.isfinite(x)] = 0.0
    return x


def rowwise_spearman_ready(x: np.ndarray) -> np.ndarray:
    return normalized_rows(rankdata(x, axis=1, method="average"))


def assign_bin(values: np.ndarray, bins: dict[str, list[float]]) -> np.ndarray:
    out = np.full(len(values), "unassigned", dtype=object)
    for label, (lower, upper) in bins.items():
        mask = (values >= float(lower)) & (values < float(upper))
        out[mask] = label
    return out


def assign_tertiles(values: np.ndarray) -> np.ndarray:
    """Assign target-level low/medium/high bins with deterministic tie handling."""
    ranks = pd.Series(values).rank(method="average", pct=True).to_numpy()
    out = np.full(len(values), "medium", dtype=object)
    out[ranks <= 1.0 / 3.0] = "low"
    out[ranks > 2.0 / 3.0] = "high"
    return out


def standard_ranked(df: pd.DataFrame, score_col: str, method: str, dataset: str, target_count: int) -> pd.DataFrame:
    out = df[["Gene1", "Gene2", score_col]].rename(columns={score_col: "score"}).copy()
    out["score"] = pd.to_numeric(out["score"], errors="coerce").abs()
    out = out.dropna(subset=["score"])
    out = out.sort_values(["score", "Gene1", "Gene2"], ascending=[False, True, True], kind="mergesort")
    out = out.drop_duplicates(["Gene1", "Gene2"], keep="first").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["method"] = method
    out["dataset"] = dataset
    out["target_gene_count"] = int(target_count)
    return out[["Gene1", "Gene2", "score", "rank", "method", "dataset", "target_gene_count"]]


def main() -> None:
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(cfg["root"])
    ds_cfg = cfg["datasets"][args.dataset]
    count_key = str(args.target_gene_count)

    edge_path = Path(cfg["final_litd_edges"][args.dataset][count_key])
    edges = pd.read_csv(edge_path, usecols=lambda c: c in {"Gene1", "Gene2"})
    edges["Gene1"] = edges["Gene1"].astype(str)
    edges["Gene2"] = edges["Gene2"].astype(str)
    edges = edges[edges["Gene1"] != edges["Gene2"]].drop_duplicates(["Gene1", "Gene2"]).reset_index(drop=True)
    tf_names = pd.Index(edges["Gene1"].drop_duplicates())
    target_names = pd.Index(edges["Gene2"].drop_duplicates())
    needed_genes = set(tf_names).union(set(target_names))

    expr = pd.read_csv(ds_cfg["expression"], index_col=0)
    expr.index = expr.index.astype(str)
    missing_genes = sorted(needed_genes - set(expr.index))
    if missing_genes:
        raise ValueError(f"Expression matrix lacks {len(missing_genes)} candidate genes; first={missing_genes[:10]}")
    pairs = pd.read_csv(ds_cfg["pairs"])
    required_pair_cols = {"source_index", "target_index"}
    if not required_pair_cols.issubset(pairs.columns):
        raise ValueError(f"Missing pair columns: {sorted(required_pair_cols - set(pairs.columns))}")
    source_ids = pairs["source_index"].astype(str).tolist()
    paired_target_ids = pairs["target_index"].astype(str).tolist()
    missing_cells = sorted((set(source_ids) | set(paired_target_ids)) - set(expr.columns.astype(str)))
    if missing_cells:
        raise ValueError(f"Expression matrix lacks {len(missing_cells)} paired cells; first={missing_cells[:10]}")

    expr.columns = expr.columns.astype(str)
    src_tf = expr.loc[tf_names, source_ids].to_numpy(dtype=np.float64, copy=True)
    src_target = expr.loc[target_names, source_ids].to_numpy(dtype=np.float64, copy=True)
    nxt_target = expr.loc[target_names, paired_target_ids].to_numpy(dtype=np.float64, copy=True)

    pearson_static = normalized_rows(src_tf.copy()) @ normalized_rows(src_target.copy()).T
    pearson_next = normalized_rows(src_tf.copy()) @ normalized_rows(nxt_target.copy()).T
    spearman_static = rowwise_spearman_ready(src_tf.copy()) @ rowwise_spearman_ready(src_target.copy()).T
    spearman_next = rowwise_spearman_ready(src_tf.copy()) @ rowwise_spearman_ready(nxt_target.copy()).T

    paired_delta = nxt_target - src_target
    target_de = np.abs(paired_delta.mean(axis=1)) / (paired_delta.std(axis=1, ddof=1) + 1e-8)
    target_expr = np.concatenate([src_target, nxt_target], axis=1).mean(axis=1)
    target_nonzero = (np.concatenate([src_target, nxt_target], axis=1) != 0).mean(axis=1)

    tf_pos = pd.Series(np.arange(len(tf_names)), index=tf_names)
    target_pos = pd.Series(np.arange(len(target_names)), index=target_names)
    i = tf_pos.loc[edges["Gene1"]].to_numpy(dtype=np.int64)
    j = target_pos.loc[edges["Gene2"]].to_numpy(dtype=np.int64)

    features = edges.copy()
    features["corr_pearson_static"] = pearson_static[i, j].astype(np.float32)
    features["abs_corr_pearson_static"] = np.abs(features["corr_pearson_static"].to_numpy()).astype(np.float32)
    features["corr_pearson_next"] = pearson_next[i, j].astype(np.float32)
    features["abs_corr_pearson_next"] = np.abs(features["corr_pearson_next"].to_numpy()).astype(np.float32)
    features["corr_spearman_static"] = spearman_static[i, j].astype(np.float32)
    features["abs_corr_spearman_static"] = np.abs(features["corr_spearman_static"].to_numpy()).astype(np.float32)
    features["corr_spearman_next"] = spearman_next[i, j].astype(np.float32)
    features["abs_corr_spearman_next"] = np.abs(features["corr_spearman_next"].to_numpy()).astype(np.float32)
    features["target_temporal_de_strength"] = target_de[j].astype(np.float32)
    features["target_mean_expression"] = target_expr[j].astype(np.float32)
    features["target_nonzero_fraction"] = target_nonzero[j].astype(np.float32)
    features["static_corr_bin"] = assign_bin(features["abs_corr_pearson_static"].to_numpy(), cfg["correlation_bins"])
    features["next_corr_bin"] = assign_bin(features["abs_corr_pearson_next"].to_numpy(), cfg["correlation_bins"])
    target_de_bins = assign_tertiles(target_de)
    target_expr_bins = assign_tertiles(target_expr)
    features["target_temporal_de_bin"] = target_de_bins[j]
    features["target_expression_bin"] = target_expr_bins[j]
    features["dataset"] = args.dataset
    features["target_gene_count"] = int(args.target_gene_count)

    feature_dir = root / "results" / "edge_features" / args.dataset
    ranked_dir = root / "results" / "ranked_edges"
    feature_dir.mkdir(parents=True, exist_ok=True)
    ranked_dir.mkdir(parents=True, exist_ok=True)
    feature_out = feature_dir / f"{args.dataset}_tgt{args.target_gene_count}_edge_features.csv.gz"
    features.to_csv(feature_out, index=False, compression={"method": "gzip", "compresslevel": 1})

    for method, score_col in [("Pearson", "abs_corr_pearson_static"), ("Spearman", "abs_corr_spearman_static")]:
        out_dir = ranked_dir / method / args.dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        ranked = standard_ranked(features, score_col, method, args.dataset, args.target_gene_count)
        ranked.to_csv(out_dir / f"{args.dataset}_tgt{args.target_gene_count}_ranked_edges.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 1})

    metadata = {
        "dataset": args.dataset,
        "target_gene_count": int(args.target_gene_count),
        "canonical_pairs_path": ds_cfg["pairs"],
        "pair_count": int(len(pairs)),
        "candidate_edge_count": int(len(features)),
        "tf_count": int(len(tf_names)),
        "target_count": int(len(target_names)),
        "weak_edge_count": int((features["static_corr_bin"] == "weak").sum()),
        "medium_edge_count": int((features["static_corr_bin"] == "medium").sum()),
        "strong_edge_count": int((features["static_corr_bin"] == "strong").sum()),
        "secondary_stratifiers": ["next_corr_bin", "target_temporal_de_bin", "target_expression_bin"],
        "correlation_definition": "Pearson/Spearman across the existing paired rows; static uses source TF and source target, next uses source TF and already-paired target target.",
        "expression_transform": "none; canonical ExpressionData.csv values used directly"
    }
    (feature_dir / f"{args.dataset}_tgt{args.target_gene_count}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

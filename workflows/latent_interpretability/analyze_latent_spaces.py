#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
import umap

from interpretability_common import WINDOWS, ensure_output_dirs, load_config, safe_key, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def fit_umap(x, cfg, seed_offset=0):
    if not np.isfinite(x).all():
        raise ValueError("UMAP input contains non-finite values")
    n_neighbors = min(int(cfg["umap_n_neighbors"]), max(2, len(x) - 1))
    return umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=float(cfg["umap_min_dist"]),
        metric=cfg["umap_metric"], random_state=int(cfg["seed"]) + seed_offset,
        init="random" if len(x) < 10 else "spectral",
    ).fit_transform(x)


def balanced_plot_indices(meta, cap, seed):
    if len(meta) <= cap:
        return np.arange(len(meta))
    rng = np.random.default_rng(seed)
    groups = list(meta.groupby(["window", "experiment"], sort=True).groups.values())
    quota = max(1, cap // len(groups))
    selected = []
    for group in groups:
        candidates = np.asarray(list(group), dtype=int)
        rng.shuffle(candidates)
        selected.extend(candidates[:quota].tolist())
    if len(selected) < cap:
        remaining = np.setdiff1d(np.arange(len(meta)), np.asarray(selected))
        rng.shuffle(remaining)
        selected.extend(remaining[:cap - len(selected)].tolist())
    return np.asarray(sorted(selected[:cap]), dtype=int)


def linear_cka(x, y):
    x = np.asarray(x, dtype=np.float64) - np.asarray(x, dtype=np.float64).mean(0, keepdims=True)
    y = np.asarray(y, dtype=np.float64) - np.asarray(y, dtype=np.float64).mean(0, keepdims=True)
    numerator = np.square(x.T @ y).sum()
    denominator = np.sqrt(np.square(x.T @ x).sum() * np.square(y.T @ y).sum())
    return float(numerator / denominator) if denominator > 0 else 0.0


def experiment_neighbor_entropy(x, labels, k):
    if len(x) < 3:
        return None
    k = min(k, len(x) - 1)
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(x).kneighbors(x, return_distance=False)[:, 1:]
    labels = np.asarray(labels)
    values = []
    for row in neighbors:
        _, counts = np.unique(labels[row], return_counts=True)
        p = counts / counts.sum()
        values.append(float(-(p * np.log(p + 1e-12)).sum() / np.log(k)) if k > 1 else 0.0)
    return float(np.mean(values))


def representation_metrics(x, meta, layer, cfg, silhouette_idx):
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite values in {layer}")
    rows = []
    stage = meta["window"].astype(str).to_numpy()
    day = meta["day_source"].to_numpy(float)
    metric_x = x[silhouette_idx]
    metric_stage = stage[silhouette_idx]
    metric_meta = meta.iloc[silhouette_idx]
    counts = pd.Series(metric_stage).value_counts()
    if len(counts) > 1 and counts.min() >= 2:
        rows.append({"layer": layer, "metric": "stage_silhouette",
                     "value": float(silhouette_score(metric_x, metric_stage, metric="cosine")), "status": "ok"})
        folds = min(5, int(counts.min()))
        model = KNeighborsClassifier(n_neighbors=min(15, max(1, len(metric_x) // 10)), metric="cosine")
        pred = cross_val_predict(model, metric_x, metric_stage, cv=StratifiedKFold(folds, shuffle=True, random_state=int(cfg["seed"])))
        rows.append({"layer": layer, "metric": "balanced_knn_stage_accuracy", "value": float(balanced_accuracy_score(metric_stage, pred)), "status": "ok"})
    else:
        for metric in ("stage_silhouette", "balanced_knn_stage_accuracy"):
            rows.append({"layer": layer, "metric": metric, "value": None, "status": "insufficient_samples"})
    if len(x) >= 3:
        pc1 = PCA(n_components=1, random_state=int(cfg["seed"])).fit_transform(StandardScaler().fit_transform(x)).ravel()
        rho = float(abs(spearmanr(pc1, day).statistic))
        rows.append({"layer": layer, "metric": "abs_pc1_day_spearman", "value": rho, "status": "ok"})
        rows.append({"layer": layer, "metric": "experiment_neighbor_entropy",
                     "value": experiment_neighbor_entropy(metric_x, metric_meta["experiment"].astype(str), int(cfg["neighbor_k"])), "status": "ok"})
    else:
        for metric in ("abs_pc1_day_spearman", "experiment_neighbor_entropy"):
            rows.append({"layer": layer, "metric": metric, "value": None, "status": "insufficient_samples"})
    return rows


def main():
    cfg = load_config(parse_args().config)
    paths = ensure_output_dirs(cfg)
    meta = pd.read_parquet(paths["export"] / "cell_metadata.parquet")
    cells = np.load(paths["export"] / "cell_embeddings.npz", allow_pickle=False)
    tokens = np.load(paths["export"] / "token_means.npz", allow_pickle=False)
    plot_idx = balanced_plot_indices(meta, int(cfg["umap_max_cells"]), int(cfg["seed"]))
    metric_idx = balanced_plot_indices(meta, int(cfg["metric_max_cells"]), int(cfg["seed"]) + 1)
    cka_idx = balanced_plot_indices(meta, int(cfg["cka_max_cells"]), int(cfg["seed"]) + 2)

    preferred = ["raw_gene", "projected_gene", "rec_pre_film", "rec_post_film", "tf_cross", "tf_layer1", "tf_final", "gene_cross", "gene_final"]
    layers = [layer for layer in preferred if safe_key("real", layer) in cells.files]
    metrics, coordinates = [], []
    for offset, layer in enumerate(layers):
        values = cells[safe_key("real", layer)]
        metrics.extend(representation_metrics(values, meta, layer, cfg, metric_idx))
        xy = fit_umap(values[plot_idx], cfg, offset)
        frame = meta.iloc[plot_idx].reset_index(drop=True).copy()
        frame["layer"], frame["UMAP1"], frame["UMAP2"] = layer, xy[:, 0], xy[:, 1]
        coordinates.append(frame)
    pd.concat(coordinates, ignore_index=True).to_csv(paths["coordinates"] / "cell_umap_coordinates.csv", index=False)
    pd.DataFrame(metrics).to_csv(paths["metrics"] / "cell_representation_metrics.csv", index=False)

    cka = []
    for left, right in itertools.product(layers, layers):
        cka.append({"condition": "real", "layer_left": left, "layer_right": right,
                    "cka": linear_cka(cells[safe_key("real", left)][cka_idx], cells[safe_key("real", right)][cka_idx])})
    for layer in ("rec_post_film", "tf_final", "gene_final"):
        for condition in cfg["conditions"]:
            cka.append({"condition": condition, "layer_left": f"real:{layer}", "layer_right": f"{condition}:{layer}",
                        "cka": linear_cka(cells[safe_key("real", layer)][cka_idx], cells[safe_key(condition, layer)][cka_idx])})
    pd.DataFrame(cka).to_csv(paths["metrics"] / "layer_cka.csv", index=False)

    receptor_names = tokens["receptor_names"].astype(str)
    displacement = []
    for _a, _b, window in WINDOWS:
        pre = tokens[safe_key("real", window, "rec_pre_film")]
        real = tokens[safe_key("real", window, "rec_post_film")]
        zero = tokens[safe_key("zero_all_ligands", window, "rec_post_film")]
        mismatch = tokens[safe_key("stage_mismatch", window, "rec_post_film")]
        for i, name in enumerate(receptor_names):
            displacement.append({"window": window, "receptor": name,
                                 "post_vs_pre_l2": float(np.linalg.norm(real[i] - pre[i])),
                                 "real_vs_zero_l2": float(np.linalg.norm(real[i] - zero[i])),
                                 "real_vs_mismatch_l2": float(np.linalg.norm(real[i] - mismatch[i]))})
    pd.DataFrame(displacement).to_csv(paths["metrics"] / "receptor_displacement.csv", index=False)

    tf_names = tokens["tf_names"].astype(str)
    tf_joint = np.concatenate([tokens[safe_key("real", window, "tf_final")] for _a, _b, window in WINDOWS])
    tf_xy = PCA(n_components=2, random_state=int(cfg["seed"])).fit_transform(tf_joint)
    tf_umap = fit_umap(tf_joint, cfg, 50)
    tf_coords, tf_umap_coords, tf_metrics = [], [], []
    for w, (_a, _b, window) in enumerate(WINDOWS):
        block = tf_xy[w * len(tf_names):(w + 1) * len(tf_names)]
        for i, name in enumerate(tf_names):
            tf_coords.append({"window": window, "tf": name, "PC1": block[i, 0], "PC2": block[i, 1]})
            pos = tf_umap[w * len(tf_names) + i]
            tf_umap_coords.append({"window": window, "tf": name, "UMAP1": pos[0], "UMAP2": pos[1]})
    for i, name in enumerate(tf_names):
        points = np.stack([tf_xy[w * len(tf_names) + i] for w in range(len(WINDOWS))])
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        tf_metrics.append({"tf": name, "trajectory_length": float(steps.sum()), "step1": float(steps[0]),
                           "step2": float(steps[1])})
    pd.DataFrame(tf_coords).to_csv(paths["coordinates"] / "tf_pca_trajectories.csv", index=False)
    pd.DataFrame(tf_umap_coords).to_csv(paths["coordinates"] / "tf_umap_trajectories.csv", index=False)
    pd.DataFrame(tf_metrics).to_csv(paths["metrics"] / "tf_trajectory_metrics.csv", index=False)

    gene_names = tokens["gene_names"].astype(str)
    gene_cap = min(int(cfg["gene_umap_max_genes"]), len(gene_names))
    rng = np.random.default_rng(int(cfg["seed"]))
    gene_idx = np.arange(len(gene_names))
    rng.shuffle(gene_idx)
    gene_idx = np.sort(gene_idx[:gene_cap])
    gene_joint = np.concatenate([tokens[safe_key("real", window, "gene_final")][gene_idx] for _a, _b, window in WINDOWS])
    gene_xy = fit_umap(gene_joint, cfg, 100)
    gene_coords = []
    for w, (_a, _b, window) in enumerate(WINDOWS):
        block = gene_xy[w * len(gene_idx):(w + 1) * len(gene_idx)]
        for local, original in enumerate(gene_idx):
            gene_coords.append({"window": window, "gene": gene_names[original],
                                "UMAP1": block[local, 0], "UMAP2": block[local, 1]})
    pd.DataFrame(gene_coords).to_csv(paths["coordinates"] / "gene_umap_coordinates.csv", index=False)

    write_json(paths["analysis"] / "analysis_metadata.json",
               {"analysis_name": cfg["analysis_name"], "n_cells": len(meta), "umap_cells": len(plot_idx),
                "gene_umap_genes_per_window": len(gene_idx), "metric_cells": len(metric_idx),
                "cka_cells": len(cka_idx), "real_layers": layers,
                "interpretation": "Learned representation organization and model-internal evidence; not causal proof."})
    print(f"[analyze] completed: {paths['analysis']}")


if __name__ == "__main__":
    main()

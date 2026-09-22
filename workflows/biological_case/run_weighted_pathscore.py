#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from case_common import DEFAULT_CONFIG_PATH, ensure_dir, load_config
from paper_analysis_common import (
    WINDOWS,
    build_positive_lr_table,
    load_checkpointed_model,
    load_real_window_bundles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact all-ligand, all-TF, all-gene ranking by three-window weighted PathScore."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def assert_aligned(bundles: dict[str, dict[str, object]]) -> tuple[list[str], list[str], list[str]]:
    first = bundles[WINDOWS[0].label]
    receptors = first["receptor_to_tf"].index.astype(str).tolist()
    tfs = first["receptor_to_tf"].columns.astype(str).tolist()
    genes = first["tf_to_gene"].index.astype(str).tolist()
    for window in WINDOWS:
        bundle = bundles[window.label]
        assert bundle["receptor_to_tf"].index.astype(str).tolist() == receptors
        assert bundle["receptor_to_tf"].columns.astype(str).tolist() == tfs
        assert bundle["tf_to_gene"].columns.astype(str).tolist() == tfs
        assert bundle["tf_to_gene"].index.astype(str).tolist() == genes
    return receptors, tfs, genes


def brute_force_validation(
    coeff: np.ndarray,
    tg: np.ndarray,
) -> dict[str, object]:
    n_edges = min(4, coeff.shape[0])
    n_tfs = min(3, coeff.shape[1])
    n_genes = min(7, tg.shape[1])
    matrix_scores = []
    loop_scores = []
    for tf_idx in range(n_tfs):
        fast = coeff[:n_edges, tf_idx, :] @ tg[:, :n_genes, tf_idx]
        matrix_scores.append(fast)
        slow = np.empty((n_edges, n_genes), dtype=np.float64)
        for edge_idx in range(n_edges):
            for gene_idx in range(n_genes):
                value = 0.0
                for window_idx in range(tg.shape[0]):
                    value += coeff[edge_idx, tf_idx, window_idx] * tg[window_idx, gene_idx, tf_idx]
                slow[edge_idx, gene_idx] = value
        loop_scores.append(slow)
    matrix_arr = np.stack(matrix_scores, axis=0)
    loop_arr = np.stack(loop_scores, axis=0)
    max_abs_error = float(np.max(np.abs(matrix_arr - loop_arr)))
    if not np.allclose(matrix_arr, loop_arr, rtol=0.0, atol=1e-15):
        raise AssertionError(f"Matrix and brute-force scores differ: max_abs_error={max_abs_error}")
    return {
        "n_edges": n_edges,
        "n_tfs": n_tfs,
        "n_genes": n_genes,
        "max_abs_error": max_abs_error,
        "passed": True,
    }


def path_components(
    ligand: str,
    receptor: str,
    tf: str,
    gene: str,
    bundles: dict[str, dict[str, object]],
    positive_lr: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    pos_lr = float(positive_lr.at[ligand, receptor])
    rows = []
    for window in WINDOWS:
        label = window.label
        bundle = bundles[label]
        ligand_expr = float(bundle["ligand_context"].at[ligand])
        attn_rt = float(bundle["receptor_to_tf"].at[receptor, tf])
        attn_tg = float(bundle["tf_to_gene"].at[gene, tf])
        score = ligand_expr * pos_lr * attn_rt * attn_tg
        rows.append(
            {
                "window_label": label,
                "n_samples": int(bundle["n_samples"]),
                "window_weight": weights[label],
                "ligand_expr": ligand_expr,
                "positive_lr": pos_lr,
                "attn_rt": attn_rt,
                "attn_tg": attn_tg,
                "path_score_window": score,
                "weighted_contribution": weights[label] * score,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    bundles = load_real_window_bundles(cfg)
    receptor_names, tf_names, gene_names = assert_aligned(bundles)
    dataset, model, _model_device = load_checkpointed_model(cfg, device_override="cpu")
    positive_lr = build_positive_lr_table(model, cfg)

    ligand_names = positive_lr.index.astype(str).tolist()
    if len(ligand_names) != len(bundles[WINDOWS[0].label]["ligand_context"]):
        raise AssertionError("Ligand list and window ligand context have different lengths")
    if positive_lr.columns.astype(str).tolist() != receptor_names:
        positive_lr = positive_lr.reindex(columns=receptor_names)
    if positive_lr.isna().any().any():
        raise AssertionError("Positive LR table could not be aligned to receptor names")

    lr_values = positive_lr.to_numpy(dtype=np.float64)
    edge_lig_idx, edge_rec_idx = np.where(lr_values > 0.0)
    edge_plr = lr_values[edge_lig_idx, edge_rec_idx]
    if len(edge_plr) == 0:
        raise ValueError("No positive learned LR edges were found")

    total_samples = int(sum(int(bundles[w.label]["n_samples"]) for w in WINDOWS))
    window_weights = {
        w.label: int(bundles[w.label]["n_samples"]) / total_samples for w in WINDOWS
    }
    n_windows = len(WINDOWS)
    n_edges = len(edge_plr)
    n_tfs = len(tf_names)
    n_genes = len(gene_names)

    # coeff[e, t, w] = p_w * E_w(L_e) * P_LR(e) * A_RT,w(R_e, t)
    coeff = np.empty((n_edges, n_tfs, n_windows), dtype=np.float64)
    tg = np.empty((n_windows, n_genes, n_tfs), dtype=np.float64)
    for window_idx, window in enumerate(WINDOWS):
        label = window.label
        bundle = bundles[label]
        expr = bundle["ligand_context"].reindex(ligand_names).to_numpy(dtype=np.float64)
        rt = bundle["receptor_to_tf"].to_numpy(dtype=np.float64)
        coeff[:, :, window_idx] = (
            window_weights[label]
            * edge_plr[:, None]
            * expr[edge_lig_idx, None]
            * rt[edge_rec_idx, :]
        )
        tg[window_idx, :, :] = bundle["tf_to_gene"].to_numpy(dtype=np.float64)

    validation_small = brute_force_validation(coeff, tg)

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    top_k = int(args.top_k)

    candidates: list[dict[str, object]] = []
    coeff_t = torch.from_numpy(coeff).to(device=device, dtype=torch.float64)
    tg_t = torch.from_numpy(tg).to(device=device, dtype=torch.float64)
    for tf_idx, tf in enumerate(tf_names):
        # [positive LR edges, 3] @ [3, genes] -> [positive LR edges, genes]
        scores = coeff_t[:, tf_idx, :] @ tg_t[:, :, tf_idx]
        k = min(top_k, scores.numel())
        values, flat_indices = torch.topk(scores.reshape(-1), k=k, largest=True, sorted=True)
        edge_indices = torch.div(flat_indices, n_genes, rounding_mode="floor").cpu().numpy()
        gene_indices = torch.remainder(flat_indices, n_genes).cpu().numpy()
        values_np = values.cpu().numpy()
        for value, edge_idx, gene_idx in zip(values_np, edge_indices, gene_indices):
            ligand_idx = int(edge_lig_idx[edge_idx])
            receptor_idx = int(edge_rec_idx[edge_idx])
            candidates.append(
                {
                    "ligand": ligand_names[ligand_idx],
                    "receptor": receptor_names[receptor_idx],
                    "tf": tf,
                    "target_gene": gene_names[int(gene_idx)],
                    "global_weighted_pathscore": float(value),
                    "positive_lr": float(edge_plr[edge_idx]),
                }
            )
        del scores, values, flat_indices

    ranked = (
        pd.DataFrame(candidates)
        .sort_values("global_weighted_pathscore", ascending=False, kind="stable")
        .head(top_k)
        .reset_index(drop=True)
    )
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    if ranked.duplicated(["ligand", "receptor", "tf", "target_gene"]).any():
        raise AssertionError("Duplicate paths found in global Top-K")
    if not ranked["global_weighted_pathscore"].is_monotonic_decreasing:
        raise AssertionError("Global Top-K is not monotonically decreasing")

    ranked.to_csv(out_dir / "global_weighted_pathscore_top1000.csv", index=False)

    winner = ranked.iloc[0]
    winner_components = path_components(
        str(winner.ligand),
        str(winner.receptor),
        str(winner.tf),
        str(winner.target_gene),
        bundles,
        positive_lr,
        window_weights,
    )
    winner_components.insert(0, "target_gene", str(winner.target_gene))
    winner_components.insert(0, "tf", str(winner.tf))
    winner_components.insert(0, "receptor", str(winner.receptor))
    winner_components.insert(0, "ligand", str(winner.ligand))
    winner_components.to_csv(out_dir / "winner_three_window_decomposition.csv", index=False)

    window_scores = winner_components[["window_label", "path_score_window"]].copy()
    max_score = float(window_scores["path_score_window"].max())
    sorted_scores = np.sort(window_scores["path_score_window"].to_numpy(dtype=np.float64))[::-1]
    second_score = float(sorted_scores[1])
    other_mean = float((sorted_scores[1] + sorted_scores[2]) / 2.0)
    window_scores["relative_to_peak"] = window_scores["path_score_window"] / max_score
    window_scores["score_fraction"] = window_scores["path_score_window"] / float(
        window_scores["path_score_window"].sum()
    )
    window_scores["peak_window"] = window_scores.loc[
        window_scores["path_score_window"].idxmax(), "window_label"
    ]
    window_scores["peak_to_second_ratio"] = max_score / second_score
    window_scores["peak_to_other_mean_ratio"] = max_score / other_mean
    window_scores.to_csv(out_dir / "per_window_pathscore.csv", index=False)

    hub_rows = []
    for column in ["ligand", "receptor", "tf"]:
        counts = ranked[column].value_counts(dropna=False)
        for value, count in counts.items():
            hub_rows.append(
                {
                    "level": column,
                    "name": value,
                    "count_in_top1000": int(count),
                    "fraction_in_top1000": float(count / len(ranked)),
                }
            )
    pd.DataFrame(hub_rows).sort_values(
        ["level", "count_in_top1000", "name"], ascending=[True, False, True]
    ).to_csv(out_dir / "top1000_hub_summary.csv", index=False)

    winner_recomputed = float(winner_components["weighted_contribution"].sum())
    winner_saved = float(winner.global_weighted_pathscore)
    winner_error = abs(winner_recomputed - winner_saved)
    if winner_error > 1e-12:
        raise AssertionError(f"Winner weighted decomposition mismatch: {winner_error}")

    validation_report = {
        "small_bruteforce": validation_small,
        "winner_weighted_sum_error": winner_error,
        "winner_weighted_sum_tolerance": 1e-12,
        "top1000_monotonic": True,
        "top1000_unique": True,
        "all_output_lr_edges_positive": bool((ranked["positive_lr"] > 0).all()),
        "n_tfs_expected_and_observed": [579, n_tfs],
        "n_genes_expected_and_observed": [10112, n_genes],
        "passed": True,
    }
    (out_dir / "validation_report.json").write_text(
        json.dumps(validation_report, indent=2, ensure_ascii=False) + "\n"
    )

    metadata = {
        "analysis": "all-ligand all-579-TF all-10112-gene exact three-window weighted PathScore ranking",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(cfg.checkpoint_path),
        "formula": "sum_w (n_w/N) * E_w(L) * positive_LR(L,R) * AttnRT_w(R,T) * AttnTG_w(T,G)",
        "ligand_expression_transform": "none; raw window artifact mean",
        "positive_lr_definition": "max(0, learned_lr_weight * binary_lr_prior_mask)",
        "n_ligands": len(ligand_names),
        "n_receptors": len(receptor_names),
        "n_tfs": n_tfs,
        "n_genes": n_genes,
        "n_binary_prior_edges": int(np.count_nonzero(model.lr_layer.mask.detach().cpu().numpy())),
        "n_positive_learned_lr_edges": n_edges,
        "n_candidate_paths": int(n_edges * n_tfs * n_genes),
        "total_samples": total_samples,
        "window_samples": {w.label: int(bundles[w.label]["n_samples"]) for w in WINDOWS},
        "window_weights": window_weights,
        "device": str(device),
        "dtype": "float64",
        "top_k_saved": top_k,
        "external_evidence_used_for_ranking": False,
        "winner": {
            "ligand": str(winner.ligand),
            "receptor": str(winner.receptor),
            "tf": str(winner.tf),
            "target_gene": str(winner.target_gene),
            "global_weighted_pathscore": winner_saved,
        },
    }
    (out_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("\nTop 20 paths:")
    print(ranked.head(20).to_string(index=False))
    print("\nWinner decomposition:")
    print(winner_components.to_string(index=False))


if __name__ == "__main__":
    main()

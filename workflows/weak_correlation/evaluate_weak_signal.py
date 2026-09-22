#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", choices=["hesc", "hhep"], required=True)
    p.add_argument("--target-gene-count", type=int, choices=[500, 1000], required=True)
    return p.parse_args()


def gt_path(cfg: dict, dataset: str, network: str) -> str:
    return cfg["datasets"][dataset]["chip"] if network == "ChIP" else cfg["ground_truth"][network]


def clean_gt(path: str) -> set[str]:
    d = pd.read_csv(path, usecols=["Gene1", "Gene2"])
    d["Gene1"] = d["Gene1"].astype(str)
    d["Gene2"] = d["Gene2"].astype(str)
    d = d[d["Gene1"] != d["Gene2"]].drop_duplicates(["Gene1", "Gene2"])
    return set(d["Gene1"] + "|" + d["Gene2"])


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    positives = int(y.sum())
    if positives == 0:
        return np.nan
    return float(average_precision_score(y, score))


def selection_weights_at_k(score: np.ndarray, k: int) -> np.ndarray:
    """Return tie-aware expected top-k membership weights."""
    n = len(score)
    k = min(max(int(k), 0), n)
    weights = np.zeros(n, dtype=float)
    if k == 0:
        return weights
    if k == n:
        weights[:] = 1.0
        return weights
    threshold = np.partition(score, n - k)[n - k]
    above = score > threshold
    tied = score == threshold
    weights[above] = 1.0
    remaining = k - int(above.sum())
    tied_n = int(tied.sum())
    if remaining > 0 and tied_n:
        weights[tied] = remaining / tied_n
    return weights


def deterministic_exact_topk_weights(
    score: np.ndarray,
    gene1: np.ndarray,
    gene2: np.ndarray,
    k: int,
) -> np.ndarray:
    """Apply the paper's exact-topK rule: score desc, Gene1/Gene2 asc."""
    n = len(score)
    k = min(max(int(k), 0), n)
    weights = np.zeros(n, dtype=float)
    if k == 0:
        return weights
    order = np.lexsort((gene2.astype(str), gene1.astype(str), -score))
    weights[order[:k]] = 1.0
    return weights


def metric_row(
    method: str,
    dataset: str,
    target_count: int,
    network: str,
    stratifier: str,
    stratum: str,
    y: np.ndarray,
    score: np.ndarray,
    gene1: np.ndarray,
    gene2: np.ndarray,
) -> dict:
    n = int(len(y))
    p = int(y.sum())
    prevalence = float(p / n) if n else np.nan
    auprc = average_precision(y, score) if n else np.nan
    k = min(p, n)
    # Primary EPR follows the reported evaluator exactly: K is the number of
    # gold edges in the evaluated space and ties are resolved deterministically.
    weights = deterministic_exact_topk_weights(score, gene1, gene2, k)
    hits = float(np.dot(y, weights)) if k else 0.0
    precision_k = float(hits / k) if k else np.nan
    recall_k = float(hits / p) if p else np.nan
    return {
        "method": method,
        "dataset": dataset,
        "target_gene_count": int(target_count),
        "gene_set": f"TFs+{target_count}",
        "network": network,
        "stratifier": stratifier,
        "stratum": stratum,
        "candidate_edge_count": n,
        "gold_edge_count": p,
        "gold_prevalence": prevalence,
        "AUPRC": auprc,
        "AUPRC_ratio": float(auprc / prevalence) if prevalence and np.isfinite(auprc) else np.nan,
        "within_stratum_K": int(k),
        "within_stratum_expected_hits_at_K": hits,
        "within_stratum_precision_at_K": precision_k,
        "within_stratum_recall_at_K": recall_k,
        "within_stratum_EPR": float(precision_k / prevalence) if prevalence and np.isfinite(precision_k) else np.nan,
        "within_stratum_selection_mode": "exact_topk",
        "within_stratum_tie_policy": "deterministic_sort_score_desc_Gene1_Gene2_head_k",
    }


def main() -> None:
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(cfg["root"])
    feature_path = root / "results" / "edge_features" / args.dataset / f"{args.dataset}_tgt{args.target_gene_count}_edge_features.csv.gz"
    stratifier_columns = {
        "static_correlation": "static_corr_bin",
        "next_correlation": "next_corr_bin",
        "target_temporal_DE": "target_temporal_de_bin",
        "target_expression": "target_expression_bin",
    }
    features = pd.read_csv(feature_path, usecols=["Gene1", "Gene2", *stratifier_columns.values()])
    features["Gene1"] = features["Gene1"].astype(str)
    features["Gene2"] = features["Gene2"].astype(str)
    features["edge_id"] = features["Gene1"] + "|" + features["Gene2"]
    candidate_set = set(features["edge_id"])

    method_root = root / "results" / "ranked_edges"
    method_files: dict[str, Path] = {}
    for method_dir in sorted(method_root.iterdir() if method_root.exists() else []):
        if not method_dir.is_dir():
            continue
        candidates = [
            method_dir / args.dataset / f"{args.dataset}_tgt{args.target_gene_count}_ranked_edges.csv.gz",
            method_dir / args.dataset / f"{args.dataset}_tgt{args.target_gene_count}_ranked_edges.csv",
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found:
            method_files[method_dir.name] = found
    missing_methods = sorted(set(cfg.get("expected_methods", [])) - set(method_files))
    if missing_methods:
        raise RuntimeError(f"Incomplete paper comparison; missing methods: {missing_methods}")
    if cfg.get("expected_methods"):
        method_files = {name: method_files[name] for name in cfg["expected_methods"]}
    if not method_files:
        raise RuntimeError("No ranked edge files found")

    metric_rows: list[dict] = []
    budget_rows: list[dict] = []
    coverage_rows: list[dict] = []
    strata_specs = [("all", "all", np.ones(len(features), dtype=bool))]
    for stratifier, column in stratifier_columns.items():
        for stratum in pd.unique(features[column].dropna()):
            strata_specs.append((stratifier, str(stratum), features[column].to_numpy() == stratum))
    for method, path in method_files.items():
        ranked = pd.read_csv(path)
        score_col = next((c for c in ["score", "EdgeWeightAbs", "EdgeWeight", "importance", "weight"] if c in ranked.columns), None)
        if score_col is None:
            raise ValueError(f"No score column in {path}")
        ranked["Gene1"] = ranked["Gene1"].astype(str)
        ranked["Gene2"] = ranked["Gene2"].astype(str)
        ranked["edge_id"] = ranked["Gene1"] + "|" + ranked["Gene2"]
        ranked["score"] = pd.to_numeric(ranked[score_col], errors="coerce").abs()
        ranked = ranked.dropna(subset=["score"]).drop_duplicates("edge_id", keep="first")
        score_map = ranked.set_index("edge_id")["score"]
        scores = features["edge_id"].map(score_map).fillna(0.0).to_numpy(dtype=float)
        coverage_rows.append({
            "method": method,
            "dataset": args.dataset,
            "target_gene_count": int(args.target_gene_count),
            "candidate_edge_count": int(len(features)),
            "scored_edge_count": int(features["edge_id"].isin(set(ranked["edge_id"])).sum()),
            "missing_score_count_filled_zero": int((~features["edge_id"].isin(set(ranked["edge_id"]))).sum()),
            "extra_scored_edges_outside_candidate_space": int((~ranked["edge_id"].isin(candidate_set)).sum()),
            "ranked_edge_path": str(path),
        })

        for network in cfg["networks"]:
            gt = clean_gt(gt_path(cfg, args.dataset, network))
            y_all = features["edge_id"].isin(gt).to_numpy(dtype=np.int8)
            total_gold = int(y_all.sum())
            gold_budget_weight = selection_weights_at_k(scores, total_gold)
            for stratifier, stratum, mask in strata_specs:
                row = metric_row(
                    method,
                    args.dataset,
                    args.target_gene_count,
                    network,
                    stratifier,
                    stratum,
                    y_all[mask],
                    scores[mask],
                    features.loc[mask, "Gene1"].to_numpy(dtype=str),
                    features.loc[mask, "Gene2"].to_numpy(dtype=str),
                )
                if stratifier != "all" and total_gold:
                    global_k = total_gold
                    stratum_gold = int((y_all & mask).sum())
                    recovered = float(np.dot((y_all & mask).astype(float), gold_budget_weight))
                    row["global_gold_budget_K"] = global_k
                    row["global_budget_stratum_gold_recovered"] = recovered
                    row["global_budget_stratum_recall"] = float(recovered / stratum_gold) if stratum_gold else np.nan
                else:
                    row["global_gold_budget_K"] = total_gold
                    recovered = float(np.dot(y_all.astype(float), gold_budget_weight)) if total_gold else 0.0
                    row["global_budget_stratum_gold_recovered"] = recovered
                    row["global_budget_stratum_recall"] = float(recovered / total_gold) if total_gold else np.nan
                metric_rows.append(row)

            budget_weights = {
                min(int(b), len(scores)): selection_weights_at_k(scores, min(int(b), len(scores)))
                for b in [*cfg["fixed_edge_budgets"], total_gold]
            }
            for budget in [*cfg["fixed_edge_budgets"], total_gold]:
                k = min(int(budget), len(scores))
                selected_weight = budget_weights[k]
                for stratifier, stratum, mask in strata_specs:
                    denom = int((y_all & mask).sum())
                    hits = float(np.dot((y_all & mask).astype(float), selected_weight))
                    budget_rows.append({
                        "method": method,
                        "dataset": args.dataset,
                        "target_gene_count": int(args.target_gene_count),
                        "network": network,
                        "stratifier": stratifier,
                        "stratum": stratum,
                        "budget_label": "gold_count" if int(budget) == total_gold else f"fixed_{budget}",
                        "K": k,
                        "gold_edge_count_in_stratum": denom,
                        "expected_gold_hits": hits,
                        "recall_at_K": float(hits / denom) if denom else np.nan,
                    })
        print(f"[evaluate] method={method} dataset={args.dataset} tgt={args.target_gene_count}", flush=True)

    out_dir = root / "results" / "metrics" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out_dir / f"{args.dataset}_tgt{args.target_gene_count}_weak_signal_metrics.csv", index=False)
    pd.DataFrame(budget_rows).to_csv(out_dir / f"{args.dataset}_tgt{args.target_gene_count}_recall_at_fixed_budget.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(out_dir / f"{args.dataset}_tgt{args.target_gene_count}_score_coverage.csv", index=False)
    print(f"[done] methods={len(method_files)} metrics={len(metric_rows)}", flush=True)


if __name__ == "__main__":
    main()

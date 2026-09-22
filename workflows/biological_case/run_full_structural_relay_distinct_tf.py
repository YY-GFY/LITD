#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from relay_search_common import positive_log, retain_global_base_topk, topk_flat


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_SCRIPTS = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "litd_biological_case.json"


@dataclass(frozen=True)
class TerminalPath:
    log_score: float
    edge_idx: int
    tf_idx: int
    gene_idx: int


@dataclass(frozen=True)
class ContinuationPath:
    log_score: float
    segment2_log_score: float
    edge1_idx: int
    tf1_idx: int
    relay2_idx: int
    terminal: TerminalPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact full structural relay Top-K with TF0, TF1, and TF2 pairwise distinct."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--source-scripts", default=str(DEFAULT_SOURCE_SCRIPTS))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--terminal-gene-top-k", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def distinct_tf_synthetic_validation(k: int = 20) -> dict[str, Any]:
    rng = np.random.default_rng(20260817)
    n_ligands, n_receptors, n_tfs, n_genes = 3, 2, 4, 5
    lr = rng.uniform(0.05, 0.9, size=(n_ligands, n_receptors))
    expr = rng.uniform(0.1, 3.0, size=(3, n_ligands))
    rt = rng.uniform(0.02, 0.8, size=(3, n_receptors, n_tfs))
    tg_lig = rng.uniform(0.02, 0.8, size=(3, n_ligands, n_tfs))
    tg_gene = rng.uniform(0.02, 0.8, size=(n_genes, n_tfs))

    explicit = []
    for l0 in range(n_ligands):
        for r0 in range(n_receptors):
            for t0 in range(n_tfs):
                for l1 in range(n_ligands):
                    s1 = expr[0, l0] * lr[l0, r0] * rt[0, r0, t0] * tg_lig[0, l1, t0]
                    for r1 in range(n_receptors):
                        for t1 in range(n_tfs):
                            if t1 == t0:
                                continue
                            for l2 in range(n_ligands):
                                s2 = expr[1, l1] * lr[l1, r1] * rt[1, r1, t1] * tg_lig[1, l2, t1]
                                for r2 in range(n_receptors):
                                    for t2 in range(n_tfs):
                                        if t2 == t0 or t2 == t1:
                                            continue
                                        for g in range(n_genes):
                                            s3 = expr[2, l2] * lr[l2, r2] * rt[2, r2, t2] * tg_gene[g, t2]
                                            explicit.append(math.log(s1) + math.log(s2) + math.log(s3))

    # Independent state-wise reduction retaining the exact Top-K for every (L1, TF0) state.
    continuation: dict[tuple[int, int], list[float]] = {}
    for l1 in range(n_ligands):
        for t0 in range(n_tfs):
            values = []
            for r1 in range(n_receptors):
                for t1 in range(n_tfs):
                    if t1 == t0:
                        continue
                    for l2 in range(n_ligands):
                        s2 = expr[1, l1] * lr[l1, r1] * rt[1, r1, t1] * tg_lig[1, l2, t1]
                        for r2 in range(n_receptors):
                            for t2 in range(n_tfs):
                                if t2 == t0 or t2 == t1:
                                    continue
                                for g in range(n_genes):
                                    s3 = expr[2, l2] * lr[l2, r2] * rt[2, r2, t2] * tg_gene[g, t2]
                                    values.append(math.log(s2) + math.log(s3))
            continuation[(l1, t0)] = sorted(values, reverse=True)[:k]
    staged = []
    for l0 in range(n_ligands):
        for r0 in range(n_receptors):
            for t0 in range(n_tfs):
                for l1 in range(n_ligands):
                    s1 = math.log(expr[0, l0] * lr[l0, r0] * rt[0, r0, t0] * tg_lig[0, l1, t0])
                    staged.extend(s1 + x for x in continuation[(l1, t0)])
    explicit_top = np.sort(np.asarray(explicit, dtype=np.float64))[::-1][:k]
    staged_top = np.sort(np.asarray(staged, dtype=np.float64))[::-1][:k]
    error = float(np.max(np.abs(explicit_top - staged_top)))
    return {
        "k": k,
        "explicit_chain_count": len(explicit),
        "max_abs_log_score_error": error,
        "passed": bool(np.allclose(explicit_top, staged_top, rtol=0.0, atol=1e-12)),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    top_k = int(args.top_k)
    gene_top_k = max(top_k, int(args.terminal_gene_top_k))
    source_scripts = Path(args.source_scripts).resolve()
    sys.path.insert(0, str(source_scripts))
    from case_common import load_config  # type: ignore
    from paper_analysis_common import (  # type: ignore
        WINDOWS,
        build_positive_lr_table,
        load_checkpointed_model,
        load_real_window_bundles,
    )

    synthetic = distinct_tf_synthetic_validation(k=min(20, top_k))
    if not synthetic["passed"]:
        raise AssertionError(f"Distinct-TF synthetic validation failed: {synthetic}")

    cfg = load_config(Path(args.config))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundles = load_real_window_bundles(cfg)
    first = bundles[WINDOWS[0].label]
    receptor_names = first["receptor_to_tf"].index.astype(str).tolist()
    tf_names = first["receptor_to_tf"].columns.astype(str).tolist()
    gene_names = first["tf_to_gene"].index.astype(str).tolist()
    for window in WINDOWS:
        bundle = bundles[window.label]
        assert bundle["receptor_to_tf"].index.astype(str).tolist() == receptor_names
        assert bundle["receptor_to_tf"].columns.astype(str).tolist() == tf_names
        assert bundle["tf_to_gene"].columns.astype(str).tolist() == tf_names
        assert bundle["tf_to_gene"].index.astype(str).tolist() == gene_names

    _dataset, model, _device = load_checkpointed_model(cfg, device_override=args.device)
    positive_lr = build_positive_lr_table(model, cfg).reindex(columns=receptor_names)
    ligand_names = positive_lr.index.astype(str).tolist()
    lr_values = positive_lr.to_numpy(dtype=np.float64)
    edge_lig_idx, edge_rec_idx = np.where(lr_values > 0.0)
    edge_values = lr_values[edge_lig_idx, edge_rec_idx]
    edge_log_lr = np.log(edge_values)
    edge_indices_by_ligand = {
        int(ligand_idx): np.flatnonzero(edge_lig_idx == ligand_idx)
        for ligand_idx in np.unique(edge_lig_idx)
    }
    relay_ligand_indices = np.asarray(sorted(edge_indices_by_ligand), dtype=np.int64)
    n_states = len(relay_ligand_indices)
    n_tfs = len(tf_names)
    gene_row_by_name = {name: idx for idx, name in enumerate(gene_names)}
    relay_gene_rows = np.asarray(
        [gene_row_by_name[ligand_names[int(idx)]] for idx in relay_ligand_indices], dtype=np.int64
    )

    expr_logs, rt_logs, tg_logs = [], [], []
    for window in WINDOWS:
        bundle = bundles[window.label]
        expr_logs.append(
            positive_log(bundle["ligand_context"].reindex(ligand_names).to_numpy(dtype=np.float64))
        )
        rt_logs.append(positive_log(bundle["receptor_to_tf"].to_numpy(dtype=np.float64)))
        tg_logs.append(positive_log(bundle["tf_to_gene"].to_numpy(dtype=np.float64)))

    top_gene_idx = np.empty((n_tfs, gene_top_k), dtype=np.int64)
    top_gene_log = np.empty((n_tfs, gene_top_k), dtype=np.float64)
    for tf_idx in range(n_tfs):
        idx, values = topk_flat(tg_logs[2][:, tf_idx], gene_top_k)
        if len(idx) < gene_top_k:
            raise AssertionError(f"TF {tf_names[tf_idx]} has only {len(idx)} positive terminal genes")
        top_gene_idx[tf_idx] = idx
        top_gene_log[tf_idx] = values

    # Best terminal module for every relay ligand and TF. The target gene is the
    # top AttnTG gene for that TF; the best LR edge is selected explicitly.
    terminal_tf_score = np.full((n_states, n_tfs), -np.inf, dtype=np.float64)
    terminal_tf_edge = np.full((n_states, n_tfs), -1, dtype=np.int64)
    for state_idx, ligand_idx in enumerate(relay_ligand_indices):
        for edge_idx in edge_indices_by_ligand[int(ligand_idx)]:
            receptor_idx = int(edge_rec_idx[edge_idx])
            values = (
                expr_logs[2][int(ligand_idx)]
                + edge_log_lr[edge_idx]
                + rt_logs[2][receptor_idx, :]
                + top_gene_log[:, 0]
            )
            better = values > terminal_tf_score[state_idx]
            terminal_tf_score[state_idx, better] = values[better]
            terminal_tf_edge[state_idx, better] = int(edge_idx)
    terminal_tf_order = np.argsort(-terminal_tf_score, axis=1, kind="stable")

    best_excluding_one_score = np.empty((n_states, n_tfs), dtype=np.float64)
    best_excluding_one_tf = np.empty((n_states, n_tfs), dtype=np.int64)
    for state_idx in range(n_states):
        first_tf, second_tf = map(int, terminal_tf_order[state_idx, :2])
        best_excluding_one_score[state_idx, :] = terminal_tf_score[state_idx, first_tf]
        best_excluding_one_tf[state_idx, :] = first_tf
        best_excluding_one_score[state_idx, first_tf] = terminal_tf_score[state_idx, second_tf]
        best_excluding_one_tf[state_idx, first_tf] = second_tf

    terminal_exclusion_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def terminal_matrix_excluding_t0(t0: int) -> tuple[np.ndarray, np.ndarray]:
        """Return [TF1, relay2_state] best terminal score/TF2 excluding TF0 and TF1."""
        if t0 in terminal_exclusion_cache:
            return terminal_exclusion_cache[t0]
        scores = np.full((n_tfs, n_states), -np.inf, dtype=np.float64)
        choices = np.full((n_tfs, n_states), -1, dtype=np.int64)
        for state_idx in range(n_states):
            first_three = [int(x) for x in terminal_tf_order[state_idx, :3]]
            for t1 in range(n_tfs):
                if t1 == t0:
                    continue
                for t2 in first_three:
                    if t2 != t0 and t2 != t1:
                        scores[t1, state_idx] = terminal_tf_score[state_idx, t2]
                        choices[t1, state_idx] = t2
                        break
        terminal_exclusion_cache[t0] = (scores, choices)
        return scores, choices

    def segment2_log_matrix(edge_idx: int, ligand_idx: int) -> np.ndarray:
        receptor_idx = int(edge_rec_idx[edge_idx])
        return (
            expr_logs[1][ligand_idx]
            + edge_log_lr[edge_idx]
            + rt_logs[1][receptor_idx, :, None]
            + tg_logs[1][relay_gene_rows, :].T
        )

    # Exact best continuation for every (relay1 ligand, TF0). One unconstrained
    # TF1!=TF2 optimum resolves all TF0 values except its two TFs; those two
    # exceptional states are rescanned with their exact exclusion matrices.
    best_continuation_score = np.full((n_states, n_tfs), -np.inf, dtype=np.float64)
    for state1, ligand1_idx_raw in enumerate(relay_ligand_indices):
        ligand1_idx = int(ligand1_idx_raw)
        best_general = (-np.inf, -1, -1)
        for edge_idx in edge_indices_by_ligand[ligand1_idx]:
            segment = segment2_log_matrix(int(edge_idx), ligand1_idx)
            combined = segment + best_excluding_one_score.T
            flat = int(np.nanargmax(combined))
            value = float(combined.reshape(-1)[flat])
            if value > best_general[0]:
                t1, state2 = np.unravel_index(flat, combined.shape)
                best_general = (value, int(t1), int(best_excluding_one_tf[state2, t1]))
        general_score, general_t1, general_t2 = best_general
        best_continuation_score[state1, :] = general_score
        for t0 in {general_t1, general_t2}:
            terminal_scores, _terminal_choices = terminal_matrix_excluding_t0(int(t0))
            best_value = -np.inf
            for edge_idx in edge_indices_by_ligand[ligand1_idx]:
                combined = segment2_log_matrix(int(edge_idx), ligand1_idx) + terminal_scores
                value = float(np.nanmax(combined))
                if value > best_value:
                    best_value = value
            best_continuation_score[state1, int(t0)] = best_value

    terminal_sequence_cache: dict[tuple[int, int, int], list[TerminalPath]] = {}

    def terminal_sequence(state2: int, t0: int, t1: int) -> list[TerminalPath]:
        key = (state2, min(t0, t1), max(t0, t1))
        if key in terminal_sequence_cache:
            return terminal_sequence_cache[key]
        ligand2_idx = int(relay_ligand_indices[state2])
        heap: list[tuple[float, int, int, int]] = []
        for edge_idx in edge_indices_by_ligand[ligand2_idx]:
            receptor_idx = int(edge_rec_idx[edge_idx])
            base_by_tf = (
                expr_logs[2][ligand2_idx]
                + edge_log_lr[edge_idx]
                + rt_logs[2][receptor_idx, :]
            )
            for t2 in range(n_tfs):
                if t2 == t0 or t2 == t1:
                    continue
                value = float(base_by_tf[t2] + top_gene_log[t2, 0])
                if np.isfinite(value):
                    heapq.heappush(heap, (-value, int(edge_idx), t2, 0))
        records: list[TerminalPath] = []
        while heap and len(records) < top_k:
            neg_score, edge_idx, t2, gene_rank = heapq.heappop(heap)
            records.append(
                TerminalPath(-neg_score, edge_idx, t2, int(top_gene_idx[t2, gene_rank]))
            )
            next_rank = gene_rank + 1
            if next_rank < gene_top_k:
                receptor_idx = int(edge_rec_idx[edge_idx])
                base = (
                    expr_logs[2][ligand2_idx]
                    + edge_log_lr[edge_idx]
                    + rt_logs[2][receptor_idx, t2]
                )
                heapq.heappush(
                    heap,
                    (-(float(base + top_gene_log[t2, next_rank])), edge_idx, t2, next_rank),
                )
        terminal_sequence_cache[key] = records
        return records

    continuation_cache: dict[tuple[int, int], list[ContinuationPath]] = {}

    def continuation_sequence(state1: int, t0: int) -> list[ContinuationPath]:
        key = (state1, t0)
        if key in continuation_cache:
            return continuation_cache[key]
        ligand1_idx = int(relay_ligand_indices[state1])
        terminal_scores, terminal_choices = terminal_matrix_excluding_t0(t0)
        bases: list[dict[str, Any]] = []
        for edge_idx in edge_indices_by_ligand[ligand1_idx]:
            segment = segment2_log_matrix(int(edge_idx), ligand1_idx)
            combined = segment + terminal_scores
            flat_idx, values = topk_flat(combined, top_k)
            additions = []
            for flat, value in zip(flat_idx, values):
                t1, state2 = np.unravel_index(int(flat), combined.shape)
                additions.append(
                    {
                        "best_log_score": float(value),
                        "segment2_log_score": float(segment[t1, state2]),
                        "edge_idx": int(edge_idx),
                        "tf1_idx": int(t1),
                        "state2": int(state2),
                        "tf2_idx": int(terminal_choices[t1, state2]),
                    }
                )
            bases = retain_global_base_topk(bases, additions, top_k)

        heap: list[tuple[float, int, int]] = []
        for base_idx, base in enumerate(bases):
            heapq.heappush(heap, (-float(base["best_log_score"]), base_idx, 0))
        out: list[ContinuationPath] = []
        while heap and len(out) < top_k:
            neg_score, base_idx, terminal_rank = heapq.heappop(heap)
            base = bases[base_idx]
            terminals = terminal_sequence(int(base["state2"]), t0, int(base["tf1_idx"]))
            terminal = terminals[terminal_rank]
            exact_score = float(base["segment2_log_score"]) + terminal.log_score
            if terminal_rank == 0 and abs(exact_score + neg_score) > 1e-12:
                raise AssertionError("Terminal best-score cache disagrees with exact terminal sequence")
            out.append(
                ContinuationPath(
                    log_score=exact_score,
                    segment2_log_score=float(base["segment2_log_score"]),
                    edge1_idx=int(base["edge_idx"]),
                    tf1_idx=int(base["tf1_idx"]),
                    relay2_idx=int(relay_ligand_indices[int(base["state2"])]),
                    terminal=terminal,
                )
            )
            next_rank = terminal_rank + 1
            if next_rank < len(terminals):
                next_score = float(base["segment2_log_score"]) + terminals[next_rank].log_score
                heapq.heappush(heap, (-next_score, base_idx, next_rank))
        continuation_cache[key] = out
        return out

    # Segment 1 base candidates use the exact best continuation for their TF0.
    segment1_bases: list[dict[str, Any]] = []
    for edge_idx in range(len(edge_values)):
        ligand0_idx = int(edge_lig_idx[edge_idx])
        receptor0_idx = int(edge_rec_idx[edge_idx])
        segment = (
            expr_logs[0][ligand0_idx]
            + edge_log_lr[edge_idx]
            + rt_logs[0][receptor0_idx, :, None]
            + tg_logs[0][relay_gene_rows, :].T
        )
        combined = segment + best_continuation_score.T
        flat_idx, values = topk_flat(combined, top_k)
        additions = []
        for flat, value in zip(flat_idx, values):
            t0, state1 = np.unravel_index(int(flat), combined.shape)
            additions.append(
                {
                    "best_log_score": float(value),
                    "segment1_log_score": float(segment[t0, state1]),
                    "edge_idx": int(edge_idx),
                    "tf0_idx": int(t0),
                    "state1": int(state1),
                }
            )
        segment1_bases = retain_global_base_topk(segment1_bases, additions, top_k)

    # Lazy exact merge: only state pairs that actually enter the global Top-K
    # materialize their full continuation list.
    heap: list[tuple[float, int, int]] = []
    for base_idx, base in enumerate(segment1_bases):
        heapq.heappush(heap, (-float(base["best_log_score"]), base_idx, 0))
    selected: list[tuple[float, dict[str, Any], ContinuationPath]] = []
    while heap and len(selected) < top_k:
        neg_score, base_idx, continuation_rank = heapq.heappop(heap)
        base = segment1_bases[base_idx]
        continuations = continuation_sequence(int(base["state1"]), int(base["tf0_idx"]))
        continuation = continuations[continuation_rank]
        exact_score = float(base["segment1_log_score"]) + continuation.log_score
        if continuation_rank == 0 and abs(exact_score + neg_score) > 1e-12:
            raise AssertionError("Continuation best-score cache disagrees with exact continuation sequence")
        selected.append((exact_score, base, continuation))
        next_rank = continuation_rank + 1
        if next_rank < len(continuations):
            next_score = float(base["segment1_log_score"]) + continuations[next_rank].log_score
            heapq.heappush(heap, (-next_score, base_idx, next_rank))

    rows = []
    direct_errors = []
    for rank, (chain_log_score, base, continuation) in enumerate(selected, start=1):
        terminal = continuation.terminal
        edge0 = int(base["edge_idx"])
        l0 = int(edge_lig_idx[edge0])
        r0 = int(edge_rec_idx[edge0])
        t0 = int(base["tf0_idx"])
        l1 = int(relay_ligand_indices[int(base["state1"])] )
        r1 = int(edge_rec_idx[continuation.edge1_idx])
        t1 = int(continuation.tf1_idx)
        l2 = int(continuation.relay2_idx)
        r2 = int(edge_rec_idx[terminal.edge_idx])
        t2 = int(terminal.tf_idx)
        if len({t0, t1, t2}) != 3:
            raise AssertionError(f"Non-distinct TF chain escaped constraint: {t0}, {t1}, {t2}")
        s1 = float(np.exp(float(base["segment1_log_score"])))
        s2 = float(np.exp(continuation.segment2_log_score))
        s3 = float(np.exp(terminal.log_score))
        direct = math.log(s1) + math.log(s2) + math.log(s3)
        direct_errors.append(abs(direct - chain_log_score))
        rows.append(
            {
                "rank": rank,
                "ligand_0": ligand_names[l0],
                "receptor_0": receptor_names[r0],
                "tf_0": tf_names[t0],
                "relay_ligand_1": ligand_names[l1],
                "segment_1_pathscore": s1,
                "receptor_1": receptor_names[r1],
                "tf_1": tf_names[t1],
                "relay_ligand_2": ligand_names[l2],
                "segment_2_pathscore": s2,
                "receptor_2": receptor_names[r2],
                "tf_2": tf_names[t2],
                "terminal_gene": gene_names[terminal.gene_idx],
                "segment_3_pathscore": s3,
                "chain_log_score": chain_log_score,
                "chain_score_product": float(np.exp(chain_log_score)),
                "chain_score_geometric_mean": float(np.exp(chain_log_score / 3.0)),
                "segment_1": f"{ligand_names[l0]}->{receptor_names[r0]}->{tf_names[t0]}->{ligand_names[l1]}",
                "segment_2": f"{ligand_names[l1]}->{receptor_names[r1]}->{tf_names[t1]}->{ligand_names[l2]}",
                "segment_3": f"{ligand_names[l2]}->{receptor_names[r2]}->{tf_names[t2]}->{gene_names[terminal.gene_idx]}",
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "full_structural_relay_distinct_tf_top1000.csv", index=False)
    winner = result.iloc[0]
    pd.DataFrame(
        [
            {"window_label": "day0_to_day1", "path": winner.segment_1, "path_score": winner.segment_1_pathscore},
            {"window_label": "day1_to_day2", "path": winner.segment_2, "path_score": winner.segment_2_pathscore},
            {"window_label": "day2_to_day3", "path": winner.segment_3, "path_score": winner.segment_3_pathscore},
        ]
    ).to_csv(out_dir / "winner_three_segment_decomposition.csv", index=False)

    metadata = {
        "analysis": "exact full_structural_relay with pairwise-distinct TFs",
        "definition": "fixed window order, adjacent target-to-next-ligand identity, positive learned masked LR edges, TF0!=TF1!=TF2 pairwise",
        "ranking": "product of three assigned-window PathScores, computed in log domain",
        "config": str(Path(args.config).resolve()),
        "checkpoint": cfg.checkpoint_path,
        "n_ligands": len(ligand_names),
        "n_receptors": len(receptor_names),
        "n_tfs": n_tfs,
        "n_genes": len(gene_names),
        "n_positive_lr_edges": len(edge_values),
        "n_relay_eligible_ligands": n_states,
        "top_k_saved": top_k,
        "only_added_filter": "TF0, TF1, and TF2 must be pairwise distinct",
        "other_filters_not_used": [
            "endogenous-only ligand filter", "ligand induction filter", "loop-free filter",
            "external evidence",
            "stage-peak filter", "per-window pre-ranking truncation"
        ],
        "winner": rows[0],
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = {
        "synthetic_distinct_tf_staged_vs_explicit": synthetic,
        "expected_and_observed_tfs": [579, n_tfs],
        "expected_and_observed_genes": [10112, len(gene_names)],
        "all_output_tfs_pairwise_distinct": bool(
            ((result.tf_0 != result.tf_1) & (result.tf_0 != result.tf_2) & (result.tf_1 != result.tf_2)).all()
        ),
        "output_monotonic": bool(result.chain_log_score.is_monotonic_decreasing),
        "output_unique": bool(~result.duplicated(["segment_1", "segment_2", "segment_3"]).any()),
        "max_abs_direct_log_score_error": float(max(direct_errors)),
    }
    validation["passed"] = bool(
        synthetic["passed"] and n_tfs == 579 and len(gene_names) == 10112
        and validation["all_output_tfs_pairwise_distinct"] and validation["output_monotonic"]
        and validation["output_unique"] and validation["max_abs_direct_log_score_error"] < 1e-12
    )
    (out_dir / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    if not validation["passed"]:
        raise AssertionError(f"Final validation failed: {validation}")


if __name__ == "__main__":
    main()

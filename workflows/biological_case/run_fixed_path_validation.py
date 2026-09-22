#!/usr/bin/env python3
"""Reproduce Figure 6 internal readouts for one fixed, pre-ranked LITD path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from case_common import ensure_dir, load_config, load_metadata_frames
from paper_analysis_common import (
    WINDOWS, WindowConditionDataset, build_global_env_by_day, build_positive_lr_table,
    load_checkpointed_model, load_real_window_bundles, to_numpy,
)


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str, base: Path = ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else base / value


def quantile(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def experiment_bootstrap(
    frame: pd.DataFrame, value: str, repeats: int, seed: int,
) -> np.ndarray:
    grouped = [group[value].to_numpy(float) for _, group in frame.groupby("experiment", sort=True)]
    if not grouped:
        raise ValueError("No experiments available for bootstrap")
    rng = np.random.default_rng(seed)
    result = np.empty(repeats, dtype=float)
    for i in range(repeats):
        chosen = rng.integers(0, len(grouped), size=len(grouped))
        result[i] = float(np.concatenate([grouped[j] for j in chosen]).mean())
    return result


def replacement_environment(
    env_dict: dict[Any, Any], global_by_day: dict[float, torch.Tensor],
    experiments: list[str], mismatch_day: float, dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    vectors, fallback_count = [], 0
    for experiment in experiments:
        vector = env_dict.get((str(experiment), float(mismatch_day)))
        if vector is None:
            vector = global_by_day.get(float(mismatch_day))
            fallback_count += 1
        if vector is None:
            raise RuntimeError(f"No ligand environment for experiment={experiment}, day={mismatch_day}")
        vectors.append(torch.as_tensor(vector, dtype=dtype))
    return torch.stack(vectors), fallback_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/figure6_fixed_path.json")
    args = parser.parse_args()
    config_path = resolve(args.config)
    run_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    model_cfg = load_config(resolve(run_cfg["model_config"]))
    output_dir = resolve(run_cfg["output_dir"])
    ensure_dir(output_dir)
    path = {key: str(value) for key, value in run_cfg["path"].items() if key != "expected_rank"}

    ranked_path = resolve(run_cfg["ranked_paths"])
    ranked = pd.read_csv(ranked_path)
    selected = ranked.iloc[int(run_cfg["path"]["expected_rank"]) - 1]
    observed = tuple(str(selected[key]) for key in ("ligand", "receptor", "tf", "target_gene"))
    expected = tuple(path[key] for key in ("ligand", "receptor", "tf", "target_gene"))
    if observed != expected:
        raise ValueError(f"Configured fixed path is not rank {run_cfg['path']['expected_rank']}: {observed} != {expected}")

    dataset, model, device = load_checkpointed_model(model_cfg, device_override=run_cfg["device"])
    ligand_df, receptor_df, tf_df, _ = load_metadata_frames(model_cfg)
    ligand_names = ligand_df["gene_name"].astype(str).tolist()
    receptor_names = receptor_df["gene_name"].astype(str).tolist()
    tf_names = tf_df["gene_name"].astype(str).tolist()
    gene_names = list(map(str, dataset.var_names))
    indices = {
        "ligand": ligand_names.index(path["ligand"]),
        "receptor": receptor_names.index(path["receptor"]),
        "tf": tf_names.index(path["tf"]),
        "target_gene": gene_names.index(path["target_gene"]),
    }
    positive_lr = build_positive_lr_table(model, model_cfg)
    pos_lr = float(positive_lr.at[path["ligand"], path["receptor"]])
    if pos_lr <= 0:
        raise ValueError(f"Fixed path does not use a positive learned LR edge: {pos_lr}")
    global_by_day = build_global_env_by_day(dataset.env_dict)
    rows: list[dict[str, Any]] = []
    fallback_counts: dict[str, int] = {}

    for window in WINDOWS:
        pair_mask = (
            (dataset.pairs_df["day_source"].astype(float) == window.day_source)
            & (dataset.pairs_df["day_target"].astype(float) == window.day_target)
        )
        pair_indices = dataset.pairs_df.index[pair_mask].tolist()
        cap = int(run_cfg["max_samples_per_window"])
        if cap > 0:
            pair_indices = pair_indices[:cap]
        wrapper = WindowConditionDataset(
            dataset, pair_indices, "real", {name: i for i, name in enumerate(ligand_names)},
            global_by_day, zero_ligands=[], mask_ligands=[], stage_mismatch=False,
        )
        loader = DataLoader(wrapper, batch_size=int(run_cfg["batch_size"]), shuffle=False, num_workers=0)
        processed = 0
        for batch in loader:
            embeddings, real_ligand, source_expr, _target, experiments, _day_source, _day_target = batch
            embeddings, real_ligand, source_expr = embeddings.to(device), real_ligand.to(device), source_expr.to(device)
            model_input = embeddings if model_cfg.input_mode == "scfoundation" else source_expr
            conditions: dict[str, torch.Tensor] = {"real": real_ligand}
            zero = real_ligand.clone()
            zero[:, indices["ligand"]] = 0.0
            conditions[f"zero_{path['ligand']}"] = zero
            for mismatch_day in sorted({item.day_source for item in WINDOWS} - {window.day_source}):
                replacement, fallback = replacement_environment(
                    dataset.env_dict, global_by_day, list(experiments), mismatch_day, real_ligand.dtype
                )
                label = f"stage_mismatch_day{int(mismatch_day)}"
                conditions[label] = replacement.to(device)
                fallback_counts[f"{window.label}::{label}"] = fallback_counts.get(f"{window.label}::{label}", 0) + fallback
            for condition, ligand in conditions.items():
                # Counterfactuals use full-precision forward passes; the original autocast
                # window artifacts remain the source of the fixed global ranking.
                with torch.no_grad():
                    prediction, artifacts = model(model_input, ligand, return_artifacts=True)
                ligand_value = to_numpy(ligand[:, indices["ligand"]])
                attn_rt = to_numpy(artifacts["rec_to_tf_attn"][:, indices["tf"], indices["receptor"]])
                attn_tg = to_numpy(artifacts["tf_to_gene_attn"][:, indices["target_gene"], indices["tf"]])
                predicted = to_numpy(prediction[:, indices["target_gene"]])
                internal = pos_lr * attn_rt * attn_tg
                for experiment, lig, rt, tg, cascade, pred in zip(
                    experiments, ligand_value, attn_rt, attn_tg, internal, predicted
                ):
                    rows.append({
                        "window": window.label, "real_source_day": window.day_source,
                        "experiment": str(experiment), "condition": condition,
                        "ligand_expr": float(lig), "positive_lr": pos_lr,
                        "attn_rt": float(rt), "attn_tg": float(tg),
                        "internal_cascade": float(cascade), "pathscore": float(lig * cascade),
                        "predicted_target": float(pred),
                    })
            processed += len(experiments)
        expected_n = int(run_cfg["expected_window_sample_counts"][window.label])
        if cap == 0 and processed != expected_n:
            raise ValueError(f"{window.label}: observed {processed} samples, expected {expected_n}")

    long = pd.DataFrame(rows)
    long.to_csv(output_dir / "fixed_path_conditions_long.csv", index=False)
    repeats, seed = int(run_cfg["bootstrap_repeats"]), int(run_cfg["bootstrap_seed"])
    real = long.loc[long.condition == "real"].copy()
    window_boot: dict[str, np.ndarray] = {}
    bundles = load_real_window_bundles(model_cfg)
    stage_rows = []
    total_samples = sum(int((real.window == window.label).sum()) for window in WINDOWS)
    for offset, window in enumerate(WINDOWS):
        group = real.loc[real.window == window.label]
        boot = experiment_bootstrap(group, "pathscore", repeats, seed + offset)
        window_boot[window.label] = boot
        ci_low, ci_high = quantile(boot)
        bundle = bundles[window.label]
        expected_pathscore = (
            float(bundle["ligand_context"].at[path["ligand"]]) * pos_lr
            * float(bundle["receptor_to_tf"].at[path["receptor"], path["tf"]])
            * float(bundle["tf_to_gene"].at[path["target_gene"], path["tf"]])
        )
        score = float(group.pathscore.mean())
        stage_rows.append({
            "window": window.label, "n_samples": len(group),
            "n_experiments": int(group.experiment.nunique()), "window_weight": len(group) / total_samples,
            "pathscore": score, "pathscore_ci_low": ci_low, "pathscore_ci_high": ci_high,
            "expected_pathscore": expected_pathscore,
            "relative_error_vs_expected": abs(score - expected_pathscore) / abs(expected_pathscore),
        })
    stage = pd.DataFrame(stage_rows)
    weighted = float((stage.pathscore * stage.window_weight).sum())
    weighted_boot = sum(
        row.window_weight * window_boot[row.window]
        for row in stage.itertuples(index=False)
    )
    global_low, global_high = quantile(weighted_boot)
    peak_window = str(run_cfg["peak_window"])
    order = stage.sort_values("pathscore", ascending=False).window.tolist()
    ratio_boot = window_boot[order[0]] / window_boot[order[1]]
    ratio_low, ratio_high = quantile(ratio_boot)
    stage["global_weighted_pathscore"] = weighted
    stage["global_ci_low"] = global_low
    stage["global_ci_high"] = global_high
    stage["peak_window"] = peak_window
    stage["peak_over_second_ratio"] = float(stage.set_index("window").at[order[0], "pathscore"] / stage.set_index("window").at[order[1], "pathscore"])
    stage["peak_ratio_ci_low"] = ratio_low
    stage["peak_ratio_ci_high"] = ratio_high
    stage.to_csv(output_dir / "stage_specificity_summary.csv", index=False)

    zero_rows = []
    for offset, window in enumerate(WINDOWS):
        group = long.loc[long.window == window.label]
        r = group.loc[group.condition == "real"]
        z = group.loc[group.condition == f"zero_{path['ligand']}"]
        real_internal, zero_internal = float(r.internal_cascade.mean()), float(z.internal_cascade.mean())
        fractional = (zero_internal - real_internal) / real_internal
        real_boot = experiment_bootstrap(r, "internal_cascade", repeats, seed + 100 + offset)
        zero_boot = experiment_bootstrap(z, "internal_cascade", repeats, seed + 100 + offset)
        low, high = quantile((zero_boot - real_boot) / real_boot)
        zero_rows.append({
            "window": window.label, "real_internal_cascade": real_internal,
            f"zero_{path['ligand']}_internal_cascade": zero_internal,
            "fractional_change_internal_cascade": fractional,
            "fractional_change_ci_low": low, "fractional_change_ci_high": high,
            "real_attn_rt": float(r.attn_rt.mean()), "zero_attn_rt": float(z.attn_rt.mean()),
            "attn_rt_fractional_change": float((z.attn_rt.mean() - r.attn_rt.mean()) / r.attn_rt.mean()),
            "real_attn_tg": float(r.attn_tg.mean()), "zero_attn_tg": float(z.attn_tg.mean()),
            "attn_tg_fractional_change": float((z.attn_tg.mean() - r.attn_tg.mean()) / r.attn_tg.mean()),
            f"real_predicted_{path['target_gene']}": float(r.predicted_target.mean()),
            f"zero_predicted_{path['target_gene']}": float(z.predicted_target.mean()),
            f"predicted_{path['target_gene']}_delta": float(z.predicted_target.mean() - r.predicted_target.mean()),
            f"zero_{path['ligand']}_pathscore": float(z.pathscore.mean()),
            "pathscore_interpretation": "algebraically_zero_not_primary_evidence",
        })
    pd.DataFrame(zero_rows).to_csv(output_dir / f"zero_{path['ligand']}_counterfactual.csv", index=False)

    mismatch_rows = []
    for offset, window in enumerate(WINDOWS):
        group = long.loc[long.window == window.label]
        r = group.loc[group.condition == "real"]
        for mismatch_day in sorted({item.day_source for item in WINDOWS} - {window.day_source}):
            label = f"stage_mismatch_day{int(mismatch_day)}"
            m = group.loc[group.condition == label]
            real_internal, mismatch_internal = float(r.internal_cascade.mean()), float(m.internal_cascade.mean())
            fractional = (mismatch_internal - real_internal) / real_internal
            real_boot = experiment_bootstrap(r, "internal_cascade", repeats, seed + 200 + offset)
            mismatch_boot = experiment_bootstrap(m, "internal_cascade", repeats, seed + 200 + offset)
            low, high = quantile((mismatch_boot - real_boot) / real_boot)
            fallback = fallback_counts[f"{window.label}::{label}"]
            mismatch_rows.append({
                "real_window": window.label, "real_source_day": window.day_source,
                "mismatch_day": mismatch_day, "condition": label,
                "legacy_cyclic_mismatch": mismatch_day == {0.0: 2.0, 1.0: 0.0, 2.0: 1.0}[window.day_source],
                "real_pathscore": float(r.pathscore.mean()), "mismatch_pathscore": float(m.pathscore.mean()),
                "real_internal_cascade": real_internal, "mismatch_internal_cascade": mismatch_internal,
                "real_over_mismatch_internal_cascade": real_internal / mismatch_internal,
                "internal_cascade_fractional_change": fractional,
                "fractional_change_ci_low": low, "fractional_change_ci_high": high,
                "fallback_count": fallback, "fallback_fraction": fallback / len(m),
            })
    mismatch_frame = pd.DataFrame(mismatch_rows)
    mismatch_frame.to_csv(output_dir / "stage_mismatch_matrix.csv", index=False)
    # Figure 6 uses both other-stage forward passes with equal weight.
    mean_rows = []
    for window_name, group in mismatch_frame.groupby("real_window", sort=False):
        if len(group) != 2 or group["mismatch_day"].nunique() != 2:
            raise ValueError(f"Expected two alternative environments for {window_name}")
        matched = float(group["real_internal_cascade"].iloc[0])
        mean_mismatch = float(group["mismatch_internal_cascade"].mean())
        mean_rows.append({"real_window": window_name, "n_mismatch_environments": 2,
                          "real_internal_cascade": matched,
                          "mean_mismatch_internal_cascade": mean_mismatch,
                          "real_over_mean_mismatch_internal_cascade": matched / mean_mismatch})
    pd.DataFrame(mean_rows).to_csv(output_dir / "stage_mismatch_mean_summary.csv", index=False)

    boundary = pd.DataFrame([
        ("Selection", f"Exact global rank {run_cfg['path']['expected_rank']} in the full configured space", "model_selection_fact", "globally top-ranked model-derived candidate"),
        ("Stage specificity", f"peak={peak_window}", "internal_model_evidence", "window-varying model score"),
        ("Zero ligand", f"only the {path['ligand']} input coordinate is set to zero", "internal_counterfactual_evidence", "model sensitivity to the configured ligand input"),
        ("Stage mismatch", "all six off-diagonal environment replacements are evaluated", "internal_counterfactual_evidence", "condition dependence within the fixed model"),
        ("Causality", "no wet-lab perturbation; attention is not causal proof", "not_established", "model-derived candidate path only"),
    ], columns=["evidence_axis", "result", "status", "allowed_claim"])
    boundary.to_csv(output_dir / "interpretation_boundary.csv", index=False)
    observed_samples = {window.label: int((real.window == window.label).sum()) for window in WINDOWS}
    observed_experiments = {
        window.label: int(real.loc[real.window == window.label, "experiment"].nunique())
        for window in WINDOWS
    }
    expected_samples = {key: int(value) for key, value in run_cfg["expected_window_sample_counts"].items()}
    expected_experiments = {
        key: int(value) for key, value in run_cfg["expected_window_experiment_counts"].items()
    }
    ranking_value = float(selected["global_weighted_pathscore"])
    ranking_relative_error = abs(weighted - ranking_value) / abs(ranking_value)
    checks = {
        "fixed_rank_matches": observed == expected,
        "six_off_diagonal_mismatches": len(mismatch_rows) == 6,
        "zero_ligand_coordinate_exact_zero": bool((long.loc[long.condition == f"zero_{path['ligand']}", "ligand_expr"] == 0).all()),
        "sample_counts_match": observed_samples == expected_samples,
        "experiment_counts_match": observed_experiments == expected_experiments,
        "peak_window_matches": order[0] == peak_window,
        "counterfactual_global_score_within_tolerance": ranking_relative_error <= float(run_cfg["counterfactual_ranking_relative_tolerance"]),
        "all_numeric_outputs_finite": bool(np.isfinite(long.select_dtypes(include=[np.number])).all().all()),
    }
    passed = all(checks.values())
    report = {
        "status": "ok" if passed else "error",
        "path": path,
        "observed_sample_counts": observed_samples,
        "observed_experiment_counts": observed_experiments,
        "ranking_global_weighted_pathscore": ranking_value,
        "counterfactual_global_weighted_pathscore": weighted,
        "counterfactual_ranking_relative_error": ranking_relative_error,
        "checks": checks,
    }
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

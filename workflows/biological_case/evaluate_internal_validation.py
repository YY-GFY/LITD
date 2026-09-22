#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]

def resolve_repo_path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p

WINDOW_ORDER = ["day0_to_day1", "day1_to_day2", "day2_to_day3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one fixed distinct-TF relay without reranking.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(data: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator == 0:
        raise ValueError(f"Cannot compute finite ratio with denominator={denominator}")
    return float(numerator / denominator)


def validate_manifest(manifest: pd.DataFrame) -> None:
    required = {
        "segment_id", "assigned_window", "ligand", "receptor", "tf", "target_gene",
        "selection_pathscore", "selection_rank", "source_file",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    if manifest.shape[0] != 3:
        raise ValueError(f"Expected exactly three segments, found {manifest.shape[0]}")
    manifest.sort_values("segment_id", inplace=True, ignore_index=True)
    if manifest["assigned_window"].tolist() != WINDOW_ORDER:
        raise ValueError(f"Unexpected assigned window order: {manifest['assigned_window'].tolist()}")
    if not (manifest["selection_rank"].astype(int) == 1).all():
        raise ValueError("Every selected segment must originate from rank 1.")
    tfs = manifest["tf"].astype(str).tolist()
    if len(set(tfs)) != 3:
        raise ValueError(f"TFs must be pairwise distinct, found {tfs}")
    for idx in range(2):
        upstream_target = str(manifest.iloc[idx]["target_gene"])
        downstream_ligand = str(manifest.iloc[idx + 1]["ligand"])
        if upstream_target != downstream_ligand:
            raise ValueError(f"Broken adjacent handoff: {upstream_target} != {downstream_ligand}")


def validate_rank1_source(manifest: pd.DataFrame) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for source_name, group in manifest.groupby("source_file", sort=False):
        source = resolve_repo_path(str(source_name))
        if not source.is_file():
            raise FileNotFoundError(f"Rank-1 source does not exist: {source}")
        source_df = pd.read_csv(source)
        rank1 = source_df.loc[source_df["rank"].astype(int) == 1]
        if rank1.shape[0] != 1:
            raise ValueError(f"Expected exactly one rank-1 row in {source}")
        row = rank1.iloc[0]
        expected = [
            ("segment_1", row["ligand_0"], row["receptor_0"], row["tf_0"], row["relay_ligand_1"], row["segment_1_pathscore"]),
            ("segment_2", row["relay_ligand_1"], row["receptor_1"], row["tf_1"], row["relay_ligand_2"], row["segment_2_pathscore"]),
            ("segment_3", row["relay_ligand_2"], row["receptor_2"], row["tf_2"], row["terminal_gene"], row["segment_3_pathscore"]),
        ]
        for segment_id, ligand, receptor, tf, target, score in expected:
            selected = group.loc[group["segment_id"] == segment_id]
            if selected.shape[0] != 1:
                raise ValueError(f"Manifest/source mismatch for {segment_id}")
            item = selected.iloc[0]
            observed_tuple = (str(item.ligand), str(item.receptor), str(item.tf), str(item.target_gene))
            expected_tuple = (str(ligand), str(receptor), str(tf), str(target))
            if observed_tuple != expected_tuple or not np.isclose(float(item.selection_pathscore), float(score), rtol=0, atol=1e-14):
                raise ValueError(f"Manifest is not the saved rank-1 row for {segment_id}")
        hashes[str(source)] = sha256_file(source)
    return hashes


def script_hashes(script_dir: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(script_dir.glob("*")) if path.is_file()}


def main() -> None:
    args = parse_args()
    config_path = resolve_repo_path(args.config).resolve()
    run_cfg = load_json(config_path)
    output_dir = resolve_repo_path(run_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_repo_path(run_cfg["manifest"])
    manifest = pd.read_csv(manifest_path)
    validate_manifest(manifest)
    source_hashes = validate_rank1_source(manifest)

    source_scripts = resolve_repo_path(run_cfg["source_scripts"])
    sys.path.insert(0, str(source_scripts))
    from case_common import load_config, load_metadata_frames  # type: ignore
    from paper_analysis_common import (  # type: ignore
        WINDOW_BY_LABEL,
        WindowConditionDataset,
        build_global_env_by_day,
        build_positive_lr_table,
        load_checkpointed_model,
        to_numpy,
    )

    model_config_path = resolve_repo_path(run_cfg["model_config"])
    model_cfg = load_config(model_config_path)
    checkpoint_path = Path(model_cfg.checkpoint_path)
    started_at = datetime.now(timezone.utc).isoformat()
    metadata: dict[str, Any] = {
        "analysis_name": run_cfg["analysis_name"],
        "mode": run_cfg["mode"],
        "status": "running",
        "started_at_utc": started_at,
        "definition": "InternalCascade = positive LR weight * mean AttnRT * mean AttnTG",
        "legacy_pathscore_definition": "mean ligand expression * InternalCascade",
        "mismatch_definition": "Evaluate both alternative source days separately, then average their InternalCascade values equally.",
        "selection_role": "The saved rank-1 score fixes candidates only; this run does not rerank.",
        "config_snapshot": run_cfg,
        "hashes": {
            "run_config_sha256": sha256_file(config_path),
            "model_config_sha256": sha256_file(model_config_path),
            "manifest_sha256": sha256_file(manifest_path),
            "rank1_source_sha256": source_hashes,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "scripts_sha256": script_hashes(Path(__file__).resolve().parent),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
        "requested_device": run_cfg.get("device", "cuda:0"),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
    }
    atomic_json(metadata, output_dir / "run_metadata.json")

    dataset, model, device = load_checkpointed_model(model_cfg, device_override=run_cfg.get("device", model_cfg.device))
    if device.type != "cuda":
        raise RuntimeError(f"CUDA execution required, got {device}")
    ligand_df, receptor_df, tf_df, _ = load_metadata_frames(model_cfg)
    ligand_names = ligand_df["gene_name"].astype(str).tolist()
    receptor_names = receptor_df["gene_name"].astype(str).tolist()
    tf_names = tf_df["gene_name"].astype(str).tolist()
    gene_names = list(map(str, dataset.var_names))
    if len(tf_names) != int(run_cfg["expected_tf_count"]):
        raise ValueError(f"Expected {run_cfg['expected_tf_count']} TFs, found {len(tf_names)}")
    if len(gene_names) != int(run_cfg["expected_gene_count"]):
        raise ValueError(f"Expected {run_cfg['expected_gene_count']} genes, found {len(gene_names)}")

    ligand_index = {name: idx for idx, name in enumerate(ligand_names)}
    receptor_index = {name: idx for idx, name in enumerate(receptor_names)}
    tf_index = {name: idx for idx, name in enumerate(tf_names)}
    gene_index = {name: idx for idx, name in enumerate(gene_names)}
    for row in manifest.itertuples(index=False):
        for label, name, index in (
            ("ligand", row.ligand, ligand_index), ("receptor", row.receptor, receptor_index),
            ("TF", row.tf, tf_index), ("gene", row.target_gene, gene_index),
        ):
            if str(name) not in index:
                raise KeyError(f"{row.segment_id}: {label} {name} not found in model index")

    positive_lr = build_positive_lr_table(model, model_cfg)
    path_specs: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        lr_weight = float(positive_lr.at[row.ligand, row.receptor])
        if not np.isfinite(lr_weight) or lr_weight <= 0:
            raise ValueError(f"Non-positive LR edge for {row.ligand}->{row.receptor}: {lr_weight}")
        path_specs.append({
            "segment_id": row.segment_id,
            "assigned_window": row.assigned_window,
            "ligand": row.ligand,
            "receptor": row.receptor,
            "tf": row.tf,
            "target_gene": row.target_gene,
            "selection_pathscore": float(row.selection_pathscore),
            "ligand_idx": ligand_index[row.ligand],
            "receptor_idx": receptor_index[row.receptor],
            "tf_idx": tf_index[row.tf],
            "gene_idx": gene_index[row.target_gene],
            "pos_lr": lr_weight,
        })

    global_env_by_day = build_global_env_by_day(dataset.env_dict)
    pair_df = dataset.pairs_df
    max_samples = int(run_cfg["max_samples_per_window"])
    batch_size = int(run_cfg["batch_size"])
    real_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    experiment_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    experiment_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sample_counts: dict[str, int] = {}
    experiment_denominators: dict[str, int] = {}
    mismatch_diagnostics: dict[str, dict[str, float | int]] = {}

    for window_label in WINDOW_ORDER:
        window = WINDOW_BY_LABEL[window_label]
        mask = ((pair_df["day_source"].astype(float) == window.day_source) &
                (pair_df["day_target"].astype(float) == window.day_target))
        indices = pair_df.index[mask].tolist()
        if max_samples > 0:
            indices = indices[:max_samples]
        if not indices:
            raise RuntimeError(f"No samples for {window_label}")
        sample_counts[window_label] = len(indices)
        experiment_denominators[window_label] = int(pair_df.loc[indices, "experiment"].astype(str).nunique())
        assigned = next(spec for spec in path_specs if spec["assigned_window"] == window_label)
        wrapper = WindowConditionDataset(
            dataset, indices, condition_name="real", ligand_index_map=ligand_index,
            global_env_by_day=global_env_by_day, zero_ligands=[], mask_ligands=[], stage_mismatch=False,
        )
        loader = DataLoader(wrapper, batch_size=batch_size, shuffle=False, num_workers=0)
        condition_sums: dict[str, dict[str, dict[str, float]]] = {
            "real": {spec["segment_id"]: defaultdict(float) for spec in path_specs},
            **{f"stage_mismatch_day{int(day)}": {assigned["segment_id"]: defaultdict(float)}
               for day in (0.0, 1.0, 2.0) if day != window.day_source},
            f"zero_{assigned['ligand']}": {assigned["segment_id"]: defaultdict(float)},
        }
        wrong_days = [day for day in (0.0, 1.0, 2.0) if day != window.day_source]
        mismatch_replaced = {day: 0 for day in wrong_days}
        mismatch_fallbacks = {day: 0 for day in wrong_days}
        mismatch_l1_sum = {day: 0.0 for day in wrong_days}
        processed = 0
        print(f"[{window_label}] samples={len(indices)} assigned={assigned['segment_id']}", flush=True)
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                emb_t, ligand_vec, src_expr, _tgt_expr, experiments, day_sources, _day_targets = batch
                emb_t = emb_t.to(device)
                ligand_vec = ligand_vec.to(device)
                src_expr = src_expr.to(device)
                model_input = emb_t if model_cfg.input_mode == "scfoundation" else src_expr
                conditioned: dict[str, torch.Tensor] = {"real": ligand_vec}
                for wrong_day in wrong_days:
                    replacements = []
                    for experiment in experiments:
                        replacement = dataset.env_dict.get((str(experiment), wrong_day))
                        if replacement is None:
                            replacement = global_env_by_day.get(wrong_day)
                            mismatch_fallbacks[wrong_day] += 1
                        if replacement is None:
                            raise RuntimeError(f"Missing environment: {experiment}, day {wrong_day}")
                        replacements.append(torch.as_tensor(replacement, dtype=ligand_vec.dtype))
                    mismatch_vec = torch.stack(replacements).to(device)
                    mismatch_replaced[wrong_day] += int(mismatch_vec.shape[0])
                    mismatch_l1_sum[wrong_day] += float(torch.abs(mismatch_vec - ligand_vec).sum().item())
                    conditioned[f"stage_mismatch_day{int(wrong_day)}"] = mismatch_vec
                zero_vec = ligand_vec.clone()
                zero_vec[:, assigned["ligand_idx"]] = 0.0
                conditioned[f"zero_{assigned['ligand']}"] = zero_vec

                real_sample_cache = None
                for condition, ligand_condition in conditioned.items():
                    with torch.amp.autocast("cuda", enabled=True):
                        pred, artifacts = model(model_input, ligand_condition, return_artifacts=True)
                    specs = path_specs if condition == "real" else [assigned]
                    for spec in specs:
                        acc = condition_sums[condition][spec["segment_id"]]
                        acc["ligand_expr"] += float(ligand_condition[:, spec["ligand_idx"]].sum().item())
                        acc["attn_rt"] += float(artifacts["rec_to_tf_attn"][:, spec["tf_idx"], spec["receptor_idx"]].sum().item())
                        acc["attn_tg"] += float(artifacts["tf_to_gene_attn"][:, spec["gene_idx"], spec["tf_idx"]].sum().item())
                        acc["predicted_target_mean"] += float(pred[:, spec["gene_idx"]].sum().item())
                    if condition == "real":
                        ligand_np = to_numpy(ligand_condition[:, assigned["ligand_idx"]])
                        rt_np = to_numpy(artifacts["rec_to_tf_attn"][:, assigned["tf_idx"], assigned["receptor_idx"]])
                        tg_np = to_numpy(artifacts["tf_to_gene_attn"][:, assigned["gene_idx"], assigned["tf_idx"]])
                        real_sample_cache = ligand_np * assigned["pos_lr"] * rt_np * tg_np
                if real_sample_cache is None:
                    raise AssertionError("Real sample scores were not computed")
                for experiment, value in zip(experiments, real_sample_cache):
                    experiment_totals[assigned["segment_id"]][str(experiment)] += float(value)
                    experiment_counts[assigned["segment_id"]][str(experiment)] += 1
                processed += int(ligand_vec.shape[0])
                if (batch_idx + 1) % 100 == 0:
                    print(f"[{window_label}] batches={batch_idx + 1} processed={processed}", flush=True)
        if processed != len(indices):
            raise RuntimeError(f"Processed {processed}/{len(indices)} samples in {window_label}")
        mismatch_diagnostics[window_label] = {
            str(int(day)): {
                "n_replaced": mismatch_replaced[day],
                "n_global_fallback": mismatch_fallbacks[day],
                "wrong_day_source": day,
                "mean_l1_difference_from_real": mismatch_l1_sum[day] / processed,
            } for day in wrong_days
        }
        for day in wrong_days:
            if mismatch_replaced[day] != processed or mismatch_l1_sum[day] <= 0:
                raise RuntimeError(f"Stage mismatch not applied: {window_label}, day {day}")

        for condition, segment_accs in condition_sums.items():
            for segment_id, acc in segment_accs.items():
                spec = next(item for item in path_specs if item["segment_id"] == segment_id)
                ligand_expr = acc["ligand_expr"] / processed
                attn_rt = acc["attn_rt"] / processed
                attn_tg = acc["attn_tg"] / processed
                internal = spec["pos_lr"] * attn_rt * attn_tg
                record = {
                    "segment_id": segment_id,
                    "assigned_window": spec["assigned_window"],
                    "evaluated_window": window_label,
                    "ligand": spec["ligand"], "receptor": spec["receptor"], "tf": spec["tf"],
                    "target_gene": spec["target_gene"], "condition": condition,
                    "n_samples": processed, "ligand_expr": ligand_expr, "pos_lr": spec["pos_lr"],
                    "attn_rt": attn_rt, "attn_tg": attn_tg, "internal_cascade": internal,
                    "legacy_path_score": ligand_expr * internal,
                    "predicted_target_mean": acc["predicted_target_mean"] / processed,
                    "selection_pathscore": spec["selection_pathscore"],
                }
                if condition == "real":
                    real_rows.append(record)
                    if spec["assigned_window"] == window_label:
                        counterfactual_rows.append(record)
                else:
                    counterfactual_rows.append(record)

    if run_cfg["mode"] == "formal" and sample_counts != run_cfg["expected_window_sample_counts"]:
        raise RuntimeError(f"Unexpected sample counts: {sample_counts}")
    stage_long = pd.DataFrame(real_rows).sort_values(["segment_id", "evaluated_window"]).reset_index(drop=True)
    stage_summary_rows: list[dict[str, Any]] = []
    for spec in path_specs:
        group = stage_long.loc[stage_long["segment_id"] == spec["segment_id"]]
        assigned = group.loc[group["evaluated_window"] == spec["assigned_window"]].iloc[0]
        others = group.loc[group["evaluated_window"] != spec["assigned_window"]]
        row: dict[str, Any] = {
            "segment_id": spec["segment_id"], "assigned_window": spec["assigned_window"],
            "ligand": spec["ligand"], "receptor": spec["receptor"], "tf": spec["tf"],
            "target_gene": spec["target_gene"], "selection_pathscore": spec["selection_pathscore"],
        }
        for metric in ("legacy_path_score", "internal_cascade"):
            assigned_value = float(assigned[metric])
            other_max = float(others[metric].max())
            other_mean = float(others[metric].mean())
            row[f"assigned_{metric}"] = assigned_value
            row[f"other_max_{metric}"] = other_max
            row[f"other_mean_{metric}"] = other_mean
            row[f"assigned_over_other_max_{metric}"] = safe_ratio(assigned_value, other_max)
            row[f"assigned_over_other_mean_{metric}"] = safe_ratio(assigned_value, other_mean)
        stage_summary_rows.append(row)
    stage_summary = pd.DataFrame(stage_summary_rows)

    counter_long = pd.DataFrame(counterfactual_rows).sort_values(["segment_id", "condition"]).reset_index(drop=True)
    counter_summary_rows: list[dict[str, Any]] = []
    for spec in path_specs:
        group = counter_long.loc[counter_long["segment_id"] == spec["segment_id"]]
        real = group.loc[group["condition"] == "real"].iloc[0]
        mismatches = group.loc[group["condition"].str.startswith("stage_mismatch_day")]
        if len(mismatches) != 2 or mismatches["condition"].nunique() != 2:
            raise ValueError("Expected exactly two independently evaluated mismatch environments")
        # Average the TWO separately computed scores; never average inputs or ratios.
        mismatch = mismatches.select_dtypes(include=[np.number]).mean()

        zero = group.loc[group["condition"] == f"zero_{spec['ligand']}"].iloc[0]
        out: dict[str, Any] = {
            "segment_id": spec["segment_id"], "assigned_window": spec["assigned_window"],
            "ligand": spec["ligand"], "receptor": spec["receptor"], "tf": spec["tf"],
            "target_gene": spec["target_gene"], "selection_pathscore": spec["selection_pathscore"],
            "n_mismatch_environments": 2,
            "mismatch_aggregation": "arithmetic mean of two separately evaluated other-stage environments",
            "real_over_mismatch_internal_cascade": safe_ratio(real.internal_cascade, mismatch.internal_cascade),
        }
        for metric in ("ligand_expr", "attn_rt", "attn_tg", "internal_cascade", "legacy_path_score", "predicted_target_mean"):
            real_value = float(real[metric]); mismatch_value = float(mismatch[metric]); zero_value = float(zero[metric])
            out[f"real_{metric}"] = real_value
            out[f"mismatch_{metric}"] = mismatch_value
            out[f"zero_{metric}"] = zero_value
            out[f"mismatch_minus_real_{metric}"] = mismatch_value - real_value
            out[f"zero_minus_real_{metric}"] = zero_value - real_value
            out[f"zero_fractional_change_{metric}"] = safe_ratio(zero_value - real_value, real_value)
        counter_summary_rows.append(out)
    counter_summary = pd.DataFrame(counter_summary_rows)

    support_rows: list[dict[str, Any]] = []
    support_summary_rows: list[dict[str, Any]] = []
    threshold_fraction = float(run_cfg["support_threshold_fraction"])
    for spec in path_specs:
        segment = spec["segment_id"]
        experiment_names = sorted(experiment_counts[segment])
        scores: list[float] = []
        threshold = threshold_fraction * spec["selection_pathscore"]
        for experiment in experiment_names:
            count = experiment_counts[segment][experiment]
            mean_score = experiment_totals[segment][experiment] / count
            scores.append(mean_score)
            support_rows.append({
                "segment_id": segment, "assigned_window": spec["assigned_window"], "experiment": experiment,
                "n_samples": count, "experiment_mean_pathscore": mean_score,
                "selection_pathscore": spec["selection_pathscore"], "threshold_fraction": threshold_fraction,
                "support_threshold": threshold, "supported": bool(mean_score >= threshold),
            })
        if len(scores) != experiment_denominators[spec["assigned_window"]]:
            raise RuntimeError(f"Experiment denominator mismatch for {segment}")
        values = np.asarray(scores, dtype=float)
        n_supported = int(np.sum(values >= threshold))
        support_summary_rows.append({
            "segment_id": segment, "assigned_window": spec["assigned_window"],
            "selection_pathscore": spec["selection_pathscore"], "support_threshold": threshold,
            "n_supported_experiments": n_supported, "n_experiments": len(scores),
            "support_fraction": n_supported / len(scores), "median_experiment_pathscore": float(np.median(values)),
            "q25_experiment_pathscore": float(np.quantile(values, 0.25)),
            "q75_experiment_pathscore": float(np.quantile(values, 0.75)),
            "iqr_experiment_pathscore": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        })
    support_by_experiment = pd.DataFrame(support_rows)
    support_summary = pd.DataFrame(support_summary_rows)

    for frame_name, frame in {
        "stage_specificity_long": stage_long, "stage_specificity_summary": stage_summary,
        "internal_cascade_counterfactuals_long": counter_long, "internal_cascade_summary": counter_summary,
        "experiment_support_by_experiment": support_by_experiment,
        "experiment_support_summary": support_summary,
    }.items():
        numeric = frame.select_dtypes(include=[np.number]).to_numpy()
        if numeric.size and not np.isfinite(numeric).all():
            raise ValueError(f"Non-finite output in {frame_name}")
        atomic_csv(frame, output_dir / f"{frame_name}.csv")

    boundary = {
        "internal_model_evidence": "InternalCascade, zero-ligand, and stage-mismatch results measure model-internal conditional dependence; they are not causal knockouts.",
        "stage_mismatch": "Stage mismatch is a synthetic replacement with a wrong-stage ligand environment, not a biological intervention.",
        "experiment_support": "Experiment support is reproducibility across experiments within the same Cuomo dataset, not an external cohort.",
        "external_evidence": "No database, marker, or literature evidence is used to endorse this result.",
        "segment_scope": "Each segment must be interpreted separately; metrics from different sample windows must not be multiplied into a causal-chain score.",
        "ranking": "Selection scores only fixed the candidate; internal validation does not alter the relay ranking.",
    }
    atomic_json(boundary, output_dir / "interpretation_boundary.json")
    (output_dir / "interpretation_boundary.md").write_text(
        "# Interpretation boundary\n\n" + "\n".join(f"- {value}" for value in boundary.values()) + "\n",
        encoding="utf-8",
    )

    metadata.update({
        "status": "compute_ok", "completed_compute_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device), "device_name": torch.cuda.get_device_name(device),
        "tf_count": len(tf_names), "gene_count": len(gene_names), "ligand_count": len(ligand_names),
        "receptor_count": len(receptor_names), "sample_counts": sample_counts,
        "experiment_counts": experiment_denominators, "mismatch_diagnostics": mismatch_diagnostics,
        "positive_lr_weights": {spec["segment_id"]: spec["pos_lr"] for spec in path_specs},
        "outputs": sorted(path.name for path in output_dir.glob("*.csv")),
    })
    atomic_json(metadata, output_dir / "run_metadata.json")
    print(json.dumps({"status": "compute_ok", "sample_counts": sample_counts}, indent=2), flush=True)


if __name__ == "__main__":
    main()

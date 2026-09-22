#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from case_common import DEFAULT_CONFIG_PATH, ensure_dir, load_config, load_metadata_frames
from paper_analysis_common import (
    WINDOWS,
    build_global_env_by_day,
    load_checkpointed_model,
    summarize_window_condition,
    write_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export real-condition window artifacts for paper-final analysis.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--batch-size", type=int, default=0, help="Override inference batch size from config.")
    parser.add_argument("--max-samples-per-window", type=int, default=0, help="Optional sample cap; 0 uses all pairs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(cfg.batch_size)

    out_root = Path(cfg.source_window_artifact_root)
    ensure_dir(out_root)

    dataset, model, device = load_checkpointed_model(cfg)
    ligand_df, receptor_df, tf_df, _lr_pairs_df = load_metadata_frames(cfg)
    ligand_names = ligand_df["gene_name"].astype(str).tolist()
    receptor_names = receptor_df["gene_name"].astype(str).tolist()
    tf_names = tf_df["gene_name"].astype(str).tolist()
    gene_names = dataset.var_names
    ligand_index_map = {gene: idx for idx, gene in enumerate(ligand_names)}
    global_env_by_day = build_global_env_by_day(dataset.env_dict)

    inventory_rows = []
    for window in WINDOWS:
        print(
            f"[export] window={window.label} day_source={window.day_source} day_target={window.day_target} batch_size={batch_size}",
            flush=True,
        )
        bundle = summarize_window_condition(
            dataset=dataset,
            model=model,
            cfg=cfg,
            device=device,
            window=window,
            condition_name="real",
            ligand_index_map=ligand_index_map,
            ligand_names=ligand_names,
            receptor_names=receptor_names,
            tf_names=tf_names,
            gene_names=gene_names,
            global_env_by_day=global_env_by_day,
            batch_size=batch_size,
            zero_ligands=[],
            mask_ligands=[],
            stage_mismatch=False,
            max_samples_per_window=args.max_samples_per_window,
        )
        write_bundle(bundle, out_root / window.label / "real")
        inventory_rows.append(
            {
                "window_label": window.label,
                "day_source": window.day_source,
                "day_target": window.day_target,
                "n_samples": bundle["n_samples"],
                "condition": "real",
            }
        )

    inventory_df = pd.DataFrame(inventory_rows)
    inventory_df.to_csv(out_root / "window_artifact_inventory.csv", index=False)
    summary = {
        "run_name": cfg.run_name,
        "analysis_name": cfg.analysis_name,
        "source_window_artifact_root": str(out_root),
        "batch_size": batch_size,
        "windows": inventory_rows,
    }
    (out_root / "window_artifact_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[export] completed root={out_root}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--method", choices=["SCODE", "SINCERITIES", "PIDC"], required=True)
    p.add_argument("--dataset", choices=["hesc", "hhep"], required=True)
    p.add_argument("--target-gene-count", type=int, choices=[500, 1000], required=True)
    p.add_argument("--raw-output", required=True)
    return p.parse_args()


def main() -> None:
    a = args()
    cfg = json.loads(Path(a.config).read_text())
    root = Path(cfg["root"])
    candidate = pd.read_csv(cfg["final_litd_edges"][a.dataset][str(a.target_gene_count)], usecols=["Gene1", "Gene2"])
    candidate = candidate.astype(str).drop_duplicates()
    allowed = set(candidate["Gene1"] + "|" + candidate["Gene2"])
    raw_path = Path(a.raw_output)

    if a.method == "SCODE":
        genes = pd.read_csv(root / "method_inputs" / a.dataset / f"tgt{a.target_gene_count}" / "genes.txt", header=None)[0].astype(str)
        matrix = pd.read_csv(raw_path, sep="\t", header=None).to_numpy(dtype=float)
        if matrix.shape != (len(genes), len(genes)):
            raise ValueError(f"SCODE matrix {matrix.shape} != {(len(genes), len(genes))}")
        tf_names = candidate["Gene1"].drop_duplicates().tolist()
        target_names = candidate["Gene2"].drop_duplicates().tolist()
        pos = pd.Series(np.arange(len(genes)), index=genes)
        block = np.abs(matrix[np.ix_(pos.loc[tf_names], pos.loc[target_names])])
        out = pd.DataFrame({
            "Gene1": np.repeat(tf_names, len(target_names)),
            "Gene2": np.tile(target_names, len(tf_names)),
            "score": block.ravel(),
        })
    elif a.method == "SINCERITIES":
        raw = pd.read_csv(raw_path)
        # Preserve the official BEELINE parser's documented source/target swap.
        out = raw.rename(columns={"TargetGENES": "Gene1", "SourceGENES": "Gene2", "Interaction": "score"})
        out = out[["Gene1", "Gene2", "score"]]
    else:
        raw = pd.read_csv(raw_path, sep="\t", header=None, names=["Gene1", "Gene2", "score"])
        out = raw

    out["Gene1"] = out["Gene1"].astype(str)
    out["Gene2"] = out["Gene2"].astype(str)
    out["score"] = pd.to_numeric(out["score"], errors="coerce").abs()
    out["edge_id"] = out["Gene1"] + "|" + out["Gene2"]
    out = out[out["edge_id"].isin(allowed) & out["score"].notna()]
    out = out.sort_values(["score", "Gene1", "Gene2"], ascending=[False, True, True], kind="mergesort")
    out = out.drop_duplicates(["Gene1", "Gene2"]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["method"] = a.method
    out["dataset"] = a.dataset
    out["target_gene_count"] = a.target_gene_count
    dest = root / "results" / "ranked_edges" / a.method / a.dataset
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{a.dataset}_tgt{a.target_gene_count}_ranked_edges.csv.gz"
    out[["Gene1", "Gene2", "score", "rank", "method", "dataset", "target_gene_count"]].to_csv(
        path, index=False, compression={"method": "gzip", "compresslevel": 1}
    )
    print(f"[{a.method}] rows={len(out)} path={path}", flush=True)


if __name__ == "__main__":
    main()

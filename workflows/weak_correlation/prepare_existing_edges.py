#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from workflow_common import resolve, signature, reusable, record


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def standardize(path: str, method: str, dataset: str, target_count: int) -> pd.DataFrame:
    d = pd.read_csv(path)
    score_col = next((c for c in ["score", "EdgeWeightAbs", "EdgeWeight", "importance", "weight"] if c in d.columns), None)
    if score_col is None:
        raise ValueError(f"No score column in {path}")
    out = d[["Gene1", "Gene2", score_col]].rename(columns={score_col: "score"}).copy()
    out["Gene1"] = out["Gene1"].astype(str)
    out["Gene2"] = out["Gene2"].astype(str)
    out["score"] = pd.to_numeric(out["score"], errors="coerce").abs()
    out = out.dropna(subset=["score"])
    out = out[out["Gene1"] != out["Gene2"]]
    out = out.sort_values(["score", "Gene1", "Gene2"], ascending=[False, True, True], kind="mergesort")
    out = out.drop_duplicates(["Gene1", "Gene2"], keep="first").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["method"] = method
    out["dataset"] = dataset
    out["target_gene_count"] = int(target_count)
    return out[["Gene1", "Gene2", "score", "rank", "method", "dataset", "target_gene_count"]]


def main() -> None:
    args = parse_args()
    cfg = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    root = resolve(cfg["root"])
    sources = [(name, spec["edges"]) for name, spec in cfg["model_exports"].items()]
    sources += list(cfg["external_edges"].items())
    for method, mapping in sources:
        for dataset in cfg["datasets"]:
            for target_count in cfg["target_gene_counts"]:
                source = resolve(mapping[dataset][str(target_count)])
                out_dir = root / "results" / "ranked_edges" / method / dataset
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{dataset}_tgt{target_count}_ranked_edges.csv.gz"
                stamp = signature([source, Path(__file__)], {'method':method,'dataset':dataset,'target_count':target_count})
                metadata = out_path.with_suffix(out_path.suffix + '.provenance.json')
                if reusable(out_path, metadata, stamp):
                    print(f"[{method}] {dataset} tgt={target_count} already_exists={out_path}", flush=True)
                    continue
                out = standardize(source, method, dataset, target_count)
                out.to_csv(out_path, index=False, compression={"method": "gzip", "compresslevel": 1})
                record(out_path, metadata, stamp)
                print(f"[{method}] {dataset} tgt={target_count} rows={len(out)}", flush=True)
                del out
                gc.collect()


if __name__ == "__main__":
    main()

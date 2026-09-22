#!/usr/bin/env python3
"""Export complete candidate scores from existing checkpoints; never retrain."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from workflow_common import ROOT, resolve, load_config, signature, reusable, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/weak_correlation.json')
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    sys.path.insert(0, str(ROOT / 'scripts'))
    from _config import benchmark_paths, run_args
    from litd import benchmark
    bcfg = load_config(cfg['benchmark_config'])
    requested = args.methods or list(cfg['model_exports'])
    unknown = set(requested) - set(cfg['model_exports'])
    if unknown:
        raise ValueError(f'Unknown model methods: {sorted(unknown)}')
    for method in requested:
        profile = cfg['model_exports'][method]
        runtime_args = run_args(bcfg, input_mode=profile['input_mode'], lr_mode=profile['lr_mode'])
        paths = benchmark_paths(bcfg)
        if profile.get('workdir'):
            paths = benchmark.Paths(**{**paths.__dict__, 'workdir': str(resolve(profile['workdir']))})
        # Use each saved model's original data/pair order, as in the paper exports.
        for dataset in cfg['datasets']:
            checkpoint = resolve(profile['checkpoints'][dataset])
            runtime = benchmark.load_runtime_paths(paths, dataset)
            sources = [checkpoint, Path(paths.human_tfs), resolve(cfg['benchmark_config']), Path(__file__),
                       ROOT / 'litd/model.py', ROOT / 'litd/benchmark.py', ROOT / 'litd/data.py']
            sources += [Path(p) for p in runtime.values() if isinstance(p, str) and Path(p).is_file() and Path(p).suffix not in {'.npy', '.bin'}]
            # The huge embedding is identified by its size and mtime; small inputs
            # and checkpoint/code are hashed, so stale rankings cannot be reused.
            embeddings = {str(p): {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                          for v in runtime.values() if isinstance(v, str)
                          for p in [Path(v)] if p.is_file() and p.suffix in {'.npy', '.bin'}}
            stamp = signature(sources, {'benchmark': bcfg, 'method': method,
                                       'profile': profile, 'dataset': dataset,
                                       'embeddings': embeddings, 'threshold': None})
            outputs = {int(n): resolve(profile['edges'][dataset][str(n)]) for n in cfg['target_gene_counts']}
            valid = all(reusable(p, p.with_suffix(p.suffix+'.provenance.json'), stamp) for p in outputs.values())
            if valid and not args.force:
                print(f'[{method}/{dataset}] verified existing candidate scores', flush=True)
                continue
            directories = {p.parent for p in outputs.values()}
            if len(directories) != 1:
                raise ValueError('Both candidate spaces must share one output directory')
            generated = benchmark.infer_targets(paths, dataset, runtime_args, str(checkpoint),
                tuple(outputs), all_candidates=True, output_dir=next(iter(directories)),
                score_accumulation=profile['score_accumulation'])
            for n,p in outputs.items():
                if Path(generated[n]) != p:
                    raise ValueError(f'Configured output differs from exporter: {p}')
                record(p, p.with_suffix(p.suffix+'.provenance.json'), stamp)
                print(p, flush=True)


if __name__ == '__main__':
    main()

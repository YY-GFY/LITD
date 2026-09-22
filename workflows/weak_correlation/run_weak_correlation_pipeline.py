#!/usr/bin/env python3
"""Reproduce the paper weak-correlation comparison with explicit input checks."""
import argparse
import json
import subprocess
import sys
from workflow_common import ROOT, load_config, resolve


def commands(config):
    py = sys.executable
    base = 'workflows/weak_correlation/'
    cfg = load_config(config)
    result = [[py,base+'export_candidate_edges.py','--config',config],
              [py,base+'prepare_existing_edges.py','--config',config]]
    for dataset in cfg['datasets']:
        for count in cfg['target_gene_counts']:
            for script in ['build_edge_features.py','evaluate_weak_signal.py']:
                result.append([py,base+script,'--config',config,'--dataset',dataset,
                               '--target-gene-count',str(count)])
    return result


def missing_inputs(cfg):
    bcfg = load_config(cfg['benchmark_config'])
    required = [resolve(cfg['benchmark_config'])]
    for dataset, ds in cfg['datasets'].items():
        required += [resolve(v) for v in ds.values()]
        for spec in cfg['model_exports'].values():
            base = resolve(spec.get('workdir', bcfg['paths']['workdir'])) / 'datasets' / dataset / 'data'
            required += [base / name for name in ['ExpressionData_filtered.csv', 'cell_pairs_train.csv',
                         'ligand_list.tsv', 'receptor_list.tsv', 'tf_list.tsv', 'filtered_lr_pairs.tsv',
                         'env_context_dict.pt', f'{dataset}_01B-resolution_singlecell_gene_embedding_t4_resolution.npy']]
    required += [resolve(v) for v in cfg['ground_truth'].values()]
    required += [resolve(bcfg['paths'][key]) for key in ['human_tfs', 'scf_gene_list']]
    for spec in cfg['model_exports'].values():
        required += [resolve(v) for v in spec['checkpoints'].values()]
    for mapping in cfg['external_edges'].values():
        for values in mapping.values():
            required += [resolve(v) for v in values.values()]
    return sorted({str(p) for p in required if not p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/weak_correlation.json')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args()
    config = str(resolve(args.config))
    cfg = load_config(config)
    if not args.dry_run:
        missing = missing_inputs(cfg)
        if missing:
            raise SystemExit('Missing weak-correlation inputs (no computation started):\n' + '\n'.join(missing)
                             + '\nSee docs/DATA.md and docs/REPRODUCE.md.')
    if args.check_inputs:
        print('All required input files are present.'); return
    for command in commands(config):
        print(' '.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    if not args.dry_run:
        root = resolve(cfg['root'])
        summary = root / 'results' / 'run_completion.json'
        summary.write_text(json.dumps({'status':'complete','expected_methods':cfg['expected_methods'],
            'datasets':list(cfg['datasets']),'target_gene_counts':cfg['target_gene_counts'],
            'note':'Metrics are newly computed outputs; compare with the immutable paper tables.'},indent=2))


if __name__ == '__main__':
    main()

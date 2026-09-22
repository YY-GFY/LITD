#!/usr/bin/env python3
"""Rebuild the ordered case expression object and convert aligned scFoundation embeddings."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from _config import ROOT, load_config, resolve_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/litd_biological_case.json')
    p.add_argument('--counts', help='Original processed gene-by-cell CSV from the source dataset')
    p.add_argument('--write-scf-input', action='store_true', help='Write the padded cell-by-gene CSV for upstream scFoundation')
    p.add_argument('--embedding', help='Upstream gene embedding .npy, in the supplied cell order')
    p.add_argument('--embedding-cell-ids', help='One cell ID per line, without header, in embedding order')
    args = p.parse_args()
    cfg = load_config(args.config)
    out = Path(resolve_path(cfg['h5ad_file']))
    metadata = pd.read_csv(ROOT/'resources/biological_case/cell_metadata.tsv', sep='\t', index_col='cell_name')
    genes = pd.read_csv(resolve_path(cfg['target_genes_tsv']), sep='\t').gene_name.astype(str).tolist()
    vocabulary = pd.read_csv(ROOT/'resources/reference/OS_scRNA_gene_index.19264.tsv', sep='\t').gene_name.astype(str).tolist()
    if len(metadata) != cfg['n_cells'] or len(genes) != cfg['n_genes']:
        raise ValueError('Case dimensions differ from the fixed paper profile')
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.counts:
        d = pd.read_csv(resolve_path(args.counts), index_col=0)
        d.index = [str(g).split('_')[-1].upper() for g in d.index]
        d.columns = d.columns.astype(str)
        if not d.index.is_unique or not d.columns.is_unique:
            raise ValueError('Duplicate gene symbols or cell IDs; do not silently merge the paper inputs')
        if set(metadata.index) != set(d.columns):
            raise ValueError('Count matrix must contain exactly the 36,044 paper cells')
        missing = sorted(set(genes)-set(d.index))
        if missing:
            raise ValueError(f'Missing selected genes: {missing[:10]}')
        # This order is shared by the supplied metadata, pairs and gene lists.
        x = d.loc[genes, metadata.index].T.to_numpy(dtype=np.float64)
        a = ad.AnnData(x, obs=metadata.copy(), var=pd.DataFrame(index=genes))
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        a.write_h5ad(out)
        print(out)
        if args.write_scf_input:
            padded = d.reindex(vocabulary, fill_value=0).loc[:, metadata.index].T
            padded.to_csv(out.parent/'scfoundation_input.csv')
            (out.parent/'scfoundation_cell_ids.txt').write_text('\n'.join(metadata.index)+'\n')
    if args.embedding:
        if not args.embedding_cell_ids:
            p.error('--embedding requires --embedding-cell-ids to verify cell order')
        ids = Path(resolve_path(args.embedding_cell_ids)).read_text().splitlines()
        if ids != metadata.index.astype(str).tolist():
            raise ValueError('Embedding cell order differs from the paper metadata')
        emb = np.load(resolve_path(args.embedding), mmap_mode='r')
        if emb.shape != (len(metadata), len(vocabulary), cfg['embed_dim']):
            raise ValueError(f'Expected full scFoundation vocabulary embedding, got {emb.shape}')
        index = pd.Index(vocabulary).get_indexer(genes)
        if (index < 0).any():
            raise ValueError('Selected genes absent from scFoundation vocabulary')
        target = Path(resolve_path(cfg['embed_bin']))
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix+'.partial')
        mmap = np.memmap(tmp, mode='w+', dtype=np.float16,
                         shape=(len(metadata), len(genes), cfg['embed_dim']))
        for i in range(len(metadata)):
            mmap[i] = emb[i, index, :]
        mmap.flush(); del mmap
        tmp.replace(target)
        print(target)
    if not args.counts and not args.embedding:
        p.error('Supply --counts and/or --embedding')


if __name__ == '__main__':
    main()

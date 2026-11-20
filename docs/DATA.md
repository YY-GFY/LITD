# Data and frozen inputs

Paths in the configuration files are relative to the repository root unless absolute paths are supplied. Run commands below from the repository root.

## Install included inputs

```bash
python scripts/install_bundled_inputs.py
# Optional: also install the original biological-case model used for Figures 6/S3.
python scripts/install_bundled_inputs.py --paper-checkpoint
# Check every bundled file without installing anything.
python scripts/install_bundled_inputs.py --verify-only
```

The installer verifies SHA-256 hashes, decompresses the files that need installation, and refuses to replace a different existing input. It writes the default `data/` paths used by the configs. The saved case checkpoint is installed only with the explicit flag, at the configured `outputs/biological_case/results/litd_biological_case/models/best.pth` location. Training writes to that location too; use another run/output path if retaining both saved and newly trained models.

The following 50 files are included under `resources/`, with sizes, hashes and installation destinations in `resources/manifest.json`:

| Included category | Contents and provenance |
|---|---|
| hESC/hHep input | Original expression matrices (gzip, no value changes) and GeneOrdering tables recovered from the paper experiment archive. Public source: [BEELINE input data, Zenodo 3378975](https://doi.org/10.5281/zenodo.3378975). |
| Reference networks | STRING, hESC ChIP-seq, HepG2 ChIP-seq and non-specific ChIP-seq networks, plus human-tfs.csv, from the BEELINE data used in the experiments. |
| LR prior | The human CellTalkDB catalogue (`human_lr_pair.tsv`), with its original evidence column. Source: [Shao et al., CellTalkDB](https://doi.org/10.1093/bib/bbaa269). |
| TF and vocabulary tables | The exact `TF_TG_human.csv` input snapshots used for the benchmark and case, and the 19,264-gene scFoundation vocabulary. The TF table's upstream release identifier was not recorded in the recovered files; the frozen input and hash identify the version used here. Do not substitute a newer TF list for checkpoint reproduction. |
| Biological-case metadata | The model-aligned 36,044-cell metadata, temporal pairs, environment ligand context, ligand/receptor/TF lists, filtered LR pairs and ordered 10,112-gene list. Cell metadata contains only cell ID, day, cell line and experiment. |
| Biological-case model | Original trained checkpoint; decompressed SHA-256 `715eedd0b188e6a271aa311fd505eccc2423c190f3db975304e439d4f91f4b9c`. |
| Weak-correlation comparator scores | 28 gzip CSVs: six external methods (PPCOR, GENIE3, GRNBoost2, PIDC, SCODE, SINCERITIES) and the original Full LITD ablation reference, each on two datasets and two target sets. Only Gene1, Gene2 and score are retained; scores are not rounded or thresholded. These are complete saved method outputs, not figure-level summary metrics. |

The bundled comparator files are consumed directly from `resources/weak_correlation/`. They are frozen results of the paper's baseline runs; the weak-correlation workflow recomputes correlation strata and evaluation from these scores rather than retraining those external algorithms. Newly trained LITD and its five ablations are exported separately from checkpoints.

The upstream projects retain their data/software terms. The bundled files identify the experimental versions; this repository does not assign new ownership or licensing to third-party data.

## Inputs still generated or downloaded separately

- **scFoundation**: obtain the [upstream code and pretrained checkpoint](https://github.com/biomap-research/scFoundation), placing its `model/` directory under `data/scfoundation/model/`. Follow the upstream checkpoint setup. The benchmark preparation generates embeddings when no usable precomputed embedding is present, including for hHep. The LITD benchmark and ablation checkpoints are created by the training entry points.
- **Large biological-case count matrix**: the source is [Cuomo et al., 2020](https://doi.org/10.1038/s41467-020-14457-z), with processed data in [Zenodo 3625024](https://zenodo.org/records/3625024). Raw reads are in ENA `ERP016000`; controlled-access records are `EGAS00001002278` and `EGAD00001005741`. Use the study's processed gene-by-cell count CSV containing the 36,044 cells in the included metadata, placed at `data/biological_case/raw_counts.csv`. The historical filename says “raw_counts”, but the supplied study values must be preserved; do not replace them with a differently normalized matrix.
- **Case expression object and embeddings**: the original experiment stores an approximately 2.9 GB ordered expression object and a 373,224,374,272-byte float16 embedding file. These large files are not bundled in ordinary Git. The embedding shape is `(36044, 10112, 512)` in the included cell and gene order. Configure their existing locations in `configs/litd_biological_case.json`, or rebuild as below. A public download archive for these author-derived large files has not yet been assigned; do not interpret the bundled small inputs as a complete large-data download.

## Rebuild the biological-case inputs

```bash
python scripts/prepare_biological_inputs.py \
  --counts data/biological_case/raw_counts.csv --write-scf-input
```

This checks the exact cell set, converts source gene identifiers to uppercase symbols using the original suffix rule, selects the supplied ordered 10,112-gene list, normalizes each cell to 10,000 and applies log1p. It writes the configured H5AD. The optional scFoundation CSV is padded to the original 19,264-gene vocabulary, preserving the source study values before that LITD normalization, and a matching cell-ID list is saved. Large inputs require substantial RAM and disk space.

Run upstream scFoundation gene-embedding extraction on that CSV (single-cell input, gene output, `tgthighres=t4`, `pre_normalized=T`, `version=ce`). Then convert the resulting full-vocabulary `.npy` to the ordered float16 case binary:

```bash
python scripts/prepare_biological_inputs.py \
  --embedding PATH_TO_SCF_GENE_EMBEDDING.npy \
  --embedding-cell-ids data/biological_case/scfoundation_cell_ids.txt
```

The supplied pairs and environment context are the original paper inputs and are installed without recomputation. A fresh upstream embedding run can vary with upstream software/checkpoints; using the original derived binary is required for exact checkpoint reproduction. The reconstruction utility does not claim bitwise equivalence for a newly generated upstream embedding.

`figures/data_used/` contains separate, compact paper-final plotting tables. Figure regeneration requires none of these large input matrices or GPU checkpoints.

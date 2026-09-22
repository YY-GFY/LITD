# LITD

Code and figure-level result tables accompanying the LITD manuscript submitted to *Briefings in Bioinformatics*.

## Repository contents

- `litd/`: LITD model, temporal pairing and benchmark utilities.
- `configs/`: the fixed benchmark and biological-case profiles, their rationale, and the five paper ablations.
- `scripts/`: public training and evaluation entry points.
- `workflows/`: expression prediction, weak-correlation recovery, biological-case and latent-representation analyses.
- `figures/`: approved submission figures, synchronized figure builder and final plotting tables.
- `resources/`: frozen input/reference files and comparator scores, with a SHA-256 manifest.

The repository includes the exact hESC/hHep input matrices, reference tables, biological-case metadata and checkpoint, and saved weak-correlation comparator scores. Run `python scripts/install_bundled_inputs.py` to verify and install the small inputs at their configured paths. Their provenance and the remaining large-data preparation steps are in [`docs/DATA.md`](docs/DATA.md). The 36,044-cell count matrix and large scFoundation embeddings are obtained or generated separately.

Canonical submission figures are stored in `figures/submission/`; their bytes match the approved 20260913 figure package. Rebuilding writes separate files under `figures/generated/` using the synchronized plotting code and tables.
## Installation

```bash
conda create -n litd python=3.12.7
conda activate litd
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The preprocessing step that creates scFoundation embeddings additionally requires the upstream [scFoundation](https://github.com/biomap-research/scFoundation) code and checkpoint.

## Software requirements / tested environment

- Linux
- Python 3.12.7
- PyTorch 2.6.0+cu124
- CUDA 12.4
- NVIDIA RTX 3090 (24 GB)

The exact Python dependencies are listed in [`requirements.txt`](requirements.txt). PyTorch and CUDA builds are platform-specific, so the installed PyTorch build must be compatible with the local NVIDIA driver.

In model state dictionaries, `expr_input_proj.0.weight` and `.bias` are the weight and bias of a `Linear(1, hidden_dim)` layer, while `expr_input_proj.1.weight` and `.bias` are the learnable scale and offset of the following `LayerNorm(hidden_dim)`. This branch converts one scalar expression value per gene into a hidden vector and is used only by the **No scFoundation** ablation (`input_mode: raw_expr`). LITD uses `input_proj` for the 512-dimensional scFoundation embedding instead. The biological-case checkpoint-compatible analysis class retains the raw-expression branch in its state dictionary, but that branch is inactive when `input_mode` is `scfoundation`.

## Computational resources and runtime

The training workflows were tested on a single NVIDIA RTX 3090 GPU with 24 GB of memory. A 24 GB GPU was sufficient for the reported settings. On this hardware, a representative 40-epoch run on the 36,044-cell endoderm dataset takes approximately 5–7 hours; the smaller benchmark datasets complete more quickly. Runtime varies with dataset size, storage speed and GPU model. CPU-only training is supported as a fallback but is not practical for reproducing the full training experiments. Inference, metric evaluation and figure generation require substantially fewer resources.

## Main commands

Run the fixed benchmark model (40 epochs; seed 42):

```bash
python scripts/run_train.py --config configs/litd_benchmark.json
```

Run exactly the five reported ablations:

```bash
python scripts/run_ablations.py --config configs/litd_benchmark.json
```

Train the 36,044-cell biological case with its scale-adjusted profile:

```bash
python scripts/run_biological_case_train.py --config configs/litd_biological_case.json
```

Evaluate saved expression predictions or ranked GRN edges. The paper-benchmark
mode reconstructs the dataset-specific TF/target candidate space and evaluates
all applicable reference networks:

```bash
python scripts/evaluate_expression.py --truth TRUE.npy --prediction PRED.npy
python scripts/evaluate_epr.py --paper-benchmark --config configs/litd_benchmark.json --dataset hesc --target-count 500 --prediction EDGES.csv
```

For an already filtered custom candidate space, provide its dimensions
explicitly (including TF/target overlap when applicable):

```bash
python scripts/evaluate_epr.py --ground-truth GT.csv --prediction EDGES.csv --tf-count N_TF --target-count N_TARGET
```

Run or inspect the manuscript workflows through one entry point:

```bash
python scripts/run_workflows.py --workflow all --dry-run
python scripts/run_workflows.py --workflow expression
python scripts/run_workflows.py --workflow expression --skip-external
python scripts/run_workflows.py --workflow biological-case
python scripts/run_workflows.py --workflow weak-correlation
```

The expression workflow uses one shared 90/10 split for LITD and all five reported comparators. PRESCIENT is installed from its upstream repository rather than vendored. The relay validation uses **Mean mismatch**: evaluate the two alternative source-stage environments separately and average their InternalCascade scores. The biological-case workflow includes the global PathScore search, the fixed `MDK -> ITGB1 -> PTMA -> LIMD1` Figure 6 analysis, the distinct-TF relay search and its configured internal validation.

Rebuild all main and supplementary figures from the included figure-level tables:

```bash
python figures/build_paper_figures.py
```

Workflow details and the manuscript-to-code map are in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Ablations

The public manifest contains only: **Shuffled temporal order**, **Static reconstruction**, **No ligand context**, **No LR mask**, and **No scFoundation**. The full model is the reference, not an ablation.

## Citation

Please cite the LITD paper. Full bibliographic information will be added after publication.

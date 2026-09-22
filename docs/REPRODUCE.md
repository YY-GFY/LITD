# Reproduction map

| Manuscript analysis | Configuration | Entry point |
|---|---|---|
| hESC/hHep benchmark | `configs/litd_benchmark.json` | `scripts/run_train.py` |
| Five ablations | `configs/ablation_manifest.csv` | `scripts/run_ablations.py` |
| Expression prediction | `configs/expression_prediction.json` | `workflows/expression_prediction/run_expression_pipeline.py` |
| Correlation-stratified recovery | `configs/weak_correlation.json` | `workflows/weak_correlation/` |
| 36,044-cell biological-case training | `configs/litd_biological_case.json` | `scripts/run_biological_case_train.py` |
| Global PathScore and Figure 6 fixed-path validation | `configs/figure6_fixed_path.json` | `workflows/biological_case/run_biological_case_pipeline.py` |
| Distinct-TF relay and internal validation | `configs/relay_internal_validation.json` | `workflows/biological_case/run_biological_case_pipeline.py` |
| Hierarchical latent representations | `configs/latent_interpretability.json` | `workflows/latent_interpretability/` |
| Main and supplementary figures | included plot tables | `figures/build_paper_figures.py` |

The benchmark profile uses hidden dimension 160, four heads, two TF layers, two gene layers, MLP ratio 3, dropout 0.25, AdamW learning rate `1e-4`, weight decay `0.015`, entropy coefficient `1e-4`, Huber delta `0.5`, gradient clipping at `1.0`, batch size 4, 40 epochs and seed 42. The 36,044-cell biological-case profile keeps the same optimization and regularization settings but uses hidden dimension 128 and one gene layer. This scale-aware reduction limits memory and training cost for a dataset that is much larger than hESC/hHep while retaining the same LITD mechanism and the settings used for the reported biological analysis. The two JSON files remain separate because the benchmark uses a shared two-dataset interface, whereas the biological case uses one large temporal dataset with different data layout, checkpoint and post-hoc output paths. They are dataset-specific profiles of one method, not distinct model versions. `latent_interpretability.json` inherits the biological-case model and data settings and adds only post-hoc sampling and dimensionality-reduction options.

Internal run identifiers remain only in supplied result tables where they support provenance; they are not model names. The name **Full LITD** is reserved for comparisons between the complete model and its ablations; elsewhere the method is called **LITD**.

Weak-correlation strata use absolute Pearson correlation: weak `<0.1`, medium `[0.1,0.3)`, and strong `>=0.3`. Within each stratum, the final evaluation selects exact top K after deterministic sorting by score descending, TF ascending and target gene ascending.

For the standard hESC/hHep evaluation, use `scripts/evaluate_epr.py --paper-benchmark` with the benchmark config, dataset name, target count and saved ranked-edge file. This mode rebuilds the same dataset-specific candidate space used by the paper and evaluates all applicable reference networks. The lower-level mode without `--paper-benchmark` is intended only for inputs that have already been filtered to a known candidate space; its TF count, target count and overlap count must be supplied explicitly.

PathScore is computed as `Expr × positive LR weight × receptor-to-TF attention × TF-to-gene attention`. The global analysis retains the top 1,000 paths after per-TF top-1,000 pruning; relay analysis requires distinct TFs across consecutive segments.

## Unified entry point

Print the complete command plan without reading data or using a GPU:

```bash
python scripts/run_workflows.py --workflow all --dry-run
```

Run one workflow with its fixed public configuration:

```bash
python scripts/run_workflows.py --workflow expression
python scripts/run_workflows.py --workflow expression --skip-external
python scripts/run_workflows.py --workflow biological-case
python scripts/run_workflows.py --workflow weak-correlation
python scripts/run_workflows.py --workflow latent-interpretability
```

The unified entry point stops on the first failed stage. Existing outputs are not silently substituted for missing computations.

## Expression prediction

The expression workflow prepares one shared dataset tree and one fixed 90/10 pair split (`seed=42`) for all methods. Every method writes `test_predictions.npy`, `test_targets.npy`, and `test_indices.npy` under `outputs/expression_prediction/results/<method>/<dataset>/predictions/`. Metrics are recomputed from those arrays instead of trusting a mutable method-specific summary.

```bash
# Inspect commands only
python workflows/expression_prediction/run_expression_pipeline.py --dry-run

# Prepare data and the shared split
python workflows/expression_prediction/run_expression_pipeline.py --stage prepare

# Run the repository-native methods
python workflows/expression_prediction/run_expression_pipeline.py --stage methods \
  --methods LITD MLP persistence scFoundation-MLP "stage mean" --skip-external

# Validate and combine available prediction matrices
python workflows/expression_prediction/run_expression_pipeline.py --stage collect --skip-external
```

PRESCIENT is an upstream method and is not vendored. Install its source at the configured `prescient.source_dir`; the included `run_prescient.py` adapter consumes the shared indices and writes the same prediction contract. A complete six-method run fails explicitly when the upstream source is absent. The immutable table in `figures/data_used/expression__expression_prediction_paper_results.csv` remains the provenance source for Supplementary Figure S1. Newly generated values are written under `outputs/` and never overwrite that table.

## Figure 6 fixed path

The global search ranks the full candidate space using CPU float64 arithmetic. The fixed-path stage then verifies the configured rank and evaluates exactly `MDK -> ITGB1 -> PTMA -> LIMD1`; it does not select a path from marker lists, TF priority lists or a development shortlist.

```bash
python workflows/biological_case/run_biological_case_pipeline.py --stage windows
python workflows/biological_case/run_biological_case_pipeline.py --stage pathscore
python workflows/biological_case/run_biological_case_pipeline.py --stage figure6
```

The Figure 6 stage writes the three-window stage profile, per-window zero-MDK prediction and InternalCascade changes, all six off-diagonal stage-mismatch combinations, their per-window Mean mismatch summary (`stage_mismatch_mean_summary.csv`), experiment-cluster bootstrap intervals, a long audit table and an interpretation-boundary table. Counterfactuals use full-precision forward passes. They test model-internal conditional dependence and do not replace the original autocast artifacts used for global ranking.

## Distinct-TF relay

The public relay manifest fixes the three rank-1 segments reported in Supplementary Figure S3. The internal-validation configuration contains the required paths, expected sample counts and dimensions, batch size and support threshold.

```bash
python workflows/biological_case/run_biological_case_pipeline.py --stage relay
python workflows/biological_case/run_biological_case_pipeline.py --stage relay-validation
```

Use `--stage all` only after the biological-case checkpoint and required inputs are present. The superseded non-distinct relay entry point was removed; `relay_search_common.py` now contains only the numerical Top-K helpers required by the reported distinct-TF search.


## Weak-correlation pipeline

Install the bundled inputs first (see `DATA.md`), then train the benchmark and the five ablations using the README commands. The unified weak-correlation entry point performs these steps in order: export complete LITD/ablation candidate scores, standardize the saved comparator scores, compute source-cell correlations and paired-cell features, and evaluate all 15 methods on each of the four dataset/target configurations.

```bash
python workflows/weak_correlation/run_weak_correlation_pipeline.py --dry-run
python workflows/weak_correlation/run_weak_correlation_pipeline.py --check-inputs
python scripts/run_workflows.py --workflow weak-correlation
# Export only one model if desired; this does not retrain it.
python workflows/weak_correlation/export_candidate_edges.py --methods LITD
```

The ordinary thresholded benchmark CSV is **not** a weak-correlation input. The exporter writes `*_all_candidate_edges.csv.gz` beneath `outputs/weak_correlation/inputs/`, keeping every non-self candidate edge regardless of score. It uses the first 100 pair records, the paper TF/target universe, and each checkpoint's dataset/pair ordering. The main benchmark export preserves the original native-precision concatenation/mean; ablation exports preserve the original float32 batch-sum/sample-count calculation. Deterministic sorting is score descending, TF ascending, target ascending. Provenance sidecars check checkpoint, small input and code hashes (and embedding size/mtime) before reusing outputs. Use `--force` after replacing an embedding in a way that preserves both size and mtime.

Six external baseline outputs and the **Full LITD** ablation reference are frozen, original score files under `resources/weak_correlation/`. Full LITD is a separately saved reference run and is not a duplicate of the benchmark LITD output. This workflow re-evaluates those archived scores; it does not train the external methods. Newly trained/exported LITD results can differ from the historical figure tables. Missing files or methods stop the run rather than silently producing a partial paper comparison.

## Mean mismatch and approved figure files

Both Figure 6C and Supplementary Figure S3C use **Mean mismatch**. For each source window, hold the source cells fixed, run the model separately under each of the two other source-stage ligand environments, compute each InternalCascade, and average the two scores with equal weight. This is neither a single cyclic mismatch nor a forward pass through an averaged ligand vector. The matched/mean-mismatch ratio is calculated after averaging the two scores. Relay validation retains the original autocast convention; Figure 6 counterfactuals retain their original full-precision convention.

The frozen S3 ratios are **2.47, 1.06 and 127.70**. Segment 2 is only 5.81% above the mean mismatch and is below its day-2 mismatch score. The six off-diagonal results are included in `figures/data_used/relay_mean_mismatch__internal_cascade_counterfactuals_long.csv`; model-internal sensitivity does not establish causality.

`figures/submission/` contains exact copies of all six main and five supplementary figures from the approved 20260913 package. `figures/submission/manifest.json` records their SHA-256 hashes. `python figures/build_paper_figures.py` reproduces the corresponding content and panels under `figures/generated/`; font/rendering-library differences can change raster pixels or PDF bytes. It never overwrites the canonical submission files. Figure 6 includes all three window comparisons and S4 includes the approved three-panel layout.

# Supplementary Figure S5. Hierarchical latent representation interpretability

## Panel scaffold

- A: frozen scFoundation gene representation UMAP using all 24,425 unique source cells
- B: projected gene representation UMAP using all 24,425 unique source cells
- C: TF-informed gene representation UMAP using all 24,425 unique source cells
- D: final gene representation UMAP using all 24,425 unique source cells
- E: stage silhouette score, balanced kNN accuracy and PC1-day Spearman across latent representations

Panels F-H from the former version were removed.

## Original data

- `full_cell_umap_coordinates`
  - remote original: the archived full-cell UMAP coordinate snapshot
  - local snapshot: `/Users/guofeiyu/Desktop/WOT_GRN_article/LITD_Github/figures/data_used/interpretability_cell_umap_coordinates_full_24425__cell_umap_coordinates_full_24425.csv`
  - rows: `97,700` (`24,425 cells × 4 layers`)
  - SHA-256: `5f5b9d6a8d6dd9297de27536b3a070d7cd8b45ed0bfd43b8e9a7d355b663269f`
- `cell_representation_metrics`
  - local snapshot: `/Users/guofeiyu/Desktop/WOT_GRN_article/LITD_Github/figures/data_used/interpretability_cell_representation_metrics__cell_representation_metrics.csv`
  - SHA-256: `22e4aba2cfe6c1b4d2ea075db95fd60d7894599c0972c69a2b2cb03d7703cbb2`
- `full_cell_umap_script`
  - local scaffold: `/Users/guofeiyu/Desktop/WOT_GRN_article/LITD_Github/figures/generated/scaffolds/Supplementary_Figure_S5/compute_full_cell_umap.py`

## UMAP parameters

- Each layer was fitted independently on all 24,425 unique source cells.
- `n_neighbors=30`, `min_dist=0.3`, `metric=cosine`, base seed `42` with deterministic layer offsets.
- UMAP coordinates are visualization-only. Panel E uses the original latent-space metrics and not the UMAP coordinates.

## Evidence boundary

This figure provides model-internal representation evidence. It shows developmental-state organization in learned latent spaces and does not establish a causal biological mechanism.

## Rebuild

Run `python build_all_figures.py` from this directory after the full-cell coordinate snapshot is present.

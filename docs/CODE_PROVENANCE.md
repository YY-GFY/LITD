# Code provenance

The repository was assembled from the benchmark, ablation, biological-case, relay, weak-correlation and latent-analysis sources used for the manuscript. Development search grids and superseded public entry points were removed. Server-specific paths were replaced by repository-relative configuration values, and the benchmark and 36,044-cell biological-case workflows are presented as dataset-specific profiles of LITD.

Internal run identifiers are retained only in the supplied result tables and provenance-path fields when needed to trace a reported value back to its originating artifact. Public class names, comments, commands and plotting logic use method or analysis names instead of internal run numbers.

Expression-prediction methods are joined through a fixed split and shared prediction-array contract rather than mutable server summary files. The supplied Supplementary Figure S1 table is an immutable result snapshot; reruns are written separately under `outputs/expression_prediction`.

The Figure 6 path is fixed in `configs/figure6_fixed_path.json` after the full-space ranking. Its workflow contains no marker panel, TF priority, generic-gene filter or external-evidence reranking. The distinct-TF relay is likewise fixed by `configs/relay_selected_manifest.csv` before internal validation.

The relay Mean mismatch implementation was ported from the separately verified 20260919 two-alternative-stage analysis. Plotting logic and source tables were synchronized with the approved 20260913 figure package; canonical exported figures are preserved separately from regenerated outputs. The weak-correlation adapter now explicitly exports unthresholded candidates and retains the distinct averaging conventions of the historical main-model and ablation exporters. Frozen comparator scores were recovered from the original experiment archive, reduced to the three required columns without score rounding, and assigned content hashes.

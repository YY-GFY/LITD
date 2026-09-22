#!/usr/bin/env python3
"""Reproducible vector figure builder for the current LITD manuscript plan."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

LOCAL_DEPS = Path(__file__).resolve().parent / ".pydeps"
if LOCAL_DEPS.exists() and os.environ.get("LITD_USE_LOCAL_DEPS") == "1":
    sys.path.insert(0, str(LOCAL_DEPS))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from pypdf import PageObject, PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "generated"
MAIN = OUT / "main"
SUPP = OUT / "supplementary"
DATA = Path(__file__).resolve().parent / "data_used"
SCAFF = OUT / "scaffolds"
RENDER = OUT / "qa_render"
FINAL_WIDTH_IN = 7.15
MIN_TEXT_PT = 5.8
A4_WIDTH_PT = 595.276
A4_HEIGHT_PT = 841.890
FILES = {
    'plan': ROOT / "docs/REPRODUCE.md",
    'figure1_original_pdf': DATA / 'figure1_original_pdf__Figure1_LITD_model_architecture.pdf',
    'figure1_original_svg': DATA / 'figure1_original_svg__Figure1_LITD_model_architecture.svg',
    'figure1_original_png': DATA / 'figure1_original_png__Figure1_LITD_model_architecture_600dpi.png',
    'architecture_svg': DATA / 'architecture_svg__LITD架构图0812_矢量版.svg',
    'architecture_ai': DATA / 'architecture_ai__LITD架构图0812_矢量版.ai',
    'benchmark': DATA / 'benchmark__Supplementary_Table_S1_human_benchmark_metrics.csv',
    'weak_signal_metrics': DATA / 'weak_signal_metrics__all_weak_signal_metrics.csv',
    'weak_signal_validation': DATA / 'weak_signal_validation__validation_report.json',
    'weak_signal_config': ROOT / 'configs/weak_correlation.json',
    'weak_signal_readme': ROOT / 'docs/REPRODUCE.md',
    'ablation_full_hesc': DATA / 'ablation_full_hesc__hesc_full_summary.csv',
    'ablation_full_hhep': DATA / 'ablation_full_hhep__hhep_full_summary.csv',
    'ablation_no_ligand_hesc': DATA / 'ablation_no_ligand_hesc__hesc_no_ligand_input_summary.csv',
    'ablation_no_ligand_hhep': DATA / 'ablation_no_ligand_hhep__hhep_no_ligand_input_summary.csv',
    'ablation_no_lr_hesc': DATA / 'ablation_no_lr_hesc__hesc_no_lr_prior_summary.csv',
    'ablation_no_lr_hhep': DATA / 'ablation_no_lr_hhep__hhep_no_lr_prior_summary.csv',
    'ablation_no_scfoundation_hesc': DATA / 'ablation_no_scfoundation_hesc__hesc_no_scfoundation_summary.csv',
    'ablation_no_scfoundation_hhep': DATA / 'ablation_no_scfoundation_hhep__hhep_no_scfoundation_summary.csv',
    'ablation_same_time_hesc': DATA / 'ablation_same_time_hesc__hesc_same_time_reconstruction_summary.csv',
    'ablation_same_time_hhep': DATA / 'ablation_same_time_hhep__hhep_same_time_reconstruction_summary.csv',
    'ablation_shuffled_hesc': DATA / 'ablation_shuffled_hesc__hesc_shuffled_temporal_summary.csv',
    'ablation_shuffled_hhep': DATA / 'ablation_shuffled_hhep__hhep_shuffled_temporal_summary.csv',
    'backbone': DATA / 'backbone__final_backbone_evidence_table.csv',
    'backbone_support': DATA / 'backbone_support__cd44_foxa2_tdgf1_backbone_experiment_support_long.csv',
    'relay_manifest': DATA / 'relay_manifest__selected_relay_manifest.csv',
    'relay_ranked_top1000': DATA / 'relay_ranked_top1000__full_structural_relay_distinct_tf_top1000.csv',
    'relay_stage_specificity': DATA / 'relay_stage_specificity__stage_specificity_summary.csv',
    'relay_counterfactuals': DATA / 'relay_counterfactuals__internal_cascade_summary.csv',
    'relay_interpretation_boundary': DATA / 'relay_interpretation_boundary__interpretation_boundary.json',
    'expression': DATA / 'expression__expression_prediction_all_results_with_mlp_c38parammatch_100ep_20260519.csv',
    'weighted_path_top1000': DATA / 'weighted_path_top1000__global_weighted_pathscore_top1000.csv',
    'weighted_path_winner_windows': DATA / 'weighted_path_winner_windows__winner_three_window_decomposition.csv',
    'weighted_path_metadata': DATA / 'weighted_path_metadata__analysis_metadata.json',
    'weighted_path_validation': DATA / 'weighted_path_validation__validation_report.json',
    'mdk_stage_specificity': DATA / 'mdk_stage_specificity__stage_specificity_summary.csv',
    'mdk_zero_ligand': DATA / 'mdk_zero_ligand__zero_MDK_counterfactual.csv',
    'mdk_stage_mismatch': DATA / 'mdk_stage_mismatch__stage_mismatch_matrix.csv',
    'mdk_interpretation_boundary': DATA / 'mdk_interpretation_boundary__interpretation_boundary.csv',
    'mdk_internal_validation_report': DATA / 'mdk_internal_validation_report__validation_report.json',
    'scale': DATA / 'scale__cross_dataset_scale_context_tidy.csv',
    'scale_training_sizes': DATA / 'scale_training_sizes__scale_training_sizes_verified_20260825.csv',
    'window_shortlist': DATA / 'window_shortlist__window_axis_shortlist.csv',
    'window_candidates': DATA / 'window_candidates__window_axis_candidates.csv',
    'relay_chains': DATA / 'relay_chains__relay_chains.csv',
    'relay_edges': DATA / 'relay_edges__relay_edges.csv',
}

COLORS = {
    "litd": "#1f4e79", "backbone": "#25855a", "entry": "#e67e22",
    "relay1": "#b13b48", "relay2": "#4069a1", "relay3": "#7652a7",
    "real": "#1f4e79", "mismatch": "#8a94a3", "zero": "#d6d9de",
    "ink": "#18212b", "grid": "#dfe5eb", "pale": "#f5f7fa",
}

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 8.2, "axes.titlesize": 7.8,
    "axes.labelsize": 8.2, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def ensure_dirs():
    for p in [OUT, MAIN, SUPP, DATA, SCAFF]:
        p.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in FILES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing source files:\n" + "\n".join(missing))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_sources():
    rows = [{"key": key, "source": str(src.relative_to(ROOT)),
             "bytes": src.stat().st_size, "sha256": sha256(src)}
            for key, src in FILES.items()]
    pd.DataFrame(rows).to_csv(OUT / "source_manifest.csv", index=False)


def panel(ax, label, title=None):
    ax.text(-0.075, 1.055, label, transform=ax.transAxes, fontsize=9.5,
            weight="bold", va="top")
    if title:
        ax.set_title(title, loc="left", pad=6, fontsize=7.8, weight="normal")


def clean(ax):
    ax.grid(axis="y", color=COLORS["grid"], lw=.6, alpha=.8)
    ax.set_axisbelow(True)


def box(ax, xy, wh, text, fc, ec=None, fs=8, weight="normal", radius=.02):
    x, y = xy; w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={radius}",
                           transform=ax.transAxes, fc=fc, ec=ec or fc, lw=1.1)
    ax.add_patch(patch)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", transform=ax.transAxes,
            fontsize=fs, weight=weight, color=COLORS["ink"])
    return patch


def arrow(ax, start, end, color=COLORS["ink"], lw=1.4, rad=0):
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle="-|>",
                                mutation_scale=11, lw=lw, color=color,
                                connectionstyle=f"arc3,rad={rad}"))


def note_for(name, title, inputs, panels, caveat):
    d = SCAFF / name
    d.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "## Panel scaffold", ""] + [f"- {x}" for x in panels]
    lines += ["", "## Original data", ""]
    for key in inputs:
        src = FILES[key]
        snap = DATA / f"{key}__{src.name}"
        lines += [f"- `{key}`", f"  - original: `{src}`", f"  - local snapshot: `{snap}`",
                  f"  - SHA-256: `{sha256(src)}`"]
    lines += ["", "## Evidence boundary", "", caveat, "", "## Rebuild", "",
              "Run `python figures/build_paper_figures.py` from the repository root.", ""]
    text = "\n".join(lines)
    (d / "README_原始数据与scaffold.md").write_text(text, encoding="utf-8")
    (OUT / f"{name}_原始数据说明.md").write_text(text, encoding="utf-8")


def save(fig, folder, stem):
    original_width, original_height = fig.get_size_inches()
    target_height = original_height * FINAL_WIDTH_IN / original_width
    fig.set_size_inches(FINAL_WIDTH_IN, target_height, forward=True)
    for text_artist in fig.findobj(match=mpl.text.Text):
        if text_artist.get_text() and text_artist.get_fontsize() < MIN_TEXT_PT:
            text_artist.set_fontsize(MIN_TEXT_PT)
    fig.savefig(folder / f"{stem}.pdf")
    fig.savefig(folder / f"{stem}.svg")
    fig.savefig(folder / f"{stem}_600dpi.png", dpi=600)
    shutil.copy2(folder / f"{stem}.pdf", folder / f"{stem}.ai")
    plt.close(fig)


def make_figure1():
    stem = "Figure1_LITD_model_architecture"
    # Keep the three representation-update modules visually separate from the
    # shared input and two readouts, which occupy one aligned bottom row.
    fig, ax = plt.subplots(figsize=(7.15, 5.72))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9.0); ax.axis("off")
    receptor_c, tf_c, gene_c, neutral_c = "#2F7D4A", "#B33A3A", "#2F5EA8", "#555555"
    receptor_fill, tf_fill, gene_fill = "#EDF6EF", "#FAEEEE", "#EDF2FA"
    module_shift = .68

    def compact_box(x, y, w, h, text, *, edge=neutral_c, fill="white",
                    fs=6.6, weight="normal", detail=None, text_color="black", lw=1.0,
                    fixed=False):
        if not fixed:
            y += module_shift
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.022,rounding_size=0.075",
                               facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3)
        ax.add_patch(patch)
        ax.text(x+w/2, y+h/2 + (.10 if detail else 0), text,
                ha="center", va="center", fontsize=fs, weight=weight,
                color=text_color, zorder=4)
        if detail:
            ax.text(x+w/2, y+.10, detail, ha="center", va="bottom",
                    fontsize=5.8, color="black", zorder=4)
        return patch

    def compact_module(x, title, color):
        ax.add_patch(FancyBboxPatch((x, 1.42 + module_shift), 3.45, 6.25,
                                   boxstyle="round,pad=0.025,rounding_size=0.11",
                                   facecolor="white", edgecolor=color,
                                   linewidth=1.35, zorder=0))
        ax.text(x+1.725, 7.28 + module_shift, title, ha="center", va="center",
                fontsize=7.2, weight="bold", color=color)

    def compact_arrow(x1, y1, x2, y2, *, zorder=2, fixed=False,
                      mutation_scale=9.5):
        if not fixed:
            y1 += module_shift
            y2 += module_shift
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                    mutation_scale=mutation_scale, linewidth=1.0,
                                    color="black", shrinkA=0, shrinkB=0,
                                    zorder=zorder))

    def compact_poly(points, *, zorder=2, fixed=False, mutation_scale=9.5):
        if not fixed:
            points = [(x, y + module_shift) for x, y in points]
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points)-1))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=mutation_scale,
                                    linewidth=1.0, color="black", joinstyle="miter",
                                    capstyle="projecting", shrinkA=0, shrinkB=0,
                                    zorder=zorder))

    compact_module(.18, "Receptor representation update", receptor_c)
    compact_module(4.00, "TF representation update", tf_c)
    compact_module(7.82, "Gene representation update", gene_c)

    # Receptor update and ligand-conditioned entry.
    compact_box(.45, 1.74, 1.25, .70, "Ligand context", edge=receptor_c,
                fill=receptor_fill, fs=6.1, detail=r"$B\times N_L$")
    compact_box(.45, 2.74, 1.25, .78, r"$W_{LR}\odot M_{LR}$", edge=receptor_c,
                fill=receptor_fill, fs=7.0, detail=r"$N_L\times N_R$", text_color=receptor_c)
    compact_box(.55, 4.45, 1.05, .75, r"$A_R$", edge=receptor_c,
                fill=receptor_fill, fs=7.3, detail=r"$B\times N_R$", text_color=receptor_c, lw=1.2)
    ax.text(1.08, 6.18, "LR-prior-guided\nreceptor activation", ha="center", va="center",
            fontsize=5.2, weight="bold", color=receptor_c, linespacing=1.05)
    compact_box(2.14, 1.74, 1.20, .74, r"$H_R^{(0)}$", edge=receptor_c,
                fill=receptor_fill, fs=7.3, detail=r"$B\times N_R\times h$", text_color=receptor_c, lw=1.2)
    compact_box(1.95, 4.48, 1.50, .96, "FiLM modulation", fs=6.0,
                detail=r"$\gamma_R,\,\beta_R=f(A_R)$")
    compact_box(2.14, 6.18, 1.20, .74, r"$H_R^{(1)}$", edge=receptor_c,
                fill=receptor_fill, fs=7.3, detail=r"$B\times N_R\times h$", text_color=receptor_c, lw=1.2)
    compact_arrow(1.075, 2.44, 1.075, 2.74); compact_arrow(1.075, 3.52, 1.075, 4.45)
    compact_arrow(1.60, 4.825, 1.95, 4.825)
    compact_arrow(2.74, 2.48, 2.74, 4.48); compact_arrow(2.70, 5.44, 2.70, 6.18)

    # Matched TF and gene update blocks.
    for x, sub, shape, cross, detail, color, fill in [
        (4.00, "TF", r"$B\times N_{TF}\times h$", "R → TF", "Q: TF; K,V: R", tf_c, tf_fill),
        (7.82, "G", r"$B\times N_G\times h$", "TF → Gene", "Q: Gene; K,V: TF", gene_c, gene_fill),
    ]:
        compact_box(x+1.12, 1.52, 1.22, .74, rf"$H_{{{sub}}}^{{(0)}}$",
                    edge=color, fill=fill, fs=7.3, detail=shape, text_color=color, lw=1.2)
        compact_box(x+.68, 2.62, 2.10, .90, f"Cross-attention\n{cross}", fs=6.5, detail=detail)
        compact_box(x+.92, 3.92, 1.62, .60, "Residual + LN", fs=6.2)
        compact_box(x+.68, 4.88, 2.10, .90, "Self-attention + FFN", fs=6.3, detail="blocks ×2")
        compact_box(x+1.12, 6.08, 1.22, .76, rf"$H_{{{sub}}}^{{(1)}}$",
                    edge=color, fill=fill, fs=7.3, detail=shape, text_color=color, lw=1.2)
        compact_arrow(x+1.73, 2.26, x+1.73, 2.62)
        compact_arrow(x+1.73, 3.52, x+1.73, 3.92)
        compact_arrow(x+1.73, 4.52, x+1.73, 4.88)
        compact_arrow(x+1.73, 5.78, x+1.73, 6.08)

    compact_poly([(3.34, 6.55), (4.45, 6.55), (4.45, 3.07), (4.68, 3.07)], zorder=1)
    compact_poly([(6.34, 6.46), (8.27, 6.46), (8.27, 3.07), (8.50, 3.07)], zorder=1)

    # The input and both outputs share one bottom row. Readout routes leave the
    # gene module on its right, then descend without crossing the model blocks.
    compact_box(.25, .20, 5.15, .90,
                "Frozen per-cell scFoundation gene embeddings",
                fs=6.25, weight="bold", detail=r"$E:\ B\times N_G\times 512$", fixed=True)
    compact_box(5.70, .20, 2.40, .90, "GRN inference",
                fs=6.4, weight="bold", fixed=True)
    compact_box(8.40, .20, 2.40, .90, "Next-state\nprediction",
                fs=6.4, weight="bold", fixed=True)
    compact_poly([(10.16, 6.46 + module_shift), (10.72, 6.46 + module_shift),
                  (10.72, 1.10)], zorder=6, fixed=True, mutation_scale=13.0)
    ax.plot([10.45, 10.45, 6.90], [2.62 + module_shift, 1.42, 1.42],
            color="black", linewidth=1.0, solid_joinstyle="miter", zorder=5)
    compact_arrow(6.90, 1.42, 6.90, 1.10, zorder=6, fixed=True,
                  mutation_scale=13.0)

    ax.text(2.55, 1.80, "Linear + LN + GELU + Dropout",
            ha="right", va="center", fontsize=5.3,
            bbox={"facecolor":"white", "edgecolor":"none", "pad":.35})
    # One unbroken input trunk meets a horizontal branch line; three arrowed
    # stems then feed the receptor, TF and gene initial representations.
    ax.plot([2.74, 2.74], [1.10, 1.55], color="black", linewidth=1.0, zorder=1)
    ax.plot([2.74, 9.55], [1.55, 1.55], color="black", linewidth=1.0, zorder=1)
    compact_arrow(2.74, 1.55, 2.74, 1.74 + module_shift, fixed=True)
    compact_arrow(5.73, 1.55, 5.73, 1.52 + module_shift, fixed=True)
    compact_arrow(9.55, 1.55, 9.55, 1.52 + module_shift, fixed=True)
    save(fig, MAIN, stem)
    note_for("Figure1", "Figure 1. LITD model architecture", ["plan"],
             ["Compact receptor, TF and gene representation updates with the shared input and both readouts aligned below"],
             "This is a model schematic, not a quantitative result. The compact layout preserves the original computation order and tensor roles while separating the model blocks from the bottom input-output row.")
    return

    fig, ax = plt.subplots(figsize=(7.15, 4.60))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8.2); ax.axis("off")

    outer_lw, op_lw, state_lw, arrow_lw = 1.55, 1.05, 1.30, 1.10
    detail_fill = "#fafafa"
    receptor_c, tf_c, gene_c, neutral_c = "#2F7D4A", "#B33A3A", "#2F5EA8", "#555555"
    receptor_fill, tf_fill, gene_fill = "#EDF6EF", "#FAEEEE", "#EDF2FA"

    def box(x, y, w, h, text, *, kind="operation", fontsize=8.6,
            weight="normal", detail=None, edgecolor=neutral_c,
            facecolor=None, textcolor="black", zorder=3):
        lw = state_lw if kind == "state" else op_lw
        fill = facecolor if facecolor is not None else (detail_fill if kind == "detail" else "white")
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.025,rounding_size=0.085",
                               facecolor=fill, edgecolor=edgecolor, linewidth=lw, zorder=zorder)
        ax.add_patch(patch)
        ty = y + h / 2 + ((0.13 if kind == "state" else 0.10) if detail else 0)
        ax.text(x + w/2, ty, text, ha="center", va="center",
                fontsize=fontsize, weight=weight, family="sans-serif", color=textcolor, zorder=zorder+1)
        if detail:
            detail_y = y + (0.07 if kind == "state" else 0.18)
            ax.text(x + w/2, detail_y, detail, ha="center", va="bottom",
                    fontsize=5.8, family="sans-serif", color="black", zorder=zorder+1)
        return patch

    def module(x, title, color, width=3.7):
        ax.add_patch(FancyBboxPatch((x, 1.15), width, 6.35,
                                   boxstyle="round,pad=0.03,rounding_size=0.12",
                                   facecolor="white", edgecolor=color,
                                   linewidth=outer_lw, zorder=0))
        ax.text(x + width/2, 6.98, title, ha="center", va="center",
                fontsize=8.2, weight="bold", family="sans-serif", color=color)

    def arrow(x1, y1, x2, y2, *, connectionstyle="arc3", zorder=2,
              shrink_a=2.0, shrink_b=2.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                    arrowstyle="-|>", mutation_scale=10.5,
                                    linewidth=arrow_lw, color="black",
                                    connectionstyle=connectionstyle,
                                    shrinkA=shrink_a, shrinkB=shrink_b, zorder=zorder))

    def poly_arrow(points, *, zorder=2):
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=10.5,
                                    linewidth=arrow_lw, color="black",
                                    joinstyle="miter", capstyle="projecting", zorder=zorder))

    # Three representation-update stages use one visual grammar; outputs remain frameless.
    module(0.25, "Receptor\nrepresentation\nupdate", receptor_c)
    module(4.20, "TF representation\nupdate", tf_c)
    module(8.15, "Gene representation\nupdate", gene_c)

    # Receptor branch: both the hidden-state lane and activation lane read bottom-to-top.
    box(2.62, 1.48, 1.05, 0.58, r"$H_R^{(0)}$", kind="state", fontsize=7.5,
        detail=r"$B\times N_R\times h$", edgecolor=receptor_c,
        facecolor=receptor_fill, textcolor=receptor_c)
    box(2.20, 3.78, 1.70, 0.92, "FiLM\nmodulation", fontsize=6.4,
        detail=r"$\gamma_R,\,\beta_R=f(A_R)$")
    box(2.62, 5.50, 1.05, 0.58, r"$H_R^{(1)}$", kind="state", fontsize=7.5,
        detail=r"$B\times N_R\times h$", edgecolor=receptor_c,
        facecolor=receptor_fill, textcolor=receptor_c)
    arrow(3.15, 2.06, 3.15, 3.78); arrow(3.15, 4.70, 3.15, 5.50)

    ax.text(1.24, 4.78, "LR-prior–guided\nreceptor activation",
            ha="center", va="center", fontsize=5.8, weight="bold",
            family="sans-serif", color=receptor_c)
    box(0.48, 1.48, 1.52, 0.58, "Ligand context", kind="detail", fontsize=6.0,
        detail=r"$B\times N_L$", edgecolor=receptor_c, facecolor=receptor_fill)
    box(0.48, 2.62, 1.52, 0.67, r"$W_{LR}\odot M_{LR}$", kind="detail", fontsize=7.0,
        detail=r"$N_L\times N_R$", edgecolor=receptor_c,
        facecolor=receptor_fill, textcolor=receptor_c)
    ax.text(1.24, 3.55, r"$M_{LR}$: known LR-pair mask", ha="center", va="center",
            fontsize=5.8, family="sans-serif", color=receptor_c)
    box(0.78, 3.95, 0.92, 0.58, r"$A_R$", kind="state", fontsize=7.5,
        detail=r"$B\times N_R$", edgecolor=receptor_c,
        facecolor=receptor_fill, textcolor=receptor_c)
    arrow(1.24, 2.06, 1.24, 2.62); arrow(1.24, 3.29, 1.24, 3.95)
    arrow(1.70, 4.24, 2.20, 4.24)

    # TF and gene representation updates: identical bottom-to-top state grammar.
    for x, sub, state_shape, cross, attn_detail, theme, theme_fill in [
        (4.20, "TF", r"$B\times N_{TF}\times h$", "R → TF", "Q: TF; K,V: R", tf_c, tf_fill),
        (8.15, "G", r"$B\times N_G\times h$", "TF → Gene", "Q: Gene; K,V: TF", gene_c, gene_fill),
    ]:
        h0 = rf"$H_{{{sub}}}^{{(0)}}$"
        h1 = rf"$H_{{{sub}}}^{{(1)}}$"
        box(x + 1.32, 1.42, 1.05, 0.58, h0, kind="state", fontsize=7.5,
            detail=state_shape, edgecolor=theme, facecolor=theme_fill, textcolor=theme)
        box(x + 0.78, 2.35, 2.12, 0.92, f"Cross-attention\n{cross}", fontsize=6.4,
            detail=attn_detail)
        box(x + 1.04, 3.66, 1.60, 0.62, "Residual + LN", fontsize=6.2)
        box(x + 0.78, 4.66, 2.12, 0.92, "Self-attention\n+ FFN", fontsize=6.3,
            detail="blocks ×2")
        box(x + 1.32, 6.00, 1.05, 0.58, h1, kind="state", fontsize=7.5,
            detail=state_shape, edgecolor=theme, facecolor=theme_fill, textcolor=theme)
        arrow(x + 1.845, 2.00, x + 1.845, 2.35)
        arrow(x + 1.845, 3.27, x + 1.845, 3.66)
        arrow(x + 1.845, 4.28, x + 1.845, 4.66)
        arrow(x + 1.845, 5.58, x + 1.845, 6.00)

    # Left-to-right inter-stage signal flow.
    poly_arrow([(3.67, 5.79), (4.65, 5.79), (4.65, 2.81), (4.98, 2.81)], zorder=1)
    poly_arrow([(6.57, 6.29), (8.60, 6.29), (8.60, 2.81), (8.93, 2.81)], zorder=1)

    # Frameless outputs. Expression branches from the refined gene state, whereas
    # the GRN readout is returned directly by TF-to-gene cross-attention.
    box(12.62, 5.69, 2.62, 1.20, "Next-state prediction", kind="operation",
        fontsize=7.2, weight="bold",
        detail="LN + Linear + GELU + Dropout\n+ gene-specific weight/bias")
    box(12.62, 2.21, 2.62, 1.20, "GRN inference", kind="operation",
        fontsize=7.2, weight="bold", detail=r"TF→gene attention   $A_{TF\to Gene}$")
    arrow(10.52, 6.29, 12.62, 6.29)
    arrow(11.05, 2.81, 12.62, 2.81)

    # Shared initialization source; details are deliberately subordinate.
    box(5.20, 0.01, 5.60, 0.82,
        "Frozen per-cell scFoundation\ngene embeddings\n" + r"$E:\ B\times N_G\times 512$",
        kind="detail", fontsize=5.8, weight="bold")
    ax.text(6.00, 0.91, "Input projection: Linear + LN + GELU + Dropout",
            ha="center", va="center", fontsize=5.8, family="sans-serif", color="black")
    ax.plot([3.15, 9.995], [1.08, 1.08], color="black", linewidth=arrow_lw, zorder=1)
    arrow(8.00, 0.76, 8.00, 1.08, shrink_b=0)
    arrow(3.15, 1.08, 3.15, 1.48, shrink_a=0)
    arrow(6.045, 1.08, 6.045, 1.42, shrink_a=0)
    arrow(9.995, 1.08, 9.995, 1.42, shrink_a=0)
    mpl.rcParams["mathtext.fontset"] = "dejavusans"
    save(fig, MAIN, stem)
    note_for("Figure1", "Figure 1. LITD model architecture", ["plan"],
             ["Left-to-right Receptor → TF → Gene → Output computation graph",
              "Bottom-to-top hidden-state updates H^(0) → H^(1)",
              "LR-prior–guided receptor activation with known M_LR mask",
              "Frameless next-state prediction and GRN interpretation outputs",
              "GRN readout branches directly from TF-to-gene cross-attention",
              "Selective tensor-shape labels and explicit cross-attention Q/K/V roles"],
             "This is a model schematic, not a quantitative result. Vector SVG/PDF/AI outputs are generated directly from the reproducible figure builder; the computation order and tensor roles follow the fixed c38 implementation.")


def load_benchmark():
    d = pd.read_csv(FILES["benchmark"])
    d["setting_short"] = "TFs+" + d.target_genes.astype(str) + "\n" + d.network.astype(str)
    order = [f"TFs+{n}\n{k}" for n in [500,1000] for k in ["STRING","NonSpecific","ChIP"]]
    return d, order


def make_figure2():
    d, order = load_benchmark(); methods = list(dict.fromkeys(d.method))
    fig = plt.figure(figsize=(7.15, 7.80))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, .72], hspace=.55, wspace=.43)
    for j, dataset in enumerate(["hESC", "hHep"]):
        ax = fig.add_subplot(gs[j,:]); p=d[d.dataset==dataset].pivot(index="method",columns="setting_short",values="EPR").reindex(index=methods,columns=order)
        sns.heatmap(p, ax=ax, cmap="Blues", annot=True, fmt=".2f", linewidths=.5, linecolor="white",
                    cbar=True, cbar_kws={"label":"EPR", "shrink":.58, "aspect":18}, annot_kws={"fontsize":6.6})
        panel(ax, "AB"[j], dataset)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(order, rotation=0, fontsize=6.4)
        for t in ax.get_yticklabels():
            if t.get_text()=="LITD": t.set_color(COLORS["litd"]); t.set_weight("bold")
    ax=fig.add_subplot(gs[2,0]); ranks=d.groupby("method").agg(mean_rank=("rank_by_epr","mean"),first=("rank_by_epr",lambda x:(x==1).sum())).sort_values("mean_rank")
    y=np.arange(len(ranks)); ax.barh(y,ranks.mean_rank,color=[COLORS["litd"] if x=="LITD" else "#aeb8c2" for x in ranks.index])
    ax.set_yticks(y,ranks.index); ax.invert_yaxis(); ax.set_xlabel("Mean rank across 12 settings"); panel(ax,"C","Across-setting mean rank"); clean(ax)
    ax.set_xlim(0, max(6.7, ranks.mean_rank.max() * 1.18))
    ax=fig.add_subplot(gs[2,1]); mean_epr=d.groupby("method",as_index=True).EPR.mean().sort_values(ascending=True)
    y=np.arange(len(mean_epr));bars=ax.barh(y,mean_epr.values,color=[COLORS["litd"] if m=="LITD" else "#aeb8c2" for m in mean_epr.index]);ax.set_yticks(y,mean_epr.index);ax.set_xlabel("Mean EPR across 12 settings");panel(ax,"D","Across-setting mean EPR");clean(ax)
    ax.set_xlim(0,mean_epr.max()*1.18)
    fig.subplots_adjust(left=.15, right=.95, top=.97, bottom=.07)
    save(fig,MAIN,"Figure2_human_benchmark_GRN_performance")
    note_for("Figure2","Figure 2. Human benchmark GRN inference performance",["benchmark"],
             ["A: hESC EPR matrix","B: hHep EPR matrix","C: mean rank across 12 settings","D: mean EPR across all 12 settings"],
             "Published comparator EPR values and this-study LITD EPR values are displayed as recorded in the source table; missing Eprec/Erec values are not imputed.")


ABL_LABELS={"shuffled_temporal":"Shuffled temporal order","same_time_reconstruction":"Static reconstruction","no_ligand_input":"No ligand context","no_lr_prior":"No LR mask","no_scfoundation":"No scFoundation"}
ABLATION_INPUTS=[key for key in FILES if key.startswith("ablation_")]
def load_formal_ablation():
    frames=[]
    for key in ABLATION_INPUTS:
        d=pd.read_csv(FILES[key])
        d["source_key"]=key
        frames.append(d)
    return pd.concat(frames,ignore_index=True)

def ablation_summary():
    d=load_formal_ablation(); d=d[d.EPR.notna()]
    g=d[["ablation_name","dataset","target_genes","network","EPR"]].copy()
    full=g[g.ablation_name=="full"][["dataset","target_genes","network","EPR"]].rename(columns={"EPR":"full_EPR"})
    g=g.merge(full,on=["dataset","target_genes","network"]); g["delta_EPR"]=g.EPR-g.full_EPR
    g["setting"]=g.dataset.map({"hesc":"hESC","hhep":"hHep"})+"\nTFs+"+g.target_genes.astype(str)+" "+g.network
    return g[g.ablation_name.isin(ABL_LABELS)].copy()


def load_weak_static():
    d = pd.read_csv(FILES["weak_signal_metrics"])
    return d[(d.stratifier == "static_correlation") & d.stratum.isin(["weak", "medium", "strong"])].copy()


def faceted_epr_row(fig, spec, data, row_order, dataset, panel_label, lead_label, vmax):
    """Three aligned heatmaps make correlation stratum the primary reading axis."""
    inner = spec.subgridspec(1, 3, wspace=.07)
    strata = ["weak", "medium", "strong"]
    strata_titles = ["Weak |r| < 0.1", "Medium 0.1 ≤ |r| < 0.3", "Strong |r| ≥ 0.3"]
    setting_order = [f"TFs+{n}\n{net}" for n in [500, 1000] for net in ["STRING", "NonSpecific", "ChIP"]]
    axes = []
    for j, (stratum, stratum_title) in enumerate(zip(strata, strata_titles)):
        ax = fig.add_subplot(inner[0, j]); axes.append(ax)
        p = data[(data.dataset == dataset) & (data.stratum == stratum)].pivot(
            index="method_label", columns="setting_short", values="within_stratum_EPR"
        ).reindex(index=row_order, columns=setting_order)
        sns.heatmap(p, ax=ax, cmap="Blues", vmin=0, vmax=vmax, annot=True, fmt=".2f",
                    linewidths=.45, linecolor="white", cbar=j == 2,
                    cbar_kws={"label": "EPR", "shrink": .58, "aspect":18}, annot_kws={"fontsize": 6.0})
        ax.set_title((f"{lead_label} — " if j == 0 else "") + stratum_title,
                     loc="left", pad=6, fontsize=7.8, weight="normal")
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xticklabels(setting_order, rotation=0, fontsize=6.1)
        if j:
            ax.set_yticklabels([]); ax.tick_params(axis="y", left=False)
        else:
            ax.text(-.16, 1.08, panel_label, transform=ax.transAxes, fontsize=11, weight="bold", va="top")
            for tick in ax.get_yticklabels():
                if tick.get_text() in {"LITD", "Full LITD"}:
                    tick.set_color(COLORS["litd"]); tick.set_weight("bold")
    return axes


def make_figure3():
    raw = load_formal_ablation()
    method_order = ["full"] + list(ABL_LABELS)
    method_labels = {"full": "Full LITD", **ABL_LABELS}
    setting_order = [f"TFs+{n}\n{net}" for n in [500, 1000] for net in ["STRING", "NonSpecific", "ChIP"]]
    raw["setting_short"] = "TFs+" + raw.target_genes.astype(str) + "\n" + raw.network.astype(str)
    raw["method_label"] = raw.ablation_name.map(method_labels)
    raw.to_csv(DATA / "Figure3_standard_benchmark_ablation_all_settings.csv", index=False)

    fig = plt.figure(figsize=(7.15, 7.90))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, .76], hspace=.57, wspace=.44)
    for ax, dataset, label, title in zip(
        [fig.add_subplot(gs[0,:]), fig.add_subplot(gs[1,:])],
        ["hesc", "hhep"], ["A", "B"], ["hESC", "hHep"]):
        p = raw[raw.dataset.eq(dataset)].pivot(index="method_label", columns="setting_short", values="EPR")
        p = p.reindex(index=[method_labels[x] for x in method_order], columns=setting_order)
        sns.heatmap(p, ax=ax, cmap="Blues", annot=True, fmt=".2f", linewidths=.55, linecolor="white",
                    cbar_kws={"label":"EPR", "shrink":.58, "aspect":18}, annot_kws={"fontsize":6.5})
        panel(ax, label, title)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xticklabels(setting_order, rotation=0, fontsize=6.5)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        for tick in ax.get_yticklabels():
            if tick.get_text() == "Full LITD": tick.set_color(COLORS["litd"]); tick.set_weight("bold")

    rank_keys = ["dataset", "target_genes", "network"]
    raw["rank"] = raw.groupby(rank_keys).EPR.rank(method="min", ascending=False)
    rank = raw.groupby("method_label")["rank"].mean().sort_values(ascending=False)
    mean = raw.groupby("method_label").EPR.mean().sort_values()
    rank.to_csv(DATA / "Figure3_standard_benchmark_ablation_mean_rank.csv")
    mean.to_csv(DATA / "Figure3_standard_benchmark_ablation_mean_EPR.csv")
    for ax, series, label, title, xlabel, xmax in [
        (fig.add_subplot(gs[2,0]), rank, "C", "Across-setting mean rank", "Mean rank across 12 settings", max(6.8, float(rank.max())*1.18)),
        (fig.add_subplot(gs[2,1]), mean, "D", "Across-setting mean EPR", "Mean EPR across 12 settings", float(mean.max())*1.20)]:
        y = np.arange(len(series)); bars = ax.barh(y, series.values, color=[COLORS["litd"] if m == "Full LITD" else "#aeb8c2" for m in series.index])
        ax.set_yticks(y, series.index); ax.set_xlabel(xlabel); panel(ax, label, title); clean(ax); ax.set_xlim(0, xmax)
    fig.subplots_adjust(left=.20, right=.96, top=.97, bottom=.07)
    save(fig, MAIN, "Figure3_standard_benchmark_ablation")
    note_for("Figure3", "Figure 3. Standard benchmark ablation", ABLATION_INPUTS,
             ["A: hESC EPR matrix", "B: hHep EPR matrix", "C: mean rank across 12 settings", "D: mean EPR across 12 settings"],
             "All five ablations use fixed-c38 checkpoints and the canonical temporal-pair files. Source zeros are retained as measured collapse-like outcomes.")


def correlation_summary(q, stratum):
    keys = ["dataset", "target_gene_count", "network", "stratum"]
    part = q[q.stratum.eq(stratum)].copy()
    part["rank"] = part.groupby(keys).within_stratum_EPR.rank(method="min", ascending=False)
    return part.groupby("method_label").agg(
        mean_EPR=("within_stratum_EPR", "mean"),
        mean_rank=("rank", "mean"),
    )


def correlation_epr_panel(ax, q, stratum, label, title, litd_label="LITD"):
    summary = correlation_summary(q, stratum).sort_values("mean_EPR")
    y = np.arange(len(summary))
    bars = ax.barh(y, summary.mean_EPR,
                   color=[COLORS["litd"] if m == litd_label else "#aeb8c2" for m in summary.index])
    ax.set_yticks(y, summary.index); ax.set_xlabel("Mean EPR across 12 settings")
    panel(ax, label, title); clean(ax); xmax = float(summary.mean_EPR.max()) * 1.20; ax.set_xlim(0, xmax)
    return summary


def make_figure4():
    d = load_weak_static()
    method_order = ["LITD_c38", "PPCOR", "SCODE", "GENIE3", "GRNBoost2", "PIDC", "SINCERITIES"]
    q = d[d.method.isin(method_order)].copy(); q["method_label"] = q.method.replace({"LITD_c38":"LITD"})
    fig, axs = plt.subplots(1, 3, figsize=(7.15, 3.05), gridspec_kw={"wspace":.48})
    specs = [("weak", "A", "Weak |r| < 0.1"), ("medium", "B", "Medium 0.1 ≤ |r| < 0.3"), ("strong", "C", "Strong |r| ≥ 0.3")]
    exports=[]
    for ax, (stratum, label, title) in zip(axs, specs):
        summary=correlation_epr_panel(ax, q, stratum, label, title)
        out=summary.reset_index(); out["stratum"]=stratum; exports.append(out)
    pd.concat(exports, ignore_index=True).to_csv(DATA / "Figure4_correlation_stratified_mean_EPR_summary.csv", index=False)
    fig.subplots_adjust(left=.11, right=.985, top=.84, bottom=.22)
    save(fig, MAIN, "Figure4_correlation_stratified_GRN_recovery")
    note_for("Figure4", "Figure 4. Correlation-stratified GRN recovery", ["weak_signal_metrics", "weak_signal_validation", "weak_signal_config", "weak_signal_readme"],
             ["A: Weak-stratum mean EPR", "B: Medium-stratum mean EPR", "C: Strong-stratum mean EPR"],
             "Pearson and Spearman are excluded. Thresholds are fixed globally at 0.1 and 0.3; each panel reports mean EPR across the same 12 dataset, target-space and reference-network settings.")


def make_figure5():
    d = load_weak_static()
    method_order = ["LITD_full_ablation", "LITD_shuffled_temporal", "LITD_same_time", "LITD_no_ligand", "LITD_no_lr_mask", "LITD_no_scfoundation"]
    method_labels = {"LITD_full_ablation": "Full LITD", "LITD_shuffled_temporal": "Shuffled temporal order", "LITD_same_time": "Static reconstruction", "LITD_no_ligand": "No ligand context", "LITD_no_lr_mask": "No LR mask", "LITD_no_scfoundation": "No scFoundation"}
    q = d[d.method.isin(method_order)].copy(); q["method_label"] = q.method.map(method_labels)
    q["setting_short"] = "TFs+" + q.target_gene_count.astype(str) + "\n" + q.network.astype(str)
    weak = q[q.stratum.eq("weak")].copy()
    weak.to_csv(DATA / "Figure5_weak_ablation_setting_level.csv", index=False)
    setting_order = [f"TFs+{n}\n{net}" for n in [500, 1000] for net in ["STRING", "NonSpecific", "ChIP"]]
    row_order = [method_labels[x] for x in method_order]
    vmax = float(weak.within_stratum_EPR.max())

    # Show the two setting-level heat maps first and place their across-setting
    # summary beneath them; panel letters follow the top-to-bottom reading order.
    fig = plt.figure(figsize=(7.15, 8.10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, .82], hspace=.65)

    for idx, (dataset, label, title) in enumerate([
        ("hesc", "A", "hESC — Weak |r| < 0.1"),
        ("hhep", "B", "hHep — Weak |r| < 0.1"),
    ]):
        ax = fig.add_subplot(gs[idx, 0])
        p = weak[weak.dataset.eq(dataset)].pivot(
            index="method_label", columns="setting_short", values="within_stratum_EPR"
        ).reindex(index=row_order, columns=setting_order)
        sns.heatmap(p, ax=ax, cmap="Blues", vmin=0, vmax=vmax, annot=True, fmt=".2f",
                    linewidths=.5, linecolor="white", cbar=True,
                    cbar_kws={"label": "EPR", "shrink": .52, "aspect": 16},
                    annot_kws={"fontsize": 6.6})
        panel(ax, label, title); ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xticklabels(setting_order, rotation=0, fontsize=6.4)
        for tick in ax.get_yticklabels():
            if tick.get_text() == "Full LITD":
                tick.set_color(COLORS["litd"]); tick.set_weight("bold")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    ax = fig.add_subplot(gs[2, 0])
    correlation_epr_panel(ax, weak, "weak", "C", "Weak |r| < 0.1", litd_label="Full LITD")
    fig.subplots_adjust(left=.23, right=.96, top=.97, bottom=.07)
    save(fig,MAIN,"Figure5_correlation_stratified_ablation")
    note_for("Figure5","Figure 5. Correlation-stratified ablation",["weak_signal_metrics", "weak_signal_validation", "weak_signal_config", "weak_signal_readme"],
             ["A: hESC Weak-stratum setting-level EPR", "B: hHep Weak-stratum setting-level EPR", "C: Weak-stratum ablation mean EPR across 12 settings"],
             "Only the Weak |r| < 0.1 stratum is shown. Panel C summarizes the same 12 matched settings detailed by dataset in panels A and B. Former panels B, D and E and the former Supplementary Figure S3 ablation-detail figure are removed.")


def make_s1_expression():
    d=pd.read_csv(FILES["expression"])
    methods=["c38","mlp_c38parammatch","prescient","scfoundation_mlp","persistence","stage_mean"]
    labels={"c38":"LITD","mlp_c38parammatch":"MLP","prescient":"PRESCIENT","scfoundation_mlp":"scFoundation-MLP","persistence":"Persistence","stage_mean":"Stage mean"}
    q=d[d.method.isin(methods)].copy();q["dataset"]=q.dataset.map({"hesc":"hESC","hhep":"hHep"});q["display_method"]=q.method.map(labels)
    q.to_csv(DATA/"FigureS1_expression_prediction_paper_subset.csv",index=False)
    avg=q.groupby(["method","display_method"],as_index=False)[["pearson","genewise_pearson"]].mean();avg.to_csv(DATA/"FigureS1_expression_prediction_hesc_hhep_mean.csv",index=False)
    fig,axs=plt.subplots(1,2,figsize=(7.15,3.35),gridspec_kw={"wspace":.58})
    metrics=[("pearson","Overall Pearson"),("genewise_pearson","Gene-wise Pearson")]
    for ax,(metric,title),lab in zip(axs,metrics,"AB"):
        ordered = avg.sort_values(metric, ascending=False)
        y = np.arange(len(ordered))
        bars = ax.barh(y, ordered[metric], color=[COLORS["litd"] if m == "LITD" else "#aeb8c2" for m in ordered.display_method])
        ax.set_yticks(y, ordered.display_method); ax.invert_yaxis()
        panel(ax,lab);clean(ax);ax.set_ylabel("")
        ax.set_xlabel(title)
        for tick in ax.get_yticklabels():
            if tick.get_text() == "LITD": tick.set_color(COLORS["litd"]); tick.set_weight("bold")
        if metric == "pearson": ax.set_xlim(0,1.02)
        elif metric == "genewise_pearson": ax.set_xlim(0, float(avg[metric].max()) * 1.28)
    fig.subplots_adjust(left=.15, right=.97, top=.93, bottom=.16)
    save(fig,SUPP,"Supplementary_Figure_S1_next_state_expression_prediction")
    note_for("Supplementary_Figure_S1","Supplementary Figure S1. Next-state expression prediction",["expression"],
             ["A: mean overall Pearson across hESC and hHep","B: mean gene-wise Pearson across hESC and hHep"],
             "Within each panel, methods are ordered from the largest value at the top to the smallest at the bottom. LITD uses the same blue highlight and gray-comparator palette as Figure 2 panel C. Each bar is the unweighted arithmetic mean of the hESC and hHep values for that method. The parameter-matched comparator is labelled MLP; MSE is omitted.")


def path_schematic(ax, genes, color, title, label):
    ax.axis("off");panel(ax,label,title)
    n=len(genes)
    for i,g in enumerate(genes):
        box(ax,(.02+i*(.96/n),.38),(.17,.25),g,"#fff4e8" if color==COLORS["entry"] else "#f3eff9",ec=color,fs=8,weight="bold")
        if i<n-1: arrow(ax,(.19+i*(.96/n),.505),(.02+(i+1)*(.96/n),.505),color,lw=1.8)


def make_figure6():
    top=pd.read_csv(FILES["weighted_path_top1000"]).head(5).copy()
    top["path_label"]=top[["ligand","receptor","tf","target_gene"]].astype(str).agg("→".join,axis=1)
    top.to_csv(DATA/"Figure6_global_weighted_PathScore_top5.csv",index=False)
    win=pd.read_csv(FILES["weighted_path_winner_windows"])
    zero=pd.read_csv(FILES["mdk_zero_ligand"])
    mismatch_all=pd.read_csv(FILES["mdk_stage_mismatch"])
    window_order=["day0_to_day1","day1_to_day2","day2_to_day3"]
    mismatch_rows=[]
    for window in window_order:
        subset=mismatch_all[mismatch_all.real_window==window]
        if len(subset)!=2:
            raise ValueError(f"Expected two alternative-stage mismatches for {window}, found {len(subset)}")
        real_values=subset.real_internal_cascade.to_numpy(dtype=float)
        if not np.allclose(real_values,real_values[0]):
            raise ValueError(f"Inconsistent real InternalCascade values for {window}")
        mismatch_values=subset.mismatch_internal_cascade.to_numpy(dtype=float)
        mismatch_rows.append({
            "window":window,
            "real_internal_cascade":float(real_values[0]),
            "mean_mismatch_internal_cascade":float(mismatch_values.mean()),
            "min_mismatch_internal_cascade":float(mismatch_values.min()),
            "max_mismatch_internal_cascade":float(mismatch_values.max()),
            "n_mismatch_environments":int(len(mismatch_values)),
        })
    mismatch_summary=pd.DataFrame(mismatch_rows)
    mismatch_summary.to_csv(DATA/"Figure6_three_window_stage_mismatch_summary.csv",index=False)
    # The path ranking uses the full top row; the three compact internal
    # validation panels share the second row.
    fig=plt.figure(figsize=(7.15,5.60));gs=fig.add_gridspec(2,3,height_ratios=[1.0,1.05],hspace=.58,wspace=.62)
    ax=fig.add_subplot(gs[0,:]);ranked=top.sort_values("global_weighted_pathscore");y=np.arange(len(ranked));vals=ranked.global_weighted_pathscore.values
    xmin = .153595
    bar_colors=[COLORS["entry"] if r==1 else "#7892aa" for r in ranked["rank"]]
    bars=ax.barh(y,vals-xmin,left=xmin,color=bar_colors,height=.52,edgecolor="white")
    ax.set_yticks(y,ranked.path_label);ax.set_xlabel("GlobalPathScore");panel(ax,"A","Top five candidate paths by GlobalPathScore");clean(ax)
    span=max(.00005,float(vals.max()-vals.min()));ax.set_xlim(xmin,.153715)
    ax.set_xticks(np.arange(.15360,.153701,.00002));ax.ticklabel_format(axis="x",style="plain",useOffset=False)
    x=np.array([0.0,.58,1.16]);window_labels=["day0→1","day1→2","day2→3"]
    ax=fig.add_subplot(gs[1,0]);target_change=(zero.zero_predicted_LIMD1/zero.real_predicted_LIMD1-1)*100
    bars=ax.bar(x,target_change,color=["#9cb4c8","#658eae",COLORS["litd"]],width=.34,edgecolor="white")
    ax.axhline(0,color="#59636e",lw=1.0);ax.set_xticks(x,window_labels);ax.set_xlim(-.30,1.46)
    ax.set_ylabel("Predicted LIMD1 change vs real (%)");panel(ax,"B");clean(ax)
    pad=max(.15,float(np.ptp(target_change))*.08)
    ax.set_ylim(float(target_change.min())-pad*2.2,float(target_change.max())+pad*2.2)
    ax=fig.add_subplot(gs[1,1]);bar_width=.20
    real_cascade=mismatch_summary.real_internal_cascade.to_numpy()
    mean_mismatch_cascade=mismatch_summary.mean_mismatch_internal_cascade.to_numpy()
    real_bars=ax.bar(x-bar_width/2,real_cascade,width=bar_width,color=COLORS["real"],edgecolor="white",label="Matched")
    mismatch_bars=ax.bar(x+bar_width/2,mean_mismatch_cascade,width=bar_width,color=COLORS["mismatch"],edgecolor="white",label="Mean mismatch")
    ax.set_xticks(x,window_labels);ax.set_xlim(-.30,1.46);ax.set_ylabel("InternalCascade");panel(ax,"C");clean(ax)
    ymax=float(max(real_cascade.max(),mean_mismatch_cascade.max()))*1.18
    ax.set_ylim(0,ymax);ax.legend(frameon=False,fontsize=6.4,loc="upper center",
                                 bbox_to_anchor=(.64,1.04),ncol=2,handlelength=1.5,
                                 columnspacing=.8,borderaxespad=0)
    ax.ticklabel_format(axis="y",style="plain",useOffset=False)
    ax=fig.add_subplot(gs[1,2]);scores=win["path_score_window"].to_numpy();bars=ax.bar(x,scores,color=["#9cb4c8","#658eae",COLORS["litd"]],width=.34,edgecolor="white");ax.set_xticks(x,window_labels);ax.set_xlim(-.30,1.46);ax.set_ylabel("PathScore");panel(ax,"D");clean(ax)
    fig.subplots_adjust(left=.25, right=.98, top=.95, bottom=.10)
    save(fig,MAIN,"Figure6_MDK_ITGB1_PTMA_LIMD1_global_PathScore")
    note_for("Figure6","Figure 6. MDK-ITGB1-PTMA-LIMD1 global weighted PathScore and internal validation",["weighted_path_top1000","weighted_path_winner_windows","weighted_path_metadata","weighted_path_validation","mdk_stage_specificity","mdk_zero_ligand","mdk_stage_mismatch","mdk_interpretation_boundary","mdk_internal_validation_report"],
             ["A: exact global Top 5 horizontal bars","B: three-window zero-MDK predicted-LIMD1 changes","C: three-window matched-versus-mean-mismatch InternalCascade grouped bars","D: three-window stage-specific PathScore"],
             "Panel A uses Figure 4-style horizontal bars with a zoomed x-axis showing the original GlobalPathScore values to resolve the tightly clustered Top-5 ranking. In panel C, each mean-mismatch bar is the arithmetic mean of the two alternative-stage ligand environments for that transition window; all six underlying mismatch comparisons remain available in the stage-mismatch matrix. Panel C uses the original InternalCascade values and a zero baseline to avoid visually exaggerating the modest matched-versus-mismatch differences. Panels B–D use narrow bars because they contain only a few observations. These are model-internal conditional-dependence results, not causal validation.")


def make_s3_relay():
    ranked_relay=pd.read_csv(FILES["relay_ranked_top1000"]).head(5).copy()
    ranked_relay["relay_label"]=(ranked_relay.ligand_0.astype(str)+"→"+
                                 ranked_relay.relay_ligand_1.astype(str)+"→"+
                                 ranked_relay.relay_ligand_2.astype(str)+"→"+
                                 ranked_relay.terminal_gene.astype(str))
    ranked_relay.to_csv(DATA / "FigureS3_distinct_TF_relay_top5.csv", index=False)
    stage=pd.read_csv(FILES["relay_stage_specificity"]).sort_values("segment_id")
    cf=pd.read_csv(FILES["relay_counterfactuals"]).sort_values("segment_id")
    cols=["#9cb4c8","#658eae",COLORS["litd"]]
    # Match Figure 6's hierarchy: a full-width ranking above three compact
    # segment-level validation panels.
    fig=plt.figure(figsize=(7.15,5.60));gs=fig.add_gridspec(2,3,height_ratios=[1.0,1.05],hspace=.58,wspace=.62)

    ax=fig.add_subplot(gs[0,:]);panel(ax,"A","Top five candidate relay chains by Relay PathScore")
    top=ranked_relay.sort_values("chain_score_geometric_mean")
    values=top.chain_score_geometric_mean.to_numpy(float)
    xmin=.15082
    bar_colors=[COLORS["entry"] if int(rank)==1 else "#8797aa" for rank in top["rank"]]
    y=np.arange(len(top)); bars=ax.barh(y,values-xmin,left=xmin,color=bar_colors,height=.50,edgecolor="white")
    ax.set_yticks(y,top.relay_label);ax.set_xlabel("Relay PathScore");clean(ax)
    span=max(.00008,float(values.max()-values.min()));ax.set_xlim(xmin,.15235);ax.ticklabel_format(axis="x",style="plain",useOffset=False)

    x=np.array([0.0,.58,1.16])
    ax=fig.add_subplot(gs[1,0]);panel(ax,"B")
    target_change=cf.zero_fractional_change_predicted_target_mean.to_numpy(float)*100
    bars=ax.bar(x,target_change,color=cols,width=.34,edgecolor="white")
    ax.axhline(0,color="#59636e",lw=.9)
    segment_labels=["MDK→DKK1","DKK1→FGF17","FGF17→MUL1"]
    ax.set_xticks(x,segment_labels,fontsize=6.0,rotation=18,ha="right");ax.set_xlim(-.30,1.46)
    ax.set_ylabel("Predicted target change vs real (%)");clean(ax)
    pad=max(1.1,(target_change.max()-target_change.min())*.07)
    ax.set_ylim(target_change.min()-2.8*pad,target_change.max()+2.8*pad)

    ax=fig.add_subplot(gs[1,1]);panel(ax,"C")
    real=cf.real_internal_cascade.to_numpy(float);mismatch=cf.mismatch_internal_cascade.to_numpy(float)
    width=.20;segx=x
    positive=np.concatenate([real,mismatch]);positive=positive[positive>0]
    plot_floor=float(positive.min())/8.0
    real_bars=ax.bar(segx-width/2,real-plot_floor,width,bottom=plot_floor,
                     color=COLORS["real"],label="Matched",edgecolor="white")
    mismatch_bars=ax.bar(segx+width/2,mismatch-plot_floor,width,bottom=plot_floor,
                         color=COLORS["mismatch"],label="Mean mismatch",edgecolor="white")
    ax.set_xticks(segx,segment_labels,fontsize=6.0,rotation=18,ha="right");ax.set_xlim(-.30,1.46)
    ax.set_ylabel("InternalCascade");ax.legend(frameon=False,fontsize=6.4,
                                               loc="upper center",bbox_to_anchor=(.64,1.04),
                                               ncol=2,handlelength=1.5,columnspacing=.8,
                                               borderaxespad=0);clean(ax)
    ax.set_yscale("log");ax.set_ylim(plot_floor,positive.max()*4.2)

    ax=fig.add_subplot(gs[1,2]);panel(ax,"D")
    ratios=stage.assigned_over_other_max_legacy_path_score.to_numpy(float)
    log_ratios=np.log10(ratios)
    bars=ax.bar(x,log_ratios,color=cols,width=.34,edgecolor="white")
    ax.set_xticks(x,segment_labels,fontsize=6.0,rotation=18,ha="right");ax.set_xlim(-.30,1.46)
    ax.set_ylabel("log10(PathScore enrichment)")
    ax.set_ylim(0,log_ratios.max()+.65);clean(ax)

    fig.subplots_adjust(left=.25, right=.98, top=.95, bottom=.11)
    save(fig,SUPP,"Supplementary_Figure_S3_candidate_relay_TF_handoff")
    note_for("Supplementary_Figure_S3","Supplementary Figure S3. Candidate relay and TF handoff",["relay_ranked_top1000","relay_manifest","relay_stage_specificity","relay_counterfactuals","relay_interpretation_boundary"],
             ["A: Top-5 ranked distinct-TF relay chains","B: zero-ligand predicted-target percent change","C: matched versus mean-of-two-stage-mismatches InternalCascade grouped bars","D: assigned/max-other PathScore enrichment"],
             "Panel A condenses each three-segment relay into a ligand-to-ligand-to-ligand-to-terminal label, for example MDK→DKK1→FGF17→MUL1. The relay is hypothesis-generating. Stage specificity compares real windows; zero-ligand and stage mismatch are model-internal input counterfactuals, not causal interventions. Panel C uses a logarithmic model-score axis so every real and mismatch bar remains visible. Each mismatch bar is the arithmetic mean of separately evaluated InternalCascade scores under the two alternative source-stage ligand environments, with source cells fixed (day 0: days 1 and 2; day 1: days 0 and 2; day 2: days 0 and 1). The observed-to-mean-mismatch ratios are 2.47, 1.06, and 127.70; the second segment is only 5.81% above the mean and below its day-2 mismatch score.")


def make_s2():
    d=load_weak_static();method_order=["LITD_c38","PPCOR","SCODE","GENIE3","GRNBoost2","PIDC","SINCERITIES"]
    labels={"LITD_c38":"LITD"};q=d[d.method.isin(method_order)].copy();q["method_label"]=q.method.map(lambda x:labels.get(x,x))
    q["setting_short"]="TFs+"+q.target_gene_count.astype(str)+"\n"+q.network.astype(str)
    fig=plt.figure(figsize=(7.15,7.20));gs=fig.add_gridspec(3,2,hspace=.30,wspace=.42)
    row_order=[labels.get(x,x) for x in method_order];vmax=float(np.nanmax(q.within_stratum_EPR))
    setting_order=[f"TFs+{n}\n{net}" for n in [500,1000] for net in ["STRING","NonSpecific","ChIP"]]
    strata=[("weak","Weak |r| < 0.1"),("medium","Medium 0.1 ≤ |r| < 0.3"),("strong","Strong |r| ≥ 0.3")]
    datasets=[("hesc","hESC"),("hhep","hHep")]
    # Each dataset has one short vertical scale beside the upper-right edge of
    # its first-row matrix, keeping the scale close to panel A or B without
    # interrupting the two lower rows.
    cbar_axes=[fig.add_axes([.460,.765,.010,.145]),
               fig.add_axes([.908,.765,.010,.145])]
    for row,(stratum,stratum_title) in enumerate(strata):
        for col,(dataset,dataset_title) in enumerate(datasets):
            ax=fig.add_subplot(gs[row,col])
            p=q[(q.dataset==dataset)&(q.stratum==stratum)].pivot(
                index="method_label",columns="setting_short",values="within_stratum_EPR"
            ).reindex(index=row_order,columns=setting_order)
            show_cbar=row==0
            sns.heatmap(p,ax=ax,cmap="Blues",vmin=0,vmax=vmax,annot=True,fmt=".2f",
                        linewidths=.45,linecolor="white",cbar=show_cbar,
                        cbar_ax=cbar_axes[col] if show_cbar else None,
                        cbar_kws={"label":"EPR"},
                        annot_kws={"fontsize":5.8})
            panel_label="A" if row==0 and col==0 else ("B" if row==0 and col==1 else None)
            if panel_label:
                ax.text(-.10,1.07,panel_label,transform=ax.transAxes,fontsize=9.5,weight="bold",va="top")
            ax.set_title(f"{dataset_title} — {stratum_title}",loc="left",pad=6,
                         fontsize=7.8,weight="normal")
            ax.set_xlabel("");ax.set_ylabel("")
            if col:
                ax.set_yticklabels([]);ax.tick_params(axis="y",left=False)
            else:
                ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
                for tick in ax.get_yticklabels():
                    if tick.get_text()=="LITD":tick.set_color(COLORS["litd"]);tick.set_weight("bold")
            if row<2:
                ax.set_xticklabels([]);ax.tick_params(axis="x",bottom=False)
            else:
                ax.set_xticklabels(setting_order,rotation=28,ha="right",fontsize=5.8)
    for cax in cbar_axes:
        cax.tick_params(axis="y", labelsize=5.5, length=2, pad=1)
        cax.yaxis.label.set_size(6)
        cax.yaxis.labelpad = 2
    fig.subplots_adjust(left=.14,right=.90,top=.95,bottom=.13)
    save(fig,SUPP,"Supplementary_Figure_S2_dataset_level_correlation_stratified_GRN_recovery")
    note_for("Supplementary_Figure_S2","Supplementary Figure S2. Dataset-level correlation-stratified GRN recovery",["weak_signal_metrics","weak_signal_validation","weak_signal_config","weak_signal_readme"],["A: hESC faceted Weak/Medium/Strong EPR matrices","B: hHep faceted Weak/Medium/Strong EPR matrices"],"These matrices provide dataset-level detail for main Figure 4. Pearson and Spearman are excluded; no internal row or column divider is emphasized.")


def make_s4():
    d = pd.read_csv(FILES["scale"])
    sizes = pd.read_csv(FILES["scale_training_sizes"])
    order = sizes.sort_values("training_transition_pairs").config_label.tolist()
    meta = sizes.set_index("config_label")
    short = {
        "c38_hhep_40ep": "hHep",
        "c38_hesc_40ep": "hESC",
        "g30": "Endoderm differentiation",
    }
    labels = short.copy()

    grn = d[d.metric_family == "full_gene_tftarget_epr"].copy()
    valid_networks = ["STRING", "ChIP", "NonSpecific"]
    grn = grn[grn.network.isin(valid_networks)]
    grn_summary = (grn.groupby("config_label", as_index=False)
                   .agg(mean_epr=("value", "mean"), min_epr=("value", "min"),
                        max_epr=("value", "max"), n_networks=("network", "nunique"))
                   .merge(sizes, on="config_label", how="left"))

    expr = d[d.metric_family == "expression_prediction"].copy()
    expr_labels = {
        "pearson": "Overall",
        "genewise_pearson": "Gene-wise",
        "cellwise_pearson": "Cell-wise",
    }
    corr = expr[expr.network.isin(expr_labels)].copy()
    corr["metric"] = corr.network.map(expr_labels)
    export = []
    for _, row in grn_summary.iterrows():
        export.append({"config_label": row.config_label, "dataset": row.dataset,
                       "training_transition_pairs": int(row.training_transition_pairs),
                       "metric": "mean_full_gene_EPR", "value": row.mean_epr,
                       "aggregation": "arithmetic mean across STRING, ChIP and NonSpecific"})
    for _, row in corr.iterrows():
        export.append({"config_label": row.config_label, "dataset": row.dataset,
                       "training_transition_pairs": int(meta.loc[row.config_label, "training_transition_pairs"]),
                       "metric": row.network, "value": row.value, "aggregation": "none"})
    pd.DataFrame(export).to_csv(DATA / "FigureS4_scale_aware_aggregated.csv", index=False)

    fig = plt.figure(figsize=(7.15, 7.10))
    gs = fig.add_gridspec(3, 1, hspace=.68)
    y = np.arange(len(order))

    ax = fig.add_subplot(gs[0, 0])
    q = grn_summary.set_index("config_label").reindex(order)
    bars = ax.barh(y, q.mean_epr, color=COLORS["litd"], height=.56)
    ax.set_yticks(y, [labels[k] for k in order])
    ax.invert_yaxis()
    ax.set_xlabel("Mean EPR across 3 reference networks")
    panel(ax, "A")
    clean(ax)

    corr_mat = (corr.pivot(index="config_label", columns="metric", values="value")
                .reindex(index=order, columns=["Overall", "Gene-wise"]))
    for idx, (metric, xlabel) in enumerate([("Overall", "Overall Pearson"),
                                            ("Gene-wise", "Gene-wise Pearson")], start=1):
        ax = fig.add_subplot(gs[idx, 0])
        values = corr_mat[metric].to_numpy(float)
        bars = ax.barh(y, values, color=COLORS["litd"], height=.56)
        ax.set_yticks(y, [labels[k] for k in order])
        ax.invert_yaxis(); ax.set_xlabel(xlabel); ax.set_ylabel("")
        panel(ax, "BC"[idx-1]); clean(ax)
        xmax = 1.02 if metric == "Overall" else float(np.nanmax(values)) * 1.34
        ax.set_xlim(0, xmax)

    fig.subplots_adjust(left=.24, right=.97, top=.96, bottom=.08)
    save(fig, SUPP, "Supplementary_Figure_S4_scale_aware_context")
    note_for("Supplementary_Figure_S4", "Supplementary Figure S4. Scale-aware context",
             ["scale", "scale_training_sizes"],
             ["A: mean full-gene TF-target EPR across STRING, ChIP and NonSpecific",
              "B: overall next-state Pearson correlation",
              "C: gene-wise next-state Pearson correlation"],
             "Rows show hHep, hESC and endoderm differentiation; sample counts are omitted. "
             "Only the three reference-network EPR values are pooled, using an arithmetic mean; "
             "MSE is excluded because expression scales and gene spaces differ across datasets. "
             "Pearson values remain descriptive cross-dataset context rather than a controlled ranking.")


def _unused_old_s4():
    d=pd.read_csv(FILES["weighted_path_top1000"]).copy();d["path"]=d[["ligand","receptor","tf","target_gene"]].astype(str).agg("→".join,axis=1)
    top20=d.head(20).sort_values("global_weighted_pathscore")
    fig,axs=plt.subplots(1,2,figsize=(15.5,7.5),gridspec_kw={"wspace":.48})
    axs[0].barh(top20.path,top20.global_weighted_pathscore,color=[COLORS["entry"] if x.endswith("→LIMD1") else "#8da3b6" for x in top20.path]);axs[0].set_xlabel("GlobalPathScore");panel(axs[0],"A","Exact global Top 20 fixed paths");clean(axs[0]);axs[0].ticklabel_format(axis="x",style="plain",useOffset=False)
    counts=pd.DataFrame({"level":["Ligand","Receptor","TF","Target gene"],"unique":[d.ligand.nunique(),d.receptor.nunique(),d.tf.nunique(),d.target_gene.nunique()]})
    bars=axs[1].bar(counts.level,counts.unique,color=["#d88943","#7892aa","#7a6aa6","#879b66"]);panel(axs[1],"B","Hub concentration within global Top 1,000");axs[1].set_ylabel("Number of unique entities");clean(axs[1])
    axs[1].text(.03,.94,"All Top 1,000 share\nMDK → ITGB1 → PTMA",transform=axs[1].transAxes,va="top",fontsize=9,weight="bold",color=COLORS["entry"])
    fig.suptitle("Supplementary Figure S4. Extended exact global PathScore landscape",fontsize=13,weight="bold");save(fig,SUPP,"Supplementary_Figure_S4_global_PathScore_landscape")
    note_for("Supplementary_Figure_S4","Supplementary Figure S4",["weighted_path_top1000"],["A: exact global Top 20","B: Top-1,000 hub concentration"],"All entries come from the same exact all-ligand/all-579-TF/all-10,112-gene ranking used in Figure 6. Concentration is reported as a model diagnostic and is not causal evidence.")


def _unused_old_s5():
    chains=pd.read_csv(FILES["relay_chains"]).drop_duplicates(subset=["path_1","path_2","path_3"]).nlargest(10,"relay_chain_score").sort_values("relay_chain_score");edges=pd.read_csv(FILES["relay_edges"])
    fig,axs=plt.subplots(1,2,figsize=(15.5,8))
    labs=(chains.path_1+" | "+chains.path_2+" | "+chains.path_3).str.replace("->","→",regex=False);axs[0].barh(labs,chains.relay_chain_score,color="#806195");axs[0].set_xscale("log");axs[0].set_xlabel("Relay chain score (log)");panel(axs[0],"A","Top alternative relay chains");clean(axs[0])
    edges["edge_label"]=edges.upstream_tf+"→"+edges.relay_ligand+"→"+edges.downstream_tf
    edges=edges.groupby("edge_label",as_index=False).relay_score.max().nlargest(12,"relay_score").sort_values("relay_score")
    axs[1].barh(edges.edge_label,edges.relay_score,color="#5c7894");axs[1].set_xscale("log");axs[1].set_xlabel("Relay edge score (log)");panel(axs[1],"B","Top distinct relay edges");clean(axs[1])
    fig.suptitle("Supplementary Figure S5. Relay alternatives and candidate space",fontsize=13,weight="bold");save(fig,SUPP,"Supplementary_Figure_S5_relay_alternatives")
    note_for("Supplementary_Figure_S5","Supplementary Figure S5",["relay_chains","relay_edges"],["A: alternative chains","B: alternative relay edges"],"These are candidate-space rankings and remain hypothesis-generating.")


LATENT_MAIN_STEM = "Figure8_hierarchical_latent_interpretability"
LATENT_SUPP_STEM = "Supplementary_Figure_S5_hierarchical_latent_interpretability"


def cache_latent_outputs():
    """Preserve the externally generated latent figure before main-figure cleanup."""
    for suffix in [".pdf", ".svg", ".ai", "_600dpi.png"]:
        src = MAIN / f"{LATENT_MAIN_STEM}{suffix}"
        cache = DATA / f"latent_interpretability_cached{suffix}"
        if not cache.exists():
            if not src.exists():
                raise FileNotFoundError(f"Missing latent interpretability source: {src}")
            shutil.copy2(src, cache)
    source_note = OUT / "Figure8_原始数据说明.md"
    cached_note = DATA / "latent_interpretability_source_note.md"
    if source_note.exists() and not cached_note.exists():
        shutil.copy2(source_note, cached_note)


def make_s5_latent():
    coord_path = DATA / "interpretability_cell_umap_coordinates_full_24425__cell_umap_coordinates_full_24425.csv"
    metric_path = DATA / "interpretability_cell_representation_metrics__cell_representation_metrics.csv"
    if not coord_path.exists():
        raise FileNotFoundError(f"Missing full-cell UMAP coordinates: {coord_path}")
    cells = pd.read_csv(coord_path)
    metrics = pd.read_csv(metric_path)
    layer_order = ["raw_gene", "projected_gene", "gene_cross", "gene_final"]
    layer_labels = {"raw_gene":"Frozen scFoundation gene representation",
                    "projected_gene":"Projected gene representation",
                    "gene_cross":"TF-informed gene representation",
                    "gene_final":"Final gene representation"}
    titles = ["Frozen scFoundation gene representation", "Projected gene representation",
              "TF-informed gene representation", "Final gene representation"]
    window_order = ["day0_to_day1", "day1_to_day2", "day2_to_day3"]
    window_labels = {"day0_to_day1":"day0→1", "day1_to_day2":"day1→2", "day2_to_day3":"day2→3"}
    window_colors = {"day0_to_day1":"#B83A4B", "day1_to_day2":"#3F6FA9", "day2_to_day3":"#7853A6"}
    counts = cells.groupby("layer").size().reindex(layer_order)
    if counts.isna().any() or not (counts == 24425).all():
        raise ValueError(f"Expected 24,425 cells per layer; observed {counts.to_dict()}")

    fig = plt.figure(figsize=(7.15, 8.70))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, .62], hspace=.30, wspace=.24)
    for idx, (layer, title) in enumerate(zip(layer_order, titles)):
        row,col=divmod(idx,2)
        ax = fig.add_subplot(gs[row, col]); panel(ax, "ABCD"[idx], title)
        part = cells[cells.layer.eq(layer)]
        for window in window_order:
            group = part[part.window.eq(window)]
            ax.scatter(group.UMAP1, group.UMAP2, s=2.0, alpha=.38, linewidths=0,
                       color=window_colors[window], rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color("#c9d0d8"); spine.set_linewidth(.7)
        ax.set_box_aspect(1.0)
        ax.set_aspect("equal", adjustable="datalim")
        ax.text(.02, .02, "n = 24,425", transform=ax.transAxes, fontsize=6.8, color="#5f6b76")

    ax = fig.add_subplot(gs[2, :]); panel(ax, "E", "Developmental organization across latent representations")
    metric_order = ["stage_silhouette", "balanced_knn_stage_accuracy", "abs_pc1_day_spearman"]
    metric_labels = ["Stage silhouette score", "Balanced kNN accuracy", "|PC1–day| Spearman"]
    metric_colors = ["#9BB4C9", "#1F4E79", "#5F86A6"]
    table = (metrics[metrics.layer.isin(layer_order) & metrics.metric.isin(metric_order)]
             .pivot(index="layer", columns="metric", values="value")
             .reindex(index=layer_order, columns=metric_order))
    x = np.arange(len(layer_order)); width = .23
    for j, (metric, label, color) in enumerate(zip(metric_order, metric_labels, metric_colors)):
        values = table[metric].to_numpy(float)
        bars = ax.bar(x + (j-1)*width, values, width, label=label, color=color)
    compact_layer_labels={
        "raw_gene":"Frozen scFoundation\ngene representation",
        "projected_gene":"Projected gene\nrepresentation",
        "gene_cross":"TF-informed gene\nrepresentation",
        "gene_final":"Final gene\nrepresentation",
    }
    ax.set_xticks(x, [compact_layer_labels[v] for v in layer_order])
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", color=COLORS["grid"], lw=.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.23), fontsize=7.2)

    handles = [Line2D([0],[0], marker="o", linestyle="none", markersize=5,
                      markerfacecolor=window_colors[w], markeredgewidth=0,
                      label=window_labels[w]) for w in window_order]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.965),
               ncol=3, frameon=False, fontsize=7.5)
    fig.subplots_adjust(top=.89, bottom=.12, left=.09, right=.985)
    save(fig, SUPP, LATENT_SUPP_STEM)

    text = f"""# Supplementary Figure S5. Hierarchical latent representation interpretability

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
  - local snapshot: `{coord_path}`
  - rows: `{len(cells):,}` (`24,425 cells × 4 layers`)
  - SHA-256: `{sha256(coord_path)}`
- `cell_representation_metrics`
  - local snapshot: `{metric_path}`
  - SHA-256: `{sha256(metric_path)}`
- `full_cell_umap_script`
  - local scaffold: `{SCAFF / 'Supplementary_Figure_S5' / 'compute_full_cell_umap.py'}`

## UMAP parameters

- Each layer was fitted independently on all 24,425 unique source cells.
- `n_neighbors=30`, `min_dist=0.3`, `metric=cosine`, base seed `42` with deterministic layer offsets.
- UMAP coordinates are visualization-only. Panel E uses the original latent-space metrics and not the UMAP coordinates.

## Evidence boundary

This figure provides model-internal representation evidence. It shows developmental-state organization in learned latent spaces and does not establish a causal biological mechanism.

## Rebuild

Run `python build_all_figures.py` from this directory after the full-cell coordinate snapshot is present.
"""
    (OUT / "Supplementary_Figure_S5_原始数据说明.md").write_text(text, encoding="utf-8")
    (DATA / "latent_interpretability_source_note.md").write_text(text, encoding="utf-8")
    scaffold = SCAFF / "Supplementary_Figure_S5"; scaffold.mkdir(parents=True, exist_ok=True)
    (scaffold / "README_原始数据与scaffold.md").write_text(text, encoding="utf-8")


def remove_superseded_outputs():
    stems=["Figure3_ablation_analysis","Figure3_weak_correlation_GRN_recovery_EPR","Figure4_ablation_analysis",
           "Figure4_next_state_expression_prediction","Figure5_next_state_expression_prediction",
           "Figure5_MDK_ITGB1_PTMA_LIMD1_global_PathScore","Figure6_candidate_relay_TF_handoff",
           "Figure6_MDK_ITGB1_PTMA_LIMD1_global_PathScore","Figure7_candidate_relay_TF_handoff","Figure8_hierarchical_latent_interpretability",
           "Figure4_CD44_FOXA2_TDGF1_shared_backbone","Figure5_FGF17_FGFR1_GATA6_DKK1_entry_axis",
           "Supplementary_Figure_S1_complete_benchmark_landscape","Supplementary_Figure_S1_dataset_level_correlation_stratified_GRN_recovery",
           "Supplementary_Figure_S2_full_ablation_details","Supplementary_Figure_S3_scale_aware_context",
           "Supplementary_Figure_S3_next_state_regression","Supplementary_Figure_S3_setting_level_correlation_stratified_ablation",
           "Supplementary_Figure_S4_next_state_expression_prediction",
           "Supplementary_Figure_S4_global_PathScore_landscape","Supplementary_Figure_S5_relay_alternatives",
           "Supplementary_Figure_S5_window_axis_landscape","Supplementary_Figure_S6_relay_alternatives",
           "Supplementary_Figure_S4_window_axis_landscape"]
    for folder in [MAIN,SUPP]:
        for stem in stems:
            for suffix in [".pdf",".svg",".ai","_600dpi.png"]:
                p=folder/(stem+suffix)
                if p.exists(): p.unlink()
    stale_note=OUT/"Supplementary_Figure_S6_原始数据说明.md"
    if stale_note.exists(): stale_note.unlink()
    stale_figure7_note=OUT/"Figure7_原始数据说明.md"
    if stale_figure7_note.exists(): stale_figure7_note.unlink()
    stale_scaffold=SCAFF/"Supplementary_Figure_S6"/"README_原始数据与scaffold.md"
    if stale_scaffold.exists(): stale_scaffold.unlink()
    for name in ["Figure4_BackboneScore_top10.csv", "Figure4_expression_prediction_hesc_hhep_mean.csv",
                 "Figure4_expression_prediction_paper_subset.csv", "Figure5_global_weighted_PathScore_top5.csv"]:
        stale_data = DATA / name
        if stale_data.exists(): stale_data.unlink()


def combine_pdfs():
    files=sorted(MAIN.glob("Figure*.pdf"))+sorted(SUPP.glob("Supplementary_Figure*.pdf"))
    writer=PdfWriter()
    for f in files:
        reader=PdfReader(str(f))
        for source_page in reader.pages:
            source_width = float(source_page.mediabox.width)
            source_height = float(source_page.mediabox.height)
            review_page = PageObject.create_blank_page(width=A4_WIDTH_PT, height=A4_HEIGHT_PT)
            x_offset = (A4_WIDTH_PT - source_width) / 2.0
            y_offset = (A4_HEIGHT_PT - source_height) / 2.0
            review_page.merge_translated_page(source_page, x_offset, y_offset)
            writer.add_page(review_page)
    out=OUT/"LITD_全部主图与附图_逐图分页总览.pdf"
    with out.open("wb") as fp: writer.write(fp)
    shutil.copy2(out, OUT/"LITD投稿图片.pdf")
    (OUT/"LITD_全部主图与附图_逐图分页总览.sha256").write_text(f"{sha256(out)}  {out.name}\n",encoding="utf-8")
    return files,out


def write_manifest(files, combined):
    rows=[]
    for f in files:
        stem=f.stem
        note_key = stem.split("_")[0] if stem.startswith("Figure") else "_".join(stem.split("_")[:3])
        page = PdfReader(str(f)).pages[0]
        width_pt = float(page.mediabox.width)
        height_pt = float(page.mediabox.height)
        rows.append({"figure":stem,"pdf":str(f),"svg":str(f.with_suffix('.svg')),"ai":str(f.with_suffix('.ai')),
                     "png_600dpi":str(f.with_name(stem+'_600dpi.png')),"source_note":str(OUT/(note_key+'_原始数据说明.md')),
                     "pdf_width_in":width_pt/72.0,"pdf_height_in":height_pt/72.0})
    pd.DataFrame(rows).to_csv(OUT/"FIGURE_MANIFEST.csv",index=False)
    (OUT/"README.md").write_text(
        "# LITD 20260913 figures\n\nAll figures were built with Python from the snapshotted inputs in `data_used/`. "
        "Each figure has PDF, SVG, PDF-compatible AI, and 600-dpi PNG outputs. "
        f"All individual figures use a fixed {FINAL_WIDTH_IN:.2f}-inch publication width with content-dependent heights. "
        "The `scaffolds/` folders document panel layout, exact inputs, checksums, and evidence boundaries.\n\n"
        f"Combined review PDF: `{combined.name}` (A4 pages at a constant display scale)\n",encoding="utf-8")


def main():
    ensure_dirs();remove_superseded_outputs();snapshot_sources();make_figure1();make_figure2();make_figure3();make_figure4();make_figure5();make_figure6();make_s1_expression();make_s2();make_s3_relay();make_s4();make_s5_latent();files,combined=combine_pdfs();write_manifest(files,combined)
    print(json.dumps({"figures":len(files),"combined":str(combined)},ensure_ascii=False))


if __name__ == "__main__": main()

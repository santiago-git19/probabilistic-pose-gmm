"""
Publication-Grade 3x3 Matrix Visualization Suite for High-Impact Scientific Journals (Elsevier / IEEE).

Combines:
1. Exact mathematical 1:1 scale with LaTeX \\textwidth (7.2 x 7.8 inches).
2. Typographic harmony matching LaTeX captions (8.5 - 9.5 pt) for crystal-clear 100% zoom reading.
3. Premium scientific styling: Curated HSL color palette, rounded glassmorphic metric badges,
   refined grid spines, and a unified external global banner.

Usage:
    poetry run python src/experiments/uncertainty/plot_3x3_matrices.py --models hrnet_w32 resnet50 vitpose_small
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

# Configure root paths
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.experiments.uncertainty.evaluate_uncertainty import _unroll_keypoints, compute_ause

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# Premium Typographic & Aesthetic Configuration
# ==============================================================================

RC_PARAMS_PREMIUM = {
    # Typography
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Computer Modern Sans"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": "#1A1A1A",
    "axes.labelsize": 8.5,
    "axes.labelweight": "normal",
    "axes.labelcolor": "#222222",
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.fontsize": 8.0,
    "legend.title_fontsize": 8.5,
    "figure.titlesize": 11.0,
    # Spines & Grid
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.75,
    "axes.facecolor": "#FFFFFF",
    "figure.facecolor": "#FFFFFF",
    "grid.color": "#E2E2E2",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.75,
}

# Curated High-Impact Color Palette (Publication Standard)
AESTHETIC_PALETTE = {
    "ours_volume": {
        "label": r"Ours (Volume)",
        "color": "#1E88E5",
        "linestyle": "-",
        "linewidth": 1.2,          # <-- Antes 2.0 (ahora más fina)
        "zorder": 5,
    },
    "ours_uniform": {
        "label": r"Ours (Uniform $\pi_u$)",
        "color": "#FB8C00",
        "linestyle": "-",
        "linewidth": 1.1,          # <-- Antes 1.8
        "zorder": 4,
    },
    "dark_entropy": {
        "label": r"DARK (Entropy)",
        "color": "#2E7D32",
        "linestyle": "-.",
        "linewidth": 1.0,          # <-- Antes 1.6
        "zorder": 3,
    },
    "dark_inv_prob": {
        "label": r"DARK ($1 - P_{\mathrm{DARK}}$)",
        "color": "#D32F2F",
        "linestyle": "--",
        "linewidth": 1.1,          # <-- Antes 1.8
        "zorder": 4,
    },
    "random_guess": {
        "label": r"Random Guess",
        "color": "#9E9E9E",
        "linestyle": ":",
        "linewidth": 0.8,          # <-- Antes 1.1 (línea guía fina)
        "zorder": 1,
    },
    "oracle": {
        "label": r"Oracle (Optimal)",
        "color": "#212121",
        "linestyle": "--",
        "linewidth": 1.1,          # <-- Antes 1.6
        "zorder": 2,
    },
}



def load_dataset_unrolled(model_name: str, dataset_name: str, paper_root: Path) -> pd.DataFrame:
    """Resolve and load unrolled keypoint dataset for a given model and dataset."""
    model_dir_map = {
        "hrnet_w32": f"hrnet_w32_{dataset_name.lower()}",
        "resnet50": f"Modelos_Secundarios/resnet50_{dataset_name.lower()}",
        "vitpose_small": f"Modelos_Secundarios/vitpose_small_{dataset_name.lower()}",
    }
    subpath = model_dir_map.get(model_name.lower(), f"{model_name}_{dataset_name.lower()}")
    parquet_path = paper_root / "Resultados_Incertidumbre" / subpath / "degradation_benchmark" / "all_degradations_combined.parquet"

    if not parquet_path.exists():
        logger.warning("Parquet not found: %s", parquet_path)
        return pd.DataFrame()

    df_raw = pd.read_parquet(parquet_path)
    df_unrolled = _unroll_keypoints(df_raw)
    logger.info("Loaded %s - %s: %d raw images -> %d keypoint records", model_name, dataset_name, len(df_raw), len(df_unrolled))
    return df_unrolled


# ==============================================================================
# 1. 3x3 OoD ROC Matrix (Refined Scientific Aesthetics)
# ==============================================================================

def plot_3x3_roc_matrix(
    model_name: str,
    datasets_data: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """Generate aesthetic 3x3 ROC Matrix with 1:1 LaTeX calibration and external legend."""
    plt.rcParams.update(RC_PARAMS_PREMIUM)
    sns.set_theme(style="whitegrid", rc=RC_PARAMS_PREMIUM)

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.8), sharex=True, sharey=True)
    datasets = ["COCO", "CrowdPose", "OCHuman"]
    regimes = [
        ("Global", "All", None),
        ("Unimodal", "$K = 1$", 1),
        ("Bimodal", "$K = 2$", 2),
    ]

    panel_sublabels = [
        ["(a) COCO (Global)", "(b) CrowdPose (Global)", "(c) OCHuman (Global)"],
        ["(d) COCO ($K = 1$)", "(e) CrowdPose ($K = 1$)", "(f) OCHuman ($K = 1$)"],
        ["(g) COCO ($K = 2$)", "(h) CrowdPose ($K = 2$)", "(i) OCHuman ($K = 2$)"],
    ]

    legend_handles = {}

    for row_idx, (regime_title, regime_label, k_filter) in enumerate(regimes):
        for col_idx, ds_name in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            df = datasets_data.get(ds_name, pd.DataFrame())

            if not df.empty and k_filter is not None:
                df_sub = df[df["n_components"] == k_filter].copy()
            else:
                df_sub = df.copy()

            # Elegant Subpanel Title
            sublabel = panel_sublabels[row_idx][col_idx]
            ax.set_title(sublabel, fontsize=9.5, fontweight="bold", pad=5)

            # Random Guess Reference Line
            line_rand = ax.plot(
                [0, 1], [0, 1],
                color=AESTHETIC_PALETTE["random_guess"]["color"],
                linestyle=AESTHETIC_PALETTE["random_guess"]["linestyle"],
                linewidth=AESTHETIC_PALETTE["random_guess"]["linewidth"],
                zorder=AESTHETIC_PALETTE["random_guess"]["zorder"],
                label=AESTHETIC_PALETTE["random_guess"]["label"],
            )[0]
            legend_handles["Random Guess"] = line_rand

            auc_scores = {}
            if not df_sub.empty and "vis" in df_sub.columns:
                df_valid = df_sub[df_sub["vis"].notna()].copy()
                y_true = (df_valid["vis"] == 0).astype(int)

                if y_true.nunique() >= 2 and y_true.sum() > 0 and (1 - y_true).sum() > 0:
                    # 1. Ours (Volume)
                    if "cov_det" in df_valid.columns:
                        sc = df_valid["cov_det"]
                        mask = ~sc.isna()
                        if mask.sum() > 0 and y_true[mask].nunique() >= 2:
                            fpr, tpr, _ = roc_curve(y_true[mask], sc[mask])
                            auc_val = auc(fpr, tpr)
                            auc_scores["Vol"] = auc_val
                            line_vol = ax.plot(
                                fpr, tpr,
                                color=AESTHETIC_PALETTE["ours_volume"]["color"],
                                linestyle=AESTHETIC_PALETTE["ours_volume"]["linestyle"],
                                linewidth=AESTHETIC_PALETTE["ours_volume"]["linewidth"],
                                zorder=AESTHETIC_PALETTE["ours_volume"]["zorder"],
                                label=AESTHETIC_PALETTE["ours_volume"]["label"],
                            )[0]
                            legend_handles["ours_volume"] = line_vol

                    # 2. Ours (Uniform)
                    if "uniform_weight" in df_valid.columns:
                        sc = df_valid["uniform_weight"]
                        mask = ~sc.isna()
                        if mask.sum() > 0 and y_true[mask].nunique() >= 2:
                            fpr, tpr, _ = roc_curve(y_true[mask], sc[mask])
                            auc_val = auc(fpr, tpr)
                            auc_scores["Uni"] = auc_val
                            line_uni = ax.plot(
                                fpr, tpr,
                                color=AESTHETIC_PALETTE["ours_uniform"]["color"],
                                linestyle=AESTHETIC_PALETTE["ours_uniform"]["linestyle"],
                                linewidth=AESTHETIC_PALETTE["ours_uniform"]["linewidth"],
                                zorder=AESTHETIC_PALETTE["ours_uniform"]["zorder"],
                                label=AESTHETIC_PALETTE["ours_uniform"]["label"],
                            )[0]
                            legend_handles["ours_uniform"] = line_uni

                    # 3. DARK (Entropy)
                    if "heatmap_entropy" in df_valid.columns:
                        sc = df_valid["heatmap_entropy"]
                        mask = ~sc.isna()
                        if mask.sum() > 0 and y_true[mask].nunique() >= 2:
                            fpr, tpr, _ = roc_curve(y_true[mask], sc[mask])
                            auc_val = auc(fpr, tpr)
                            auc_scores["Ent"] = auc_val
                            line_ent = ax.plot(
                                fpr, tpr,
                                color=AESTHETIC_PALETTE["dark_entropy"]["color"],
                                linestyle=AESTHETIC_PALETTE["dark_entropy"]["linestyle"],
                                linewidth=AESTHETIC_PALETTE["dark_entropy"]["linewidth"],
                                zorder=AESTHETIC_PALETTE["dark_entropy"]["zorder"],
                                label=AESTHETIC_PALETTE["dark_entropy"]["label"],
                            )[0]
                            legend_handles["dark_entropy"] = line_ent

                    # 4. DARK (1 - P_DARK)
                    if "base_score" in df_valid.columns:
                        sc = 1.0 - df_valid["base_score"]
                        mask = ~sc.isna()
                        if mask.sum() > 0 and y_true[mask].nunique() >= 2:
                            fpr, tpr, _ = roc_curve(y_true[mask], sc[mask])
                            auc_val = auc(fpr, tpr)
                            auc_scores["DARK"] = auc_val
                            line_dark = ax.plot(
                                fpr, tpr,
                                color=AESTHETIC_PALETTE["dark_inv_prob"]["color"],
                                linestyle=AESTHETIC_PALETTE["dark_inv_prob"]["linestyle"],
                                linewidth=AESTHETIC_PALETTE["dark_inv_prob"]["linewidth"],
                                zorder=AESTHETIC_PALETTE["dark_inv_prob"]["zorder"],
                                label=AESTHETIC_PALETTE["dark_inv_prob"]["label"],
                            )[0]
                            legend_handles["dark_inv_prob"] = line_dark

                        # Aesthetic AUROC Badge Card (Bottom-Right HUD)
            if auc_scores:
                lines = [f"{k}: {v:.5f}" for k, v in auc_scores.items()]
                score_str = "\n".join(lines)
                
                ax.text(
                    0.96, 0.04, score_str,
                    transform=ax.transAxes,
                    fontsize=5.8,                    
                    family="monospace",              
                    fontweight="normal",             
                    color="#2B2B2B",
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    bbox=dict(
                        boxstyle="round,pad=0.25,rounding_size=0.15", # Padding más compacto
                        facecolor="#F8F9FA",
                        edgecolor="#CCCCCC",
                        alpha=0.92,
                        linewidth=0.5,
                    ),
                    zorder=10,
                )


            # Axis limits & ticks
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal", "box")
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
            ax.tick_params(axis="both", which="major", length=3, width=0.6, labelsize=7.5)
            ax.grid(True, linestyle="--", alpha=0.5, color="#D8D8D8", linewidth=0.5)

            # Redundancy Elimination
            if col_idx == 0:
                ax.set_ylabel(r"True Positive Rate (TPR)", fontsize=8.5, fontweight="bold")
            else:
                ax.set_ylabel("")

            if row_idx == 2:
                ax.set_xlabel(r"False Positive Rate (FPR)", fontsize=8.5, fontweight="bold")
            else:
                ax.set_xlabel("")

    # Unified Premium Global Legend at Top
    ordered_keys = ["ours_volume", "ours_uniform", "dark_entropy", "dark_inv_prob", "Random Guess"]
    handles = [legend_handles[k] for k in ordered_keys if k in legend_handles]
    labels = [h.get_label() for h in handles]

    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(handles),
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        framealpha=0.98,
        fontsize=8.0,
        columnspacing=1.1,
        handletextpad=0.35,
        handlelength=1.6,
    )

    plt.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.07, wspace=0.14, hspace=0.22)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"matrix_3x3_ood_roc_{model_name}.pdf"
    png_path = out_dir / f"matrix_3x3_ood_roc_{model_name}.png"

    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    logger.info("Saved aesthetic 3x3 ROC Matrix to:\n  - %s\n  - %s", pdf_path, png_path)
    plt.close(fig)


# ==============================================================================
# 2. 3x3 ECE Calibration Matrix (Exact evaluate_uncertainty.py Style in 3x3 Grid)
# ==============================================================================

def plot_3x3_ece_matrix(
    model_name: str,
    datasets_data: Dict[str, pd.DataFrame],
    out_dir: Path,
    n_bins: int = 10,
) -> None:
    """Generate 3x3 ECE Calibration Matrix exactly replicating evaluate_uncertainty.py."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "cm"

    fig, axes = plt.subplots(3, 3, figsize=(15.0, 15.6), facecolor="white")
    datasets = ["COCO", "CrowdPose", "OCHuman"]
    regimes = [
        ("All", None),
        ("1 Gaussian", 1),
        ("2 Gaussians", 2),
    ]

    panel_sublabels = [
        ["(a) COCO (Global)", "(b) CrowdPose (Global)", "(c) OCHuman (Global)"],
        ["(d) COCO ($K = 1$)", "(e) CrowdPose ($K = 1$)", "(f) OCHuman ($K = 1$)"],
        ["(g) COCO ($K = 2$)", "(h) CrowdPose ($K = 2$)", "(i) OCHuman ($K = 2$)"],
    ]

    for row_idx, (regime_title, k_filter) in enumerate(regimes):
        for col_idx, ds_name in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            df = datasets_data.get(ds_name, pd.DataFrame())

            if not df.empty and k_filter is not None:
                df_sub = df[df["n_components"] == k_filter].copy()
            else:
                df_sub = df.copy()

            df_sub = df_sub[df_sub["vis"] > 0].dropna(subset=["oks_ours", "cov_det"]).copy()

            if not df_sub.empty and len(df_sub) >= n_bins:
                df_sub["error"] = 1.0 - df_sub["oks_ours"]
                df_sub["unc_bin"] = pd.qcut(df_sub["cov_det"], q=n_bins, labels=False, duplicates="drop")

                grouped = df_sub.groupby("unc_bin").agg(
                    mean_unc=("cov_det", "mean"),
                    mean_err=("error", "mean"),
                    count=("error", "count")
                ).reset_index()

                x_vals = np.log1p(grouped["mean_unc"])
                y_vals = grouped["mean_err"]

                # Exact scatter and plot matching evaluate_uncertainty.py
                ax.scatter(x_vals, y_vals, s=grouped["count"]/2, alpha=0.6, color="purple", zorder=4)
                ax.plot(x_vals, y_vals, color="purple", linestyle="-", alpha=0.4, linewidth=2.0, zorder=3)

                # Breathing room on X axis for the leftmost/rightmost circles
                x_min, x_max = x_vals.min(), x_vals.max()
                x_span = max(x_max - x_min, 1.0)
                ax.set_xlim(x_min - 0.08 * x_span, x_max + 0.08 * x_span)

                # Dynamic Y-axis limits matching data
                y_min, y_max = y_vals.min(), y_vals.max()
                y_span = max(y_max - y_min, 0.05)
                ax.set_ylim(max(0.0, y_min - 0.08 * y_span), min(1.0, y_max + 0.08 * y_span))

            ax.set_xlabel("Mean Uncertainty (Volume)", fontsize=10.0)
            ax.set_ylabel("Mean Error (1 - OKS)", fontsize=10.0)
            ax.set_title(f"Expected Calibration Error ({regime_title})", fontsize=11.0, pad=6)
            ax.tick_params(axis="both", which="major", labelsize=9.0)
            ax.grid(True, linestyle="--", alpha=0.5, color="#D8D8D8", linewidth=0.5)

            sublabel = panel_sublabels[row_idx][col_idx]
            ax.text(
                0.5, -0.20, sublabel,
                transform=ax.transAxes,
                fontsize=11.5,
                ha="center", va="top",
                color="#000000",
            )

    plt.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.08, wspace=0.25, hspace=0.35)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"matrix_3x3_ece_calibration_{model_name}.pdf"
    png_path = out_dir / f"matrix_3x3_ece_calibration_{model_name}.png"

    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    logger.info("Saved 3x3 ECE Matrix to:\n  - %s\n  - %s", pdf_path, png_path)
    plt.close(fig)


# ==============================================================================
# 3. 3x3 Sparsification Matrix (Refined Scientific Aesthetics)
# ==============================================================================

def plot_3x3_sparsification_matrix(
    model_name: str,
    datasets_data: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """Generate aesthetic 3x3 Sparsification Matrix with 1:1 LaTeX calibration."""
    plt.rcParams.update(RC_PARAMS_PREMIUM)
    sns.set_theme(style="whitegrid", rc=RC_PARAMS_PREMIUM)

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.8), sharex=True, sharey=True)
    datasets = ["COCO", "CrowdPose", "OCHuman"]
    regimes = [
        ("Global", "All", None),
        ("Unimodal", "$K = 1$", 1),
        ("Bimodal", "$K = 2$", 2),
    ]

    panel_sublabels = [
        ["(a) COCO (Global)", "(b) CrowdPose (Global)", "(c) OCHuman (Global)"],
        ["(d) COCO ($K = 1$)", "(e) CrowdPose ($K = 1$)", "(f) OCHuman ($K = 1$)"],
        ["(g) COCO ($K = 2$)", "(h) CrowdPose ($K = 2$)", "(i) OCHuman ($K = 2$)"],
    ]

    legend_handles = {}

    for row_idx, (regime_title, regime_label, k_filter) in enumerate(regimes):
        for col_idx, ds_name in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            df = datasets_data.get(ds_name, pd.DataFrame())

            if not df.empty and k_filter is not None:
                df_sub = df[df["n_components"] == k_filter].copy()
            else:
                df_sub = df.copy()

            sublabel = panel_sublabels[row_idx][col_idx]
            ax.set_title(sublabel, fontsize=9.5, fontweight="bold", pad=5)

            df_sub = df_sub[df_sub["vis"] > 0].dropna(subset=["oks_ours", "cov_det"]).copy()

            ause_scores = {}
            if not df_sub.empty and len(df_sub) > 20:
                df_sub["error_ours"] = 1.0 - df_sub["oks_ours"]
                df_sub["error_base"] = 1.0 - df_sub["oks_base"] if "oks_base" in df_sub.columns else df_sub["error_ours"]

                # 1. Oracle & Ours (Volume)
                ause_vol, fracs, oracle, model_vol = compute_ause(df_sub, "error_ours", "cov_det")
                ause_scores["Vol"] = ause_vol
                line_ora = ax.plot(
                    fracs, oracle,
                    color=AESTHETIC_PALETTE["oracle"]["color"],
                    linestyle=AESTHETIC_PALETTE["oracle"]["linestyle"],
                    linewidth=AESTHETIC_PALETTE["oracle"]["linewidth"],
                    zorder=AESTHETIC_PALETTE["oracle"]["zorder"],
                    label=AESTHETIC_PALETTE["oracle"]["label"],
                )[0]
                legend_handles["Oracle"] = line_ora

                line_vol = ax.plot(
                    fracs, model_vol,
                    color=AESTHETIC_PALETTE["ours_volume"]["color"],
                    linestyle=AESTHETIC_PALETTE["ours_volume"]["linestyle"],
                    linewidth=AESTHETIC_PALETTE["ours_volume"]["linewidth"],
                    zorder=AESTHETIC_PALETTE["ours_volume"]["zorder"],
                    label=AESTHETIC_PALETTE["ours_volume"]["label"],
                )[0]
                legend_handles["ours_volume"] = line_vol
                ax.fill_between(fracs, oracle, model_vol, color=AESTHETIC_PALETTE["ours_volume"]["color"], alpha=0.10)

                # 2. DARK (1 - P_DARK)
                if "base_score" in df_sub.columns and not df_sub["base_score"].isna().all():
                    df_sub["inv_base_score"] = -df_sub["base_score"]
                    ause_base, _, _, model_base = compute_ause(df_sub, "error_base", "inv_base_score")
                    ause_scores["DARK"] = ause_base
                    line_dark = ax.plot(
                        fracs, model_base,
                        color=AESTHETIC_PALETTE["dark_inv_prob"]["color"],
                        linestyle=AESTHETIC_PALETTE["dark_inv_prob"]["linestyle"],
                        linewidth=AESTHETIC_PALETTE["dark_inv_prob"]["linewidth"],
                        zorder=AESTHETIC_PALETTE["dark_inv_prob"]["zorder"],
                        label=AESTHETIC_PALETTE["dark_inv_prob"]["label"],
                    )[0]
                    legend_handles["dark_inv_prob"] = line_dark

                # 3. DARK (Entropy)
                if "heatmap_entropy" in df_sub.columns and not df_sub["heatmap_entropy"].isna().all():
                    ause_ent, _, _, model_ent = compute_ause(df_sub, "error_base", "heatmap_entropy")
                    ause_scores["Ent"] = ause_ent
                    line_ent = ax.plot(
                        fracs, model_ent,
                        color=AESTHETIC_PALETTE["dark_entropy"]["color"],
                        linestyle=AESTHETIC_PALETTE["dark_entropy"]["linestyle"],
                        linewidth=AESTHETIC_PALETTE["dark_entropy"]["linewidth"],
                        zorder=AESTHETIC_PALETTE["dark_entropy"]["zorder"],
                        label=AESTHETIC_PALETTE["dark_entropy"]["label"],
                    )[0]
                    legend_handles["dark_entropy"] = line_ent

            # Aesthetic AUSE Badge Card (Top-Right HUD)
            if ause_scores:
                lines = [f"{k}: {v:.3f}" for k, v in ause_scores.items()]
                score_str = "\n".join(lines)
                ax.text(
                    0.96, 0.96, score_str,
                    transform=ax.transAxes,
                    fontsize=6.8,
                    family="sans-serif",
                    fontweight="bold",
                    color="#2B2B2B",
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(
                        boxstyle="round,pad=0.35,rounding_size=0.2",
                        facecolor="#F8F9FA",
                        edgecolor="#CCCCCC",
                        alpha=0.93,
                        linewidth=0.6,
                    ),
                    zorder=10,
                )

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
            ax.tick_params(axis="both", which="major", length=3, width=0.6, labelsize=7.5)
            ax.grid(True, linestyle="--", alpha=0.5, color="#D8D8D8", linewidth=0.5)

            if col_idx == 0:
                ax.set_ylabel(r"Retained Error ($1 - \mathrm{OKS}$)", fontsize=8.5, fontweight="bold")
            else:
                ax.set_ylabel("")

            if row_idx == 2:
                ax.set_xlabel(r"Fraction of Keypoints Removed", fontsize=8.5, fontweight="bold")
            else:
                ax.set_xlabel("")

    # Unified Premium Global Legend at Top
    ordered_keys = ["Oracle", "Ours (Volume)", "DARK (1-P)", "DARK (Entropy)"]
    handles = [legend_handles[k] for k in ordered_keys if k in legend_handles]
    labels = [h.get_label() for h in handles]

    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles),
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        framealpha=0.98,
        fontsize=8.0,
        columnspacing=1.1,
        handletextpad=0.35,
        handlelength=1.6,
    )

    plt.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.07, wspace=0.14, hspace=0.20)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"matrix_3x3_sparsification_{model_name}.pdf"
    png_path = out_dir / f"matrix_3x3_sparsification_{model_name}.png"

    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    logger.info("Saved aesthetic 3x3 Sparsification Matrix to:\n  - %s\n  - %s", pdf_path, png_path)
    plt.close(fig)


# ==============================================================================
# Main Orchestration Routine
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3x3 publication matrices calibrated for 100% zoom legibility.")
    parser.add_argument("--models", nargs="+", default=["hrnet_w32", "resnet50", "vitpose_small"], help="Model backbones to render")
    default_out_dir = (
        project_root / "outputs" / "figures" / "uncertainty" / "matrices_3x3"
    )
    parser.add_argument("--out-dir", type=str, default=str(default_out_dir), help="Output directory")
    args = parser.parse_args()

    paper_root = (
        project_root.parent / "Paper"
        if (project_root.parent / "Paper").exists()
        else project_root / "Paper"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["COCO", "CrowdPose", "OCHuman"]

    for model_name in args.models:
        logger.info("==================================================")
        logger.info("Generating 3x3 Matrices for Model: %s", model_name.upper())
        logger.info("==================================================")

        datasets_data = {}
        for ds in datasets:
            df = load_dataset_unrolled(model_name, ds, paper_root)
            datasets_data[ds] = df

        plot_3x3_roc_matrix(model_name, datasets_data, out_dir)
        #plot_3x3_ece_matrix(model_name, datasets_data, out_dir)
        #plot_3x3_sparsification_matrix(model_name, datasets_data, out_dir)

    logger.info("All 3x3 matrices successfully generated in: %s", out_dir)


if __name__ == "__main__":
    main()

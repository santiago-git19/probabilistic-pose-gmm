"""
Publication-Grade 1x3 Composite Beta Optimization Visualization Suite (Elsevier / IEEE).

Generates a unified 1x3 composite figure (Global / Visible / Occluded) for the sensitivity
hyperparameter beta optimization in adaptive uncertainty fusion on HRNet-W32 (COCO).

- Panel (a): Global (All Points)
- Panel (b): Visible Joints (vis == 2)
- Panel (c): Occluded Joints (vis == 1)

Key Visual & Publication Standards:
-----------------------------------
- 1:1 Physical Scale with LaTeX \\textwidth (7.2 x 2.85 inches).
- Generous breathing space between top external legend and subplots (top=0.82).
- Clean Inset HUD Badges displaying exact optimal AUSE scores without beta clutter.
- Smaller, non-overlapping axis labels (fontsize=7.2 pt).
- High-precision Vectorial PDF (.pdf) and 300 DPI PNG (.png) output.

Usage:
    poetry run python src/experiments/uncertainty/plot_1x3_beta_optimization.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# Configure root paths
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.experiments.uncertainty.evaluate_uncertainty import _unroll_keypoints
from src.experiments.uncertainty.optimize_adaptive_uncertainty import (
    compute_ause_fast,
    optimize_strategies,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# Premium Typography & Aesthetic Configuration
# ==============================================================================

RC_PARAMS_PREMIUM = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Computer Modern Sans"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": "#1A1A1A",
    "axes.labelsize": 7.5,
    "axes.labelweight": "bold",
    "axes.labelcolor": "#222222",
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.fontsize": 8.0,
    "legend.title_fontsize": 8.5,
    "figure.titlesize": 10.5,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.75,
    "axes.facecolor": "#FFFFFF",
    "figure.facecolor": "#FFFFFF",
    "grid.color": "#B0B0B0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.65,
    "grid.alpha": 0.75,
}

BETA_PALETTE = {
    "softmax": {
        "label": r"Ours (Softmax)",
        "color": "#FB8C00",
        "linestyle": "-",
        "linewidth": 1.0,          
        "zorder": 5,
    },
    "max_pool": {
        "label": r"Ours (Max-Pool)",
        "color": "#E91E63",
        "linestyle": "-",
        "linewidth": 0.9,          
        "zorder": 4,
    },
    "gating_k": {
        "label": r"Ours (Gating $K=2$)",
        "color": "#00BCD4",
        "linestyle": "-",
        "linewidth": 0.9,           
        "zorder": 4,
    },
    "dark_base": {
        "label": r"DARK (Base)",
        "color": "#E53935",
        "linestyle": "--",
        "linewidth": 0.7,          
        "zorder": 6,
    },
    "ours_volume": {
        "label": r"Ours (Volume)",
        "color": "#1E88E5",
        "linestyle": ":",
        "linewidth": 0.7,         
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


def render_1x3_beta_optimization(
    df_unrolled: pd.DataFrame,
    model_name: str,
    dataset_name: str,
    out_dir: Path,
) -> None:
    """Generate publication-grade 1x3 composite figure for beta hyperparameter optimization."""
    if df_unrolled.empty:
        logger.warning("Empty dataframe, skipping 1x3 beta optimization render.")
        return

    df_valid = df_unrolled[df_unrolled["vis"] > 0].dropna(subset=["oks_ours", "cov_det"]).copy()
    if df_valid.empty:
        logger.warning("No valid points with vis > 0, skipping.")
        return

    plt.rcParams.update(RC_PARAMS_PREMIUM)
    sns.set_theme(style="whitegrid", rc=RC_PARAMS_PREMIUM)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), sharex=True, sharey=False)

    beta_candidates = np.logspace(-6, 4, num=300)

    subsets = [
        ("Global (All Points)", df_valid, "(a) Global (All Points)"),
        ("Visible Joints", df_valid[df_valid["vis"] == 2], "(b) Visible Joints"),
        ("Occluded Joints", df_valid[df_valid["vis"] == 1], "(c) Occluded Joints"),
    ]

    legend_handles = {}

    for col_idx, (subset_title, df_sub, sublabel) in enumerate(subsets):
        ax = axes[col_idx]
        ax.set_title(sublabel, fontsize=7.0, fontweight="bold", pad=6)

        if df_sub.empty or len(df_sub) < 10:
            ax.text(0.5, 0.5, "Insufficient Data", ha="center", va="center", fontsize=8.0)
            continue

        res = optimize_strategies(df_sub, beta_candidates)
        betas = res["beta_candidates"]
        mp_hist = res["max_pooling"]["history"]
        gk_hist = res["gating_k"]["history"]
        sm_hist = res["softmax"]["history"]
        base_ause = res["baseline_argmax_ause"]
        gmm_raw_ause = res["baseline_gmm_raw_ause"]

        # 1. Plot Curves
        line_sm = ax.semilogx(
            betas, sm_hist,
            color=BETA_PALETTE["softmax"]["color"],
            linestyle=BETA_PALETTE["softmax"]["linestyle"],
            linewidth=BETA_PALETTE["softmax"]["linewidth"],
            zorder=BETA_PALETTE["softmax"]["zorder"],
            label=BETA_PALETTE["softmax"]["label"],
        )[0]
        legend_handles["softmax"] = line_sm

        line_mp = ax.semilogx(
            betas, mp_hist,
            color=BETA_PALETTE["max_pool"]["color"],
            linestyle=BETA_PALETTE["max_pool"]["linestyle"],
            linewidth=BETA_PALETTE["max_pool"]["linewidth"],
            zorder=BETA_PALETTE["max_pool"]["zorder"],
            label=BETA_PALETTE["max_pool"]["label"],
        )[0]
        legend_handles["max_pool"] = line_mp

        line_gk = ax.semilogx(
            betas, gk_hist,
            color=BETA_PALETTE["gating_k"]["color"],
            linestyle=BETA_PALETTE["gating_k"]["linestyle"],
            linewidth=BETA_PALETTE["gating_k"]["linewidth"],
            zorder=BETA_PALETTE["gating_k"]["zorder"],
            label=BETA_PALETTE["gating_k"]["label"],
        )[0]
        legend_handles["gating_k"] = line_gk

        line_dark = ax.axhline(
            base_ause,
            color=BETA_PALETTE["dark_base"]["color"],
            linestyle=BETA_PALETTE["dark_base"]["linestyle"],
            linewidth=BETA_PALETTE["dark_base"]["linewidth"],
            zorder=BETA_PALETTE["dark_base"]["zorder"],
            label=BETA_PALETTE["dark_base"]["label"],
        )
        legend_handles["dark_base"] = line_dark

        line_vol = ax.axhline(
            gmm_raw_ause,
            color=BETA_PALETTE["ours_volume"]["color"],
            linestyle=BETA_PALETTE["ours_volume"]["linestyle"],
            linewidth=BETA_PALETTE["ours_volume"]["linewidth"],
            zorder=BETA_PALETTE["ours_volume"]["zorder"],
            label=BETA_PALETTE["ours_volume"]["label"],
        )
        legend_handles["ours_volume"] = line_vol

        # 2. Mark Optimal Beta Points (Clean High-Contrast Dots)
        ax.scatter([res['softmax']['beta']], [res['softmax']['ause']], color=BETA_PALETTE["softmax"]["color"], linewidths=0.6, s=15, zorder=6)
        ax.scatter([res['max_pooling']['beta']], [res['max_pooling']['ause']], color=BETA_PALETTE["max_pool"]["color"], linewidths=0.6, s=15, zorder=6)
        ax.scatter([res['gating_k']['beta']], [res['gating_k']['ause']], color=BETA_PALETTE["gating_k"]["color"], linewidths=0.6, s=15, zorder=6)

        # 3. Clean Monospace HUD Inset (Optimal AUSE values without beta clutter)
        hud_lines = [
            f"Softmax:  {res['softmax']['ause']:.4f}",
            f"Max-Pool: {res['max_pooling']['ause']:.4f}",
            f"Gating:   {res['gating_k']['ause']:.4f}",
            f"DARK:     {base_ause:.4f}",
            f"Volume:   {gmm_raw_ause:.4f}",
        ]
        hud_str = "\n".join(hud_lines)

        ax.text(
            0.96, 0.94, hud_str,
            transform=ax.transAxes,
            fontsize=5.5,
            family="monospace",
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.25,rounding_size=0.15",
                facecolor="#F8F9FA",
                edgecolor="#CCCCCC",
                alpha=0.92,
                linewidth=0.5,
            ),
            zorder=10,
        )

        # 4. Axis Limits & Grids
        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", alpha=0.5, color="#D8D8D8", linewidth=0.5)
        ax.tick_params(axis="both", which="major", length=3, width=0.6, labelsize=7.0)

        # Dynamic Y-axis fitting with breathing room
        y_all = mp_hist + gk_hist + sm_hist + [base_ause, gmm_raw_ause]
        y_min, y_max = min(y_all), max(y_all)
        y_span = max(y_max - y_min, 0.05)
        ax.set_ylim(max(0.0, y_min - 0.08 * y_span), y_max + 0.12 * y_span)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

        # Compact, clean non-overlapping X-axis label
        ax.set_xlabel(r"$\beta$ (log scale)", fontsize=7.2, fontweight="bold", labelpad=3)

        # Y-axis label only on leftmost column
        if col_idx == 0:
            ax.set_ylabel(r"Area Under Sparsification Error (AUSE)", fontsize=7, fontweight="bold", labelpad=3)
        else:
            ax.set_ylabel("")

    # Unified Horizontal Global Legend at Top Banner with ample clearance
    ordered_keys = ["softmax", "max_pool", "gating_k", "dark_base", "ours_volume"]
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
        columnspacing=1.0,
        handletextpad=0.35,
        handlelength=1.6,
    )

    # Top=0.82 provides generous clearance between legend and panel titles
    plt.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.18, wspace=0.18)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"beta_optimization_1x3_{model_name}_{dataset_name.lower()}.pdf"
    png_path = out_dir / f"beta_optimization_1x3_{model_name}_{dataset_name.lower()}.png"

    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    logger.info("Saved 1x3 Beta Optimization figure to:\n  - %s\n  - %s", pdf_path, png_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 1x3 publication figures for beta hyperparameter optimization.")
    parser.add_argument("--model", type=str, default="hrnet_w32", help="Model backbone (default: hrnet_w32)")
    default_out_dir = (
        project_root / "outputs" / "figures" / "uncertainty" / "beta_optimization"
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

    logger.info("==================================================")
    logger.info("Generating 1x3 Beta Optimization for: %s - %s", args.model.upper(), args.dataset.upper())
    logger.info("==================================================")
    df = load_dataset_unrolled(args.model, args.dataset, paper_root)
    render_1x3_beta_optimization(df, args.model, args.dataset, out_dir)

    logger.info("1x3 Beta Optimization figure generated in: %s", out_dir)


if __name__ == "__main__":
    main()

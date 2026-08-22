"""
Publication-Grade 1x3 Composite Uniform Weight Boxplot Visualization Suite (Elsevier / IEEE).

Generates a unified 1x3 composite figure (COCO / CrowdPose / OCHuman) showing the
distribution of the uniform component weight (pi_uniform) as visual degradation escalates:

- Panel (a): COCO
- Panel (b): CrowdPose
- Panel (c): OCHuman

Key Visual & Publication Standards:
-----------------------------------
- 1:1 Physical Scale with LaTeX \\textwidth (7.2 x 2.80 inches).
- LaTeX Typographic Harmony matching caption scale (7.0 - 7.5 pt).
- Unified Grid matching OoD ROC style (linestyle="--", color="#D8D8D8", alpha=0.5, linewidth=0.5).
- Redundancy Elimination: Y-axis label only on panel (a).
- High-precision Vectorial PDF (.pdf) and 300 DPI PNG (.png) output.

Usage:
    poetry run python src/experiments/uncertainty/plot_1x3_uniform_weight_boxplot.py --model hrnet_w32
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# Configure root paths
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# Premium Typography & Aesthetic Configuration
# ==============================================================================

RC_PARAMS_PREMIUM = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Computer Modern Sans"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 7.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": "#1A1A1A",
    "axes.labelsize": 7.2,
    "axes.labelweight": "bold",
    "axes.labelcolor": "#222222",
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.8,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.75,
    "axes.facecolor": "#FFFFFF",
    "figure.facecolor": "#FFFFFF",
    "grid.color": "#D8D8D8",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.5,
}

DEGRADATION_ORDER = [
    "00_Baseline_Clean",
    "01_Resize_Low",
    "02_Resize_Medium",
    "03_Resize_High",
    "04_Resize_Extreme",
]


def load_raw_dataset(model_name: str, dataset_name: str, paper_root: Path) -> pd.DataFrame:
    """Load combined degradation parquet for a given model and dataset."""
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

    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %s - %s: %d records", model_name, dataset_name, len(df))
    return df


def render_1x3_uniform_weight_boxplot(
    datasets_data: Dict[str, pd.DataFrame],
    model_name: str,
    out_dir: Path,
) -> None:
    """Render publication-grade 1x3 composite boxplot figure."""
    plt.rcParams.update(RC_PARAMS_PREMIUM)
    sns.set_theme(style="whitegrid", rc=RC_PARAMS_PREMIUM)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.80), sharex=True, sharey=False)

    panels = [
        ("COCO", "(a) COCO"),
        ("CrowdPose", "(b) CrowdPose"),
        ("OCHuman", "(c) OCHuman"),
    ]

    for col_idx, (ds_name, sublabel) in enumerate(panels):
        ax = axes[col_idx]
        df = datasets_data.get(ds_name, pd.DataFrame())

        ax.set_title(sublabel, fontsize=7.5, fontweight="bold", pad=6)

        if df.empty or "uniform_weight_mean" not in df.columns or "experiment_name" not in df.columns:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=7.0)
            continue

        # Filter and order degradations
        df_plot = df[df["experiment_name"].isin(DEGRADATION_ORDER)].copy()

        # Render refined boxplot
        sns.boxplot(
            data=df_plot,
            x="experiment_name",
            y="uniform_weight_mean",
            hue="experiment_name",
            legend=False,
            order=DEGRADATION_ORDER,
            ax=ax,
            palette="Reds",
            linewidth=0.75,
            fliersize=2.2,
            flierprops=dict(
                marker="o",
                markersize=2.0,
                markeredgecolor="#444444",
                markerfacecolor="none",
                alpha=0.45,
                markeredgewidth=0.4,
            ),
            medianprops=dict(color="#1A1A1A", linewidth=1.1),
            boxprops=dict(edgecolor="#333333", linewidth=0.75),
            whiskerprops=dict(color="#333333", linewidth=0.75),
            capprops=dict(color="#333333", linewidth=0.75),
        )

        # Unified Grid matching OoD ROC
        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", alpha=0.5, color="#D8D8D8", linewidth=0.5)
        ax.tick_params(axis="both", which="major", length=3, width=0.6)

        # X-ticks formatting and rotation
        ax.tick_params(axis="x", rotation=40, labelsize=6.5)
        for label in ax.get_xticklabels():
            label.set_ha("right")

        # Dynamic Y-axis limits with breathing room
        y_vals = df_plot["uniform_weight_mean"].dropna()
        if not y_vals.empty:
            y_max = y_vals.max()
            ax.set_ylim(-0.005, y_max * 1.08)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

        # Shared X-axis label
        ax.set_xlabel(r"Degradation Level", fontsize=7.2, fontweight="bold", labelpad=3)

        # Y-axis label only on leftmost column
        if col_idx == 0:
            ax.set_ylabel(r"Uniform Weight ($\pi_{\mathrm{uniform}}$)", fontsize=7.2, fontweight="bold", labelpad=3)
        else:
            ax.set_ylabel("")

    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.30, wspace=0.22)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"uniform_weight_1x3_{model_name}.pdf"
    png_path = out_dir / f"uniform_weight_1x3_{model_name}.png"

    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    logger.info("Saved 1x3 Uniform Weight Boxplot figure to:\n  - %s\n  - %s", pdf_path, png_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 1x3 publication boxplot figures for uniform weight distribution.")
    parser.add_argument("--model", type=str, default="hrnet_w32", help="Model backbone (default: hrnet_w32)")
    default_out_dir = (
        project_root / "outputs" / "figures" / "uncertainty" / "catastrophic_failures"
    )
    parser.add_argument("--out-dir", type=str, default=str(default_out_dir), help="Output directory")
    args = parser.parse_args()

    data_root = (
        project_root / "outputs" / "data"
        if (project_root / "outputs" / "data" / "Resultados_Incertidumbre").exists()
        else (
            project_root.parent / "Paper"
            if (project_root.parent / "Paper").exists()
            else project_root / "Paper"
        )
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["COCO", "CrowdPose", "OCHuman"]
    datasets_data = {}

    logger.info("==================================================")
    logger.info("Generating 1x3 Uniform Weight Boxplot for: %s", args.model.upper())
    logger.info("==================================================")

    for ds in datasets:
        df = load_raw_dataset(args.model, ds, data_root)
        datasets_data[ds] = df

    render_1x3_uniform_weight_boxplot(datasets_data, args.model, out_dir)

    logger.info("1x3 Uniform Weight Boxplot figure generated in: %s", out_dir)


if __name__ == "__main__":
    main()

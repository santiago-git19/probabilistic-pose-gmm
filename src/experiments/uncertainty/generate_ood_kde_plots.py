"""
Dedicated Script for Generating OoD Absence Detection KDE Distribution Plots.

Replicates the 2x2 KDE grid generation logic from evaluate_uncertainty.py
for HRNet-W32 across the 3 datasets (COCO, CrowdPose, OCHuman), with:
- Unified global external legend (removing the 4 individual subpanel legends).
- Large, highly legible typography for all titles, labels, ticks, and legends.

Layout per dataset (2x2 subgrid):
- Top-Left:     Density Distribution: Ours (Volume)
- Top-Right:    Density Distribution: Ours (Uniform)
- Bottom-Left:  Density Distribution: DARK (Entropy)
- Bottom-Right: Density Distribution: DARK (1 - P_DARK)

Output:
- 3 separate image files (one per dataset) in both PNG (300 DPI) and PDF vector format.

Usage:
    poetry run python src/experiments/uncertainty/generate_ood_kde_plots.py
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure paths
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.uncertainty.evaluate_uncertainty import _unroll_keypoints

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Configure visual styling
sns.set_theme(style="whitegrid")


def load_dataset_unrolled(model_name: str, dataset_name: str, paper_root: Path) -> pd.DataFrame:
    """Load and unroll keypoints for a given model and dataset."""
    subpath = f"{model_name}_{dataset_name.lower()}"
    parquet_path = paper_root / "Resultados_Incertidumbre" / subpath / "degradation_benchmark" / "all_degradations_combined.parquet"

    if not parquet_path.exists():
        log.warning(f"Parquet not found: {parquet_path}")
        return pd.DataFrame()

    df_raw = pd.read_parquet(parquet_path)
    df_unrolled = _unroll_keypoints(df_raw)
    log.info(f"Loaded {model_name} - {dataset_name}: {len(df_raw)} images -> {len(df_unrolled)} keypoints")
    return df_unrolled


def plot_single_dataset_kde(df: pd.DataFrame, output_dir: Path, model_name: str, dataset_name: str, suffix: str = "All"):
    """Qualitative OoD / Anomaly Detection Evaluation: Density Comparison (KDE).
    Generates a 2x2 grid with large typography and a single unified global legend outside the subfigures.
    """
    if df.empty or "vis" not in df.columns:
        log.warning(f"No valid data for {dataset_name}")
        return

    df_valid = df[df["vis"].notna()].copy()
    y_true = (df_valid["vis"] == 0).astype(int)

    if y_true.nunique() < 2 or y_true.sum() == 0 or (1 - y_true).sum() == 0:
        log.info(f"Insufficient samples of both classes for OoD KDE ({dataset_name}).")
        return

    # Categorize Presence
    df_valid["Presence"] = np.where(df_valid["vis"] == 0, "Absent / OoD (vis == 0)", "Present (vis > 0)")

    # Prepare transformed predictors
    if "cov_det" in df_valid.columns:
        df_valid["log_cov_det"] = np.log1p(np.maximum(df_valid["cov_det"], 0.0))
    if "base_score" in df_valid.columns:
        df_valid["inv_base_score"] = 1.0 - df_valid["base_score"]

    predictors = [
        ("log_cov_det", r"Ours (Volume)"),
        ("uniform_weight", r"Ours (Uniform)"),
        ("heatmap_entropy", "DARK (Entropy)"),
        ("inv_base_score", r"DARK ($1-P_{DARK}$)")
    ]

    # Palette
    palette = {
        "Present (vis > 0)": "#1f77b4",       # Blue for in-distribution
        "Absent / OoD (vis == 0)": "#d62728"  # Red for anomalies / OoD
    }

    # =========================================================================
    # Multi-panel 2x2 Grid with Large Typography & Unified Global Legend
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (col, label) in enumerate(predictors):
        ax = axes[i]
        if col not in df_valid.columns or df_valid[col].isna().all():
            ax.set_title(f"{label} (Not Available)", fontsize=17, fontweight="bold")
            continue

        df_plot = df_valid.dropna(subset=[col, "Presence"])
        if df_plot.empty:
            continue

        # Draw KDE with thicker line and legend=False
        try:
            sns.kdeplot(
                data=df_plot,
                x=col,
                hue="Presence",
                fill=True,
                common_norm=False,
                palette=palette,
                alpha=0.4,
                linewidth=2.8,
                ax=ax,
                legend=False
            )
        except Exception as e:
            log.debug(f"KDE failed for {col}, using fallback histplot: {e}")
            sns.histplot(
                data=df_plot,
                x=col,
                hue="Presence",
                stat="density",
                common_norm=False,
                palette=palette,
                alpha=0.4,
                ax=ax,
                legend=False
            )

        ax.set_xlabel(label, fontsize=16, fontweight="bold", labelpad=8)
        ax.set_ylabel("Density", fontsize=16, fontweight="bold", labelpad=8)
        ax.set_title(f"Density Distribution: {label}", fontsize=17, fontweight="bold", pad=8)
        ax.tick_params(axis="both", labelsize=15, length=4, width=1.0)

    title_suffix = f" - {dataset_name} ({suffix})" if suffix else f" - {dataset_name}"
    fig.suptitle(f"OoD Absence Detection (KDE Distributions){title_suffix}", fontsize=21, fontweight="bold", y=0.985)

    # Global Legend centered at the top banner with large text
    legend_elements = [
        mpatches.Patch(facecolor=palette["Present (vis > 0)"], edgecolor=palette["Present (vis > 0)"], alpha=0.6, label="Present (vis > 0)"),
        mpatches.Patch(facecolor=palette["Absent / OoD (vis == 0)"], edgecolor=palette["Absent / OoD (vis == 0)"], alpha=0.6, label="Absent / OoD (vis == 0)"),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        fontsize=16,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        framealpha=0.98,
        columnspacing=3.0,
        handlelength=2.0,
        handletextpad=0.6,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.86, bottom=0.08, hspace=0.30, wspace=0.22)

    out_file_png = output_dir / f"ood_absence_kde_{model_name}_{dataset_name.lower()}.png"
    out_file_pdf = output_dir / f"ood_absence_kde_{model_name}_{dataset_name.lower()}.pdf"

    plt.savefig(out_file_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
    plt.close()
    log.info(f"Saved {dataset_name} KDE plot to:\n  - {out_file_png}\n  - {out_file_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Generate 2x2 OoD Absence Detection KDE Plots with large typography and unified global legend.")
    parser.add_argument("--model", type=str, default="hrnet_w32", help="Model name (default: hrnet_w32)")
    default_out_dir = (
        _PROJECT_ROOT / "outputs" / "figures" / "uncertainty" / "ood_kde_individual"
    )
    parser.add_argument("--out-dir", type=str, default=str(default_out_dir), help="Output directory")
    args = parser.parse_args()

    data_root = (
        _PROJECT_ROOT / "outputs" / "data"
        if (_PROJECT_ROOT / "outputs" / "data" / "Resultados_Incertidumbre").exists()
        else (
            _PROJECT_ROOT.parent / "Paper"
            if (_PROJECT_ROOT.parent / "Paper").exists()
            else _PROJECT_ROOT / "Paper"
        )
    )
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["COCO", "CrowdPose", "OCHuman"]

    log.info("==================================================")
    log.info(f"Generating OoD KDE Plots for Model: {args.model.upper()}")
    log.info("==================================================")

    for ds in datasets:
        df = load_dataset_unrolled(args.model, ds, data_root)
        plot_single_dataset_kde(df, output_dir, args.model, ds, suffix="All")

    log.info(f"All 3 dataset KDE plots generated successfully in: {output_dir}")


if __name__ == "__main__":
    main()

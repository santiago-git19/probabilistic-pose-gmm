"""
Dedicated Script for Generating Expected Calibration Error (ECE) Plots.

Replicates the exact ECE calibration curve generation logic from evaluate_uncertainty.py
for HRNet-W32 across the 3 datasets (COCO, CrowdPose, OCHuman) and 3 topological regimes:
- Row 1: Global Topology (All Points)
- Row 2: Unimodal Regime (1 Gaussian / K = 1)
- Row 3: Bimodal Regime (2 Gaussians / K = 2)

Features:
- Exact binning via pd.qcut(cov_det, q=n_bins) and log-volume x-axis log(1 + mean_unc).
- Purple scatter points proportional to sample counts and connecting trend line.
- Large, legible publication typography (titles, axis labels, tick numbers).
- Generates both:
  1. A unified 3x3 multi-panel matrix figure.
  2. 3 individual dataset figures (COCO, CrowdPose, OCHuman) with 1x3 subpanels.

Usage:
    poetry run python src/experiments/uncertainty/generate_ece_plots.py --model hrnet_w32
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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


def compute_ece_curve(
    df: pd.DataFrame,
    k_filter: Optional[int] = None,
    n_bins: int = 10,
) -> Optional[pd.DataFrame]:
    """Compute ECE calibration bins matching evaluate_uncertainty.py."""
    if df.empty:
        return None

    if k_filter is not None and "n_components" in df.columns:
        df_sub = df[df["n_components"] == k_filter].copy()
    else:
        df_sub = df.copy()

    # Filter strictly valid points (vis > 0) with valid OKS and cov_det
    df_sub = df_sub[df_sub["vis"] > 0].dropna(subset=["oks_ours", "cov_det"]).copy()
    if len(df_sub) < n_bins:
        return None

    df_sub["error"] = 1.0 - df_sub["oks_ours"]
    df_sub["unc_bin"] = pd.qcut(df_sub["cov_det"], q=n_bins, labels=False, duplicates="drop")

    grouped = df_sub.groupby("unc_bin").agg(
        mean_unc=("cov_det", "mean"),
        mean_err=("error", "mean"),
        count=("error", "count")
    ).reset_index()

    grouped["log_unc"] = np.log1p(grouped["mean_unc"])
    return grouped


def plot_single_subplot_ece(
    ax: plt.Axes,
    grouped: Optional[pd.DataFrame],
    regime_title: str,
    sublabel: str = "",
    scatter_scale: float = 2.0,
    point_color: str = "purple",
    line_color: str = "purple",
) -> None:
    """Render a single ECE calibration curve on given Axes."""
    if grouped is None or grouped.empty:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=12)
        ax.set_title(f"Expected Calibration Error ({regime_title})", fontsize=14, fontweight="bold")
        return

    x_vals = grouped["log_unc"]
    y_vals = grouped["mean_err"]
    counts = grouped["count"]

    # Exact scatter and line matching evaluate_uncertainty.py
    scatter_sizes = counts / scatter_scale
    ax.scatter(x_vals, y_vals, s=scatter_sizes, alpha=0.6, color=point_color, zorder=4)
    ax.plot(x_vals, y_vals, color=line_color, linestyle="-", alpha=0.4, linewidth=2.5, zorder=3)

    # Dynamic limits with breathing room
    x_min, x_max = x_vals.min(), x_vals.max()
    x_span = max(x_max - x_min, 1.0)
    ax.set_xlim(x_min - 0.08 * x_span, x_max + 0.08 * x_span)

    y_min, y_max = y_vals.min(), y_vals.max()
    y_span = max(y_max - y_min, 0.05)
    ax.set_ylim(max(0.0, y_min - 0.08 * y_span), min(1.0, y_max + 0.08 * y_span))

    ax.set_xlabel("Mean Uncertainty (Volume)", fontsize=13, fontweight="bold", labelpad=6)
    ax.set_ylabel("Mean Error (1 - OKS)", fontsize=13, fontweight="bold", labelpad=6)
    ax.set_title(f"Expected Calibration Error ({regime_title})", fontsize=14, fontweight="bold", pad=8)
    ax.tick_params(axis="both", labelsize=12, length=4, width=0.8)

    if sublabel:
        ax.text(
            0.5, -0.22, sublabel,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="normal",
            ha="center", va="top",
            color="#000000",
        )


def generate_3x3_ece_matrix(
    datasets_data: Dict[str, pd.DataFrame],
    model_name: str,
    output_dir: Path,
) -> None:
    """Generate 3x3 ECE Calibration Matrix for the paper."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 16), facecolor="white")
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
            grouped = compute_ece_curve(df, k_filter=k_filter)
            sublabel = panel_sublabels[row_idx][col_idx]
            plot_single_subplot_ece(ax, grouped, regime_title, sublabel=sublabel)

    plt.tight_layout()
    plt.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.08, wspace=0.28, hspace=0.38)

    out_file_png = output_dir / f"ece_calibration_3x3_{model_name}.png"
    out_file_pdf = output_dir / f"ece_calibration_3x3_{model_name}.pdf"

    plt.savefig(out_file_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
    plt.close()
    log.info(f"Saved 3x3 ECE Matrix to:\n  - {out_file_png}\n  - {out_file_pdf}")


def generate_individual_dataset_ece_plots(
    datasets_data: Dict[str, pd.DataFrame],
    model_name: str,
    output_dir: Path,
) -> None:
    """Generate 1x3 ECE plots per dataset (All, 1 Gaussian, 2 Gaussians)."""
    regimes = [
        ("All", None, "(a) Global (All)"),
        ("1 Gaussian", 1, "(b) Unimodal ($K = 1$)"),
        ("2 Gaussians", 2, "(c) Bimodal ($K = 2$)"),
    ]

    for ds_name, df in datasets_data.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="white")
        fig.suptitle(f"Expected Calibration Error (ECE) - {ds_name} ({model_name.upper()})", fontsize=18, fontweight="bold", y=1.02)

        for idx, (regime_title, k_filter, sublabel) in enumerate(regimes):
            ax = axes[idx]
            grouped = compute_ece_curve(df, k_filter=k_filter)
            plot_single_subplot_ece(ax, grouped, regime_title, sublabel=sublabel, scatter_scale=1.5)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.18, wspace=0.25)

        out_file_png = output_dir / f"ece_calibration_1x3_{model_name}_{ds_name.lower()}.png"
        out_file_pdf = output_dir / f"ece_calibration_1x3_{model_name}_{ds_name.lower()}.pdf"

        plt.savefig(out_file_png, dpi=300, bbox_inches="tight")
        plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
        plt.close()
        log.info(f"Saved {ds_name} 1x3 ECE plot to:\n  - {out_file_png}\n  - {out_file_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Generate ECE Calibration Plots for HRNet-W32 across datasets and topological regimes.")
    parser.add_argument("--model", type=str, default="hrnet_w32", help="Model name (default: hrnet_w32)")
    default_out_dir = (
        _PROJECT_ROOT / "outputs" / "figures" / "uncertainty" / "ece_calibration_custom"
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
    datasets_data = {}

    log.info("==================================================")
    log.info(f"Generating ECE Calibration Plots for Model: {args.model.upper()}")
    log.info("==================================================")

    for ds in datasets:
        df = load_dataset_unrolled(args.model, ds, data_root)
        datasets_data[ds] = df

    # 1. Generate 3x3 Combined Matrix Figure
    generate_3x3_ece_matrix(datasets_data, args.model, output_dir)

    # 2. Generate Individual 1x3 Figures per Dataset
    generate_individual_dataset_ece_plots(datasets_data, args.model, output_dir)

    log.info(f"All ECE Calibration figures generated successfully in: {output_dir}")


if __name__ == "__main__":
    main()

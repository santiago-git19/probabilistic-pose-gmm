"""
Phase I: Baseline Precision vs Ours (GMM) Across Degradations.

Generates the 1x3 publication figure comparing Mean OKS across degradation levels
(Clean, Low, Medium, High, Extreme) for COCO, CrowdPose, and OCHuman.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "outputs" / "data" / "Resultados_Incertidumbre"
FALLBACK_DATA_ROOT = PROJECT_ROOT.parent / "Paper" / "Resultados_Incertidumbre"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

DATASETS = {
    "COCO": "hrnet_w32_coco",
    "CrowdPose": "hrnet_w32_crowdpose",
    "OCHuman": "hrnet_w32_ochuman",
}

DEGRADATIONS: List[Tuple[str, str, float]] = [
    ("00_Baseline_Clean", "Clean", np.nan),
    ("01_Resize_Low", "Low", 0.5),
    ("02_Resize_Medium", "Medium", 0.25),
    ("03_Resize_High", "High", 0.125),
    ("04_Resize_Extreme", "Extreme", 0.0625),
]


def resolve_data_folder(dataset_subpath: str) -> Path | None:
    """Find dataset folder in outputs/data or fallback."""
    for root in [DEFAULT_DATA_ROOT, FALLBACK_DATA_ROOT]:
        candidate = root / dataset_subpath
        if candidate.exists():
            return candidate
    return None


def generate_plot(output_dir: Path = OUTPUT_FIGURES_DIR) -> Path | None:
    """Generate and save the 1x3 Baseline vs Ours degradation precision plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Phase I: Baseline Precision vs Ours (GMM) Across Degradations", fontsize=16)

    x_labels = [d[1] for d in DEGRADATIONS]

    for idx, (ds_name, ds_suffix) in enumerate(DATASETS.items()):
        ax = axes[idx]
        folder = resolve_data_folder(ds_suffix)
        
        oks_base_vals = []
        oks_ours_vals = []

        if folder is not None:
            comb_path = folder / "degradation_benchmark" / "all_degradations_combined.parquet"
            if comb_path.exists():
                df_all = pd.read_parquet(comb_path)
                for full_name, short_name, scale_val in DEGRADATIONS:
                    if np.isnan(scale_val):
                        sub = df_all[df_all["dataset.resize_scale"].isna()]
                    else:
                        sub = df_all[df_all["dataset.resize_scale"] == scale_val]
                    
                    if not sub.empty and "oks_base" in sub.columns and "oks_ours" in sub.columns:
                        oks_base_vals.append(float(sub["oks_base"].mean()))
                        oks_ours_vals.append(float(sub["oks_ours"].mean()))
                    else:
                        oks_base_vals.append(None)
                        oks_ours_vals.append(None)
            else:
                # Try individual files
                for full_name, short_name, _ in DEGRADATIONS:
                    p = folder / "degradation_benchmark" / f"{full_name}_results.parquet"
                    if p.exists():
                        df = pd.read_parquet(p)
                        oks_base_vals.append(float(df["oks_base"].mean()))
                        oks_ours_vals.append(float(df["oks_ours"].mean()))
                    else:
                        oks_base_vals.append(None)
                        oks_ours_vals.append(None)
        else:
            oks_base_vals = [None] * len(DEGRADATIONS)
            oks_ours_vals = [None] * len(DEGRADATIONS)

        # Filter out missing
        x_plot, b_plot, o_plot = [], [], []
        for x, b, o in zip(x_labels, oks_base_vals, oks_ours_vals):
            if b is not None and o is not None:
                x_plot.append(x)
                b_plot.append(b)
                o_plot.append(o)

        if not x_plot:
            logger.warning("No data found for dataset %s", ds_name)
            continue

        # Plot with distinctive line styles
        ax.plot(
            x_plot,
            b_plot,
            marker="s",
            linewidth=2.5,
            markersize=8,
            label="DARK (Argmax)",
            color="#D95F02",
            linestyle="--",
            markeredgecolor="black",
            markeredgewidth=0.8,
            alpha=0.9,
        )
        ax.plot(
            x_plot,
            o_plot,
            marker="o",
            linewidth=2.8,
            markersize=8,
            label="Ours (GMM)",
            color="#1F77B4",
            linestyle=":",
            markeredgecolor="black",
            markeredgewidth=0.8,
            alpha=0.9,
        )

        ax.set_title(ds_name, fontsize=14, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Mean OKS (Higher is better)", fontsize=12)
        ax.set_xlabel("Degradation Level", fontsize=12)
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=12, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15)

    out_path_png = output_dir / "baseline_degradation_precision.png"
    out_path_pdf = output_dir / "baseline_degradation_precision.pdf"
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)

    logger.info("[SUCCESS] Saved Baseline Precision degradation plot to: %s", out_path_png)
    return out_path_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Baseline Precision vs Ours Degradation Plot.")
    parser.add_argument("--out-dir", type=str, default=str(OUTPUT_FIGURES_DIR), help="Output directory for generated plots")
    args = parser.parse_args()
    generate_plot(output_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()

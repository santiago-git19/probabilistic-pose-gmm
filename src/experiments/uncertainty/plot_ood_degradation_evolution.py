"""
Evolution of AUROC (OoD Anomaly Detection) vs Image Degradation.

Generates the 1x3 publication figures comparing Out-of-Distribution (OoD) keypoint
absence detection performance (DARK vs Ours Volume) across degradation levels
(Clean, Low, Medium, High, Extreme) for COCO, CrowdPose, and OCHuman.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "outputs" / "data" / "Resultados_Incertidumbre"
FALLBACK_DATA_ROOT = PROJECT_ROOT.parent / "Paper" / "Resultados_Incertidumbre"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

MODELS = ["hrnet_w32", "resnet50", "vitpose_small"]

DATASETS = {
    "COCO": "coco",
    "CrowdPose": "crowdpose",
    "OCHuman": "ochuman",
}

DEGRADATIONS: List[Tuple[str, str, float]] = [
    ("00_Baseline_Clean", "Clean", np.nan),
    ("01_Resize_Low", "Low", 0.5),
    ("02_Resize_Medium", "Medium", 0.25),
    ("03_Resize_High", "High", 0.125),
    ("04_Resize_Extreme", "Extreme", 0.0625),
]

COCO_KEYPOINT_NAMES: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def _unroll_keypoints(df: pd.DataFrame) -> pd.DataFrame:
    """Unroll image-level DataFrame to keypoint-level records for OoD evaluation."""
    rows = []
    for _, row in df.iterrows():
        for kp in COCO_KEYPOINT_NAMES:
            if f"vis_{kp}" in row:
                vis = row[f"vis_{kp}"]
                if pd.notna(vis) and vis >= 0:
                    cov_val = row.get(f"cov_det_{kp}", np.nan)
                    base_sc = row.get(f"base_score_{kp}", np.nan)
                    rows.append({
                        "vis": int(vis),
                        "cov_det": cov_val,
                        "base_score": base_sc,
                    })
    return pd.DataFrame(rows)


def resolve_model_dataset_path(model: str, dataset_key: str) -> Path | None:
    """Resolve directory path for a model-dataset combination."""
    subpath = f"{model}_{dataset_key}" if model == "hrnet_w32" else f"Modelos_Secundarios/{model}_{dataset_key}"
    for root in [DEFAULT_DATA_ROOT, FALLBACK_DATA_ROOT]:
        p = root / subpath
        if p.exists():
            return p
    return None


def compute_auroc_metrics(df_kps: pd.DataFrame) -> Tuple[float | None, float | None]:
    """Compute AUROC for Ours (cov_det) and DARK (1 - base_score) detecting absent keypoints (vis == 0)."""
    if df_kps.empty or "vis" not in df_kps.columns:
        return None, None

    y_true = (df_kps["vis"] == 0).astype(int)
    if y_true.nunique() < 2 or y_true.sum() == 0 or (1 - y_true).sum() == 0:
        return None, None

    # 1. Ours (Volume / cov_det)
    auroc_ours = None
    if "cov_det" in df_kps.columns:
        mask_cov = df_kps["cov_det"].notna()
        if mask_cov.sum() > 0 and y_true[mask_cov].nunique() >= 2:
            fpr, tpr, _ = roc_curve(y_true[mask_cov], df_kps.loc[mask_cov, "cov_det"])
            auroc_ours = float(auc(fpr, tpr))

    # 2. DARK (1 - P_DARK)
    auroc_dark = None
    if "base_score" in df_kps.columns:
        mask_base = df_kps["base_score"].notna()
        if mask_base.sum() > 0 and y_true[mask_base].nunique() >= 2:
            fpr, tpr, _ = roc_curve(y_true[mask_base], 1.0 - df_kps.loc[mask_base, "base_score"])
            auroc_dark = float(auc(fpr, tpr))

    return auroc_ours, auroc_dark


def generate_plot_for_model(model: str, output_dir: Path = OUTPUT_FIGURES_DIR) -> Path | None:
    """Generate 1x3 AUROC OoD vs Degradation plot for a specific model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Evolution of AUROC (OoD) vs Image Degradation", fontsize=20)

    x_labels = [d[1] for d in DEGRADATIONS]

    for idx, (ds_name, ds_suffix) in enumerate(DATASETS.items()):
        ax = axes[idx]
        folder = resolve_model_dataset_path(model, ds_suffix)

        gmm_vals = []
        base_vals = []

        if folder is not None:
            # Check for Metrics_JSON first if available
            metrics_dir = folder / "degradation_benchmark" / "graficas" / "Metrics_JSON"
            comb_path = folder / "degradation_benchmark" / "all_degradations_combined.parquet"

            if metrics_dir.exists():
                for full_name, short_name, _ in DEGRADATIONS:
                    json_p = metrics_dir / f"{full_name}_evaluation_metrics.json"
                    if json_p.exists():
                        with open(json_p, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        sub = data.get("subsets", {}).get("all", {})
                        gmm_vals.append(sub.get("auroc_cov_det", None))
                        base_vals.append(sub.get("auroc_inv_base_score", None))
                    else:
                        gmm_vals.append(None)
                        base_vals.append(None)
            elif comb_path.exists():
                # Compute directly from combined parquet
                df_all = pd.read_parquet(comb_path)
                for full_name, short_name, scale_val in DEGRADATIONS:
                    if np.isnan(scale_val):
                        sub_img = df_all[df_all["dataset.resize_scale"].isna()]
                    else:
                        sub_img = df_all[df_all["dataset.resize_scale"] == scale_val]

                    if not sub_img.empty:
                        df_kps = _unroll_keypoints(sub_img)
                        a_ours, a_dark = compute_auroc_metrics(df_kps)
                        gmm_vals.append(a_ours)
                        base_vals.append(a_dark)
                    else:
                        gmm_vals.append(None)
                        base_vals.append(None)
            else:
                gmm_vals = [None] * len(DEGRADATIONS)
                base_vals = [None] * len(DEGRADATIONS)
        else:
            gmm_vals = [None] * len(DEGRADATIONS)
            base_vals = [None] * len(DEGRADATIONS)

        # Filter out Nones
        x_plot, g_plot, b_plot = [], [], []
        for x, g, b in zip(x_labels, gmm_vals, base_vals):
            if g is not None and b is not None:
                x_plot.append(x)
                g_plot.append(g)
                b_plot.append(b)

        if not x_plot:
            logger.warning("No AUROC data found for %s on %s", model, ds_name)
            continue

        # Plot curves
        ax.plot(
            x_plot,
            b_plot,
            marker="s",
            linewidth=2.5,
            markersize=8,
            label=r"DARK ($1 - P_{DARK}$)",
            color="#D95F02",
            linestyle="--",
            markeredgecolor="black",
            markeredgewidth=0.8,
            alpha=0.9,
        )
        ax.plot(
            x_plot,
            g_plot,
            marker="o",
            linewidth=2.8,
            markersize=8,
            label="Ours (Volume)",
            color="#1F77B4",
            linestyle=":",
            markeredgecolor="black",
            markeredgewidth=0.8,
            alpha=0.9,
        )

        ax.set_title(ds_name, fontsize=14, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("AUROC (Higher is better)", fontsize=18)
        ax.set_xlabel("Degradation Level", fontsize=18)
        ax.set_ylim(0.35, 1.0)
        ax.tick_params(axis="x", labelsize=17)
        ax.tick_params(axis="y", labelsize=17)

        # Random guess line
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=18, bbox_to_anchor=(0.5, -0.10))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15)

    out_path_png = output_dir / f"ood_degradation_evolution_{model}.png"
    out_path_pdf = output_dir / f"ood_degradation_evolution_{model}.pdf"
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)

    logger.info("[SUCCESS] Saved OoD degradation evolution plot for %s to: %s", model, out_path_png)
    return out_path_png


def generate_all_plots(output_dir: Path = OUTPUT_FIGURES_DIR) -> List[Path]:
    """Generate OoD degradation evolution plots for all models."""
    generated = []
    for m in MODELS:
        p = generate_plot_for_model(m, output_dir=output_dir)
        if p is not None:
            generated.append(p)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OoD AUROC vs Degradation Evolution Plots.")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Models to generate plots for")
    parser.add_argument("--out-dir", type=str, default=str(OUTPUT_FIGURES_DIR), help="Output directory")
    args = parser.parse_args()

    out_d = Path(args.out_dir)
    for m in args.models:
        generate_plot_for_model(m, output_dir=out_d)


if __name__ == "__main__":
    main()

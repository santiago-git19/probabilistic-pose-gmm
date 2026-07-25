import argparse
import logging
import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Configurar estilo visual
sns.set_theme(style="whitegrid")
log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pose_uncertainty.evaluation.runner import COCO_KEYPOINT_NAMES

def _unroll_keypoints(df: pd.DataFrame) -> pd.DataFrame:
    """Desenrolla el DataFrame de nivel de imagen a nivel de keypoint."""
    rows = []
    for _, row in df.iterrows():
        for kp in COCO_KEYPOINT_NAMES:
            if f"vis_{kp}" in row:
                vis = row[f"vis_{kp}"]
                # Ignoramos los no etiquetados (vis == 0 o NaN)
                if pd.notna(vis) and vis > 0:
                    # Obtener o inferir el número de componentes (1 o 2 gaussianas)
                    if f"n_components_{kp}" in row and pd.notna(row[f"n_components_{kp}"]):
                        n_comp = int(row[f"n_components_{kp}"])
                    elif f"wasserstein_{kp}" in row and pd.notna(row[f"wasserstein_{kp}"]):
                        n_comp = 2 if row[f"wasserstein_{kp}"] > 0 else 1
                    elif f"kl_div_{kp}" in row and pd.notna(row[f"kl_div_{kp}"]):
                        n_comp = 2 if row[f"kl_div_{kp}"] > 0 else 1
                    else:
                        n_comp = np.nan

                    r = {
                        "keypoint": kp,
                        "vis": int(vis),
                        "n_components": n_comp,
                        "oks_ours": row.get(f"oks_ours_{kp}", np.nan),
                        "cov_det": row.get(f"cov_det_{kp}", np.nan),
                        "base_score": row.get(f"base_score_{kp}", np.nan),
                        "heatmap_entropy": row.get(f"heatmap_entropy_{kp}", np.nan),
                    }
                    rows.append(r)
    return pd.DataFrame(rows).dropna(subset=["oks_ours", "cov_det"])


def compute_ause(df: pd.DataFrame, error_col: str, uncertainty_col: str) -> float:
    """Calcula el Área bajo la Curva de Sparsification (AUSE)."""
    oracle_sorted = df.sort_values(by=error_col, ascending=False)
    model_sorted = df.sort_values(by=uncertainty_col, ascending=False)
    
    n_samples = len(df)
    fractions = np.linspace(0, 1, min(n_samples, 100)) # 100 puntos para la gráfica
    
    oracle_errors = oracle_sorted[error_col].values
    model_errors = model_sorted[error_col].values
    
    oracle_retained = [np.mean(oracle_errors[int(f * n_samples):]) if f < 1.0 else 0 
                       for f in fractions]
    model_retained = [np.mean(model_errors[int(f * n_samples):]) if f < 1.0 else 0 
                      for f in fractions]
                      
    ause = np.trapz(model_retained, fractions) - np.trapz(oracle_retained, fractions)
    return ause, fractions, oracle_retained, model_retained


def plot_sparsification(df: pd.DataFrame, output_dir: Path, suffix: str = ""):
    """Genera Sparsification Plot (AUSE) a nivel de keypoint."""
    if df.empty or "cov_det" not in df.columns:
        return

    df = df.copy()
    df["error"] = 1.0 - df["oks_ours"]
    
    ause, fractions, oracle, model = compute_ause(df, "error", "cov_det")
    
    plt.figure(figsize=(10, 8))
    plt.plot(fractions, model, label=f"Uncertainty (det(Sigma)) - AUSE: {ause:.4f}", color="blue", linewidth=2)
    plt.plot(fractions, oracle, label="Oracle (True Error)", color="black", linestyle="--", linewidth=2)
    plt.fill_between(fractions, oracle, model, color="blue", alpha=0.1)
    
    if "base_score" in df.columns and not df["base_score"].isna().all():
        df["inv_base_score"] = -df["base_score"]
        ause_base, _, _, model_base = compute_ause(df, "error", "inv_base_score")
        plt.plot(fractions, model_base, label=f"Baseline (Argmax Prob) - AUSE: {ause_base:.4f}", color="red", linestyle=":", linewidth=2)
        
    if "heatmap_entropy" in df.columns and not df["heatmap_entropy"].isna().all():
        ause_ent, _, _, model_ent = compute_ause(df, "error", "heatmap_entropy")
        plt.plot(fractions, model_ent, label=f"Baseline (Heatmap Entropy) - AUSE: {ause_ent:.4f}", color="green", linestyle="-.", linewidth=2)
    
    plt.xlabel("Fraction of keypoints removed")
    plt.ylabel("Mean Error (1 - OKS) on retained keypoints")
    title_suffix = suffix.replace("_", " ").title()
    plt.title(f"Sparsification Curve: GMM vs Baselines ({title_suffix.strip()})")
    plt.legend()
    
    plt.savefig(output_dir / f"sparsification_curve{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_ece(df: pd.DataFrame, output_dir: Path, n_bins: int = 10, suffix: str = ""):
    """Calcula y dibuja Expected Calibration Error."""
    if df.empty or "cov_det" not in df.columns:
        return
        
    df = df.copy()
    df["error"] = 1.0 - df["oks_ours"]
    
    df["unc_bin"] = pd.qcut(df["cov_det"], q=n_bins, labels=False, duplicates="drop")
    
    grouped = df.groupby("unc_bin").agg(
        mean_unc=("cov_det", "mean"),
        mean_err=("error", "mean"),
        count=("error", "count")
    ).reset_index()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(grouped["mean_unc"], grouped["mean_err"], s=grouped["count"], alpha=0.6, color="purple")
    plt.plot(grouped["mean_unc"], grouped["mean_err"], color="purple", linestyle="-", alpha=0.4)
    
    plt.xlabel("Mean Uncertainty (Volume)")
    plt.ylabel("Mean Error (1 - OKS)")
    title_suffix = suffix.replace("_", " ").title()
    plt.title(f"Reliability Diagram ({title_suffix.strip()})")
    
    plt.savefig(output_dir / f"ece_calibration{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_limb_swap_roc(df: pd.DataFrame, output_dir: Path):
    """Dibuja curva ROC para la detección de limb swaps usando Wasserstein y KL a nivel de imagen."""
    swaps_true = []
    wasserstein_scores = []
    kl_scores = []
    
    for _, row in df.iterrows():
        if "swaps_ours_arr" not in row or not isinstance(row["swaps_ours_arr"], (list, np.ndarray)):
            continue
            
        swaps_arr = row["swaps_ours_arr"]
        is_swap = any(swaps_arr)
        if "wasserstein_mean" in row and "kl_div_mean" in row:
            swaps_true.append(int(is_swap))
            wasserstein_scores.append(row["wasserstein_mean"])
            kl_scores.append(row["kl_div_mean"])

    if len(swaps_true) == 0 or sum(swaps_true) == 0:
        log.warning("No hay suficientes datos de limb swaps para graficar ROC.")
        return

    fpr_w, tpr_w, _ = roc_curve(swaps_true, wasserstein_scores)
    roc_auc_w = auc(fpr_w, tpr_w)
    
    fpr_kl, tpr_kl, _ = roc_curve(swaps_true, kl_scores)
    roc_auc_kl = auc(fpr_kl, tpr_kl)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_w, tpr_w, color="darkorange", lw=2, label=f"Wasserstein (AUC = {roc_auc_w:.3f})")
    plt.plot(fpr_kl, tpr_kl, color="navy", lw=2, label=f"KL Div (AUC = {roc_auc_kl:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Limb Swap Detection ROC")
    plt.legend(loc="lower right")
    plt.savefig(output_dir / "limb_swaps_roc.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_catastrophic_failures(df: pd.DataFrame, output_dir: Path):
    """Grafica la media del peso uniforme respecto al nivel de degradación."""
    if "uniform_weight_mean" not in df.columns or "experiment_name" not in df.columns:
        return
        
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="experiment_name", y="uniform_weight_mean", data=df, palette="Reds")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Degradation Level")
    plt.ylabel(r"Uniform Weight ($\pi_{uniforme}$)")
    plt.title("Catastrophic Failures vs Degradation")
    plt.tight_layout()
    plt.savefig(output_dir / "catastrophic_failures_boxplot.png", dpi=300)
    plt.close()


def generate_all_plots(df: pd.DataFrame, output_dir: Union[str, Path]):
    """Genera todas las gráficas analíticas y las guarda en output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("Desenrollando DataFrame por keypoints...")
    df_kp = _unroll_keypoints(df)
    
    df_vis = df_kp[df_kp["vis"] == 2]
    df_occ = df_kp[df_kp["vis"] == 1]
    
    log.info("Generando Sparsification Plots y ECE (por visibilidad)...")
    if not df_vis.empty:
        plot_sparsification(df_vis, output_dir, suffix="_visible")
        plot_ece(df_vis, output_dir, suffix="_visible")
        
    if not df_occ.empty:
        plot_sparsification(df_occ, output_dir, suffix="_occluded")
        plot_ece(df_occ, output_dir, suffix="_occluded")
        
    if not df_kp.empty:
        plot_sparsification(df_kp, output_dir, suffix="_all")
        plot_ece(df_kp, output_dir, suffix="_all")
        
    df_1comp = df_kp[df_kp["n_components"] == 1]
    df_2comp = df_kp[df_kp["n_components"] == 2]
    
    log.info("Generando Sparsification Plots y ECE (por número de gaussianas)...")
    if not df_1comp.empty:
        plot_sparsification(df_1comp, output_dir, suffix="_1_gaussian")
        plot_ece(df_1comp, output_dir, suffix="_1_gaussian")
        
    if not df_2comp.empty:
        plot_sparsification(df_2comp, output_dir, suffix="_2_gaussians")
        plot_ece(df_2comp, output_dir, suffix="_2_gaussians")
    
    log.info("Generando Curvas ROC de Limb Swaps (global)...")
    plot_limb_swap_roc(df, output_dir)
    
    log.info("Generando Análisis de Fallos Catastróficos (global)...")
    plot_catastrophic_failures(df, output_dir)


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Evaluar incertidumbre desde archivos parquet.")
    parser.add_argument("input_path", type=str, help="Ruta al archivo parquet o directorio.")
    parser.add_argument("--out", type=str, default=".", help="Directorio de salida para gráficas.")
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    if input_path.is_dir():
        parquets = list(input_path.glob("*.parquet"))
        if not parquets:
            log.error("No se encontraron archivos parquet en el directorio.")
            sys.exit(1)
        df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    else:
        df = pd.read_parquet(input_path)
        
    logging.basicConfig(level=logging.INFO)
    generate_all_plots(df, args.out)
    log.info(f"Análisis completado. Gráficas guardadas en {args.out}")

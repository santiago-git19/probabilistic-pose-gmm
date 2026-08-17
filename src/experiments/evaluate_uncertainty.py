import json
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

# Configure visual styling
sns.set_theme(style="whitegrid")
log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pose_uncertainty.evaluation.runner import COCO_KEYPOINT_NAMES

def _unroll_keypoints(df: pd.DataFrame) -> pd.DataFrame:
    """Unroll image-level DataFrame to keypoint-level records."""
    rows = []
    for _, row in df.iterrows():
        for kp in COCO_KEYPOINT_NAMES:
            if f"vis_{kp}" in row:
                vis = row[f"vis_{kp}"]
                # Include absent joints (vis == 0) for OoD evaluation, discarding only NaNs
                if pd.notna(vis) and vis >= 0:
                    # Retrieve or infer component count (1 or 2 Gaussians)
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
                        "oks_base": row.get(f"oks_base_{kp}", np.nan),
                        "cov_det": row.get(f"cov_det_{kp}", np.nan),
                        "uniform_weight": row.get(f"uniform_weight_{kp}", np.nan),
                        "base_score": row.get(f"base_score_{kp}", np.nan),
                        "heatmap_entropy": row.get(f"heatmap_entropy_{kp}", np.nan),
                    }
                    rows.append(r)
    return pd.DataFrame(rows).dropna(subset=["cov_det"])


def compute_ause(df: pd.DataFrame, error_col: str, uncertainty_col: str) -> float:
    """Compute Area Under Sparsification Error curve (AUSE)."""
    oracle_sorted = df.sort_values(by=error_col, ascending=False)
    model_sorted = df.sort_values(by=uncertainty_col, ascending=False)
    
    n_samples = len(df)
    fractions = np.linspace(0, 1, min(n_samples, 100)) # 100 evaluation evaluation fractions
    
    oracle_errors = oracle_sorted[error_col].values
    model_errors = model_sorted[error_col].values
    
    oracle_retained = [np.mean(oracle_errors[int(f * n_samples):]) if f < 1.0 else 0 
                       for f in fractions]
    model_retained = [np.mean(model_errors[int(f * n_samples):]) if f < 1.0 else 0 
                      for f in fractions]
                      
    ause = np.trapz(model_retained, fractions) - np.trapz(oracle_retained, fractions)
    return ause, fractions, oracle_retained, model_retained


def plot_sparsification(
    df: pd.DataFrame, 
    output_dir: Path, 
    suffix: str = "",
    beta: float = 0.1,
    strategy: str = "max_pooling",
    tau: float = 0.5
):
    """Generate Sparsification Plot (AUSE) at keypoint level with Adaptive Uncertainty Fusion."""
    if df.empty or "cov_det" not in df.columns or "vis" not in df.columns:
        return {}

    # Filter strictly valid points (vis > 0) for geometric OKS/AUSE evaluation
    df = df[df["vis"] > 0].dropna(subset=["oks_ours", "cov_det"]).copy()
    if df.empty:
        return {}

    # Compute true error for our model and baseline
    df["error_ours"] = 1.0 - df["oks_ours"]
    if "oks_base" in df.columns:
        df["error_base"] = 1.0 - df["oks_base"]
    else:
        df["error_base"] = df["error_ours"]
    
    # 1. Evaluate volumetric uncertainty of GMM against error of our model
    ause, fractions, oracle, model = compute_ause(df, "error_ours", "cov_det")
    ause_mp = None
    ause_gate = None
    ause_softmax = None
    ause_base = None
    ause_ent = None
    
    plt.figure(figsize=(10, 8))
    plt.plot(fractions, model, label=f"Ours (Volume) - AUSE: {ause:.4f}", color="blue", linewidth=2)
    plt.plot(fractions, oracle, label="Oracle (True Error)", color="black", linestyle="--", linewidth=2)
    plt.fill_between(fractions, oracle, model, color="blue", alpha=0.1)
    
    # Adaptive Fusion: Evaluate against our error
    if "base_score" in df.columns and not df["base_score"].isna().all():
        u_base = 1.0 - df["base_score"]
        u_gmm = 1.0 - np.exp(-beta * df["cov_det"])
        
        # Strategy A: Max-Pooling
        df["u_adapt_mp"] = np.maximum(u_base, u_gmm)
        ause_mp, _, _, model_mp = compute_ause(df, "error_ours", "u_adapt_mp")
        plt.plot(fractions, model_mp, label=f"Ours (Max-Pool) - AUSE: {ause_mp:.4f}", color="magenta", linewidth=2)
        
        # Strategy B: Topological Gating
        n_comp = df["n_components"] if "n_components" in df.columns else pd.Series(1, index=df.index)
        if strategy == "gating_tau":
            cond = (n_comp == 2) | (u_gmm > tau)
        else:
            cond = (n_comp == 2)
        df["u_adapt_gate"] = np.where(cond, u_gmm, u_base)
        ause_gate, _, _, model_gate = compute_ause(df, "error_ours", "u_adapt_gate")
        plt.plot(fractions, model_gate, label=f"Ours (Gating K=2) - AUSE: {ause_gate:.4f}", color="cyan", linewidth=2)
        
        # Strategy C: Weighted Softmax
        exp_u_base = np.exp(u_base)
        exp_u_gmm = np.exp(u_gmm)
        sum_exp = exp_u_base + exp_u_gmm
        w_base = exp_u_base / sum_exp
        w_gmm = exp_u_gmm / sum_exp
        df["u_adapt_softmax"] = w_base * u_base + w_gmm * u_gmm
        ause_softmax, _, _, model_softmax = compute_ause(df, "error_ours", "u_adapt_softmax")
        plt.plot(fractions, model_softmax, label=f"Ours (Softmax) - AUSE: {ause_softmax:.4f}", color="orange", linewidth=2)
    
    # Evaluate Baselines: Evaluate uncertainty against baseline error (error_base)
    if "base_score" in df.columns and not df["base_score"].isna().all():
        df["inv_base_score"] = -df["base_score"]
        ause_base, _, _, model_base = compute_ause(df, "error_base", "inv_base_score")
        plt.plot(fractions, model_base, label=f"DARK ($1-P_{{DARK}}$) - AUSE: {ause_base:.4f}", color="red", linestyle=":", linewidth=2)
        
    if "heatmap_entropy" in df.columns and not df["heatmap_entropy"].isna().all():
        ause_ent, _, _, model_ent = compute_ause(df, "error_base", "heatmap_entropy")
        plt.plot(fractions, model_ent, label=f"DARK (Entropy) - AUSE: {ause_ent:.4f}", color="green", linestyle="-.", linewidth=2)
    
    plt.xlabel("Fraction of keypoints removed")
    plt.ylabel("Mean Error (1 - OKS) on retained keypoints")
    title_suffix = suffix.replace("_", " ").title()
    plt.title(f"Sparsification Curve: GMM vs Baselines ({title_suffix.strip()})")
    plt.legend()
    
    plt.savefig(output_dir / f"sparsification_curve{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    res = {
        "ause_gmm_raw": float(ause),
        "ause_adaptive_max_pooling": float(ause_mp) if ause_mp is not None else None,
        "ause_adaptive_gating": float(ause_gate) if ause_gate is not None else None,
        "ause_adaptive_softmax": float(ause_softmax) if ause_softmax is not None else None,
        "ause_baseline_argmax": float(ause_base) if ause_base is not None else None,
        "ause_baseline_entropy": float(ause_ent) if ause_ent is not None else None,
    }
    return {k: v for k, v in res.items() if v is not None}


def plot_ece(df: pd.DataFrame, output_dir: Path, n_bins: int = 10, suffix: str = ""):
    """Compute and plot Expected Calibration Error."""
    if df.empty or "cov_det" not in df.columns or "vis" not in df.columns:
        return {}
        
    # Filter strictly valid points (vis > 0) for geometric OKS/ECE evaluation
    df = df[df["vis"] > 0].dropna(subset=["oks_ours", "cov_det"]).copy()
    if df.empty:
        return {}
        
    df["error"] = 1.0 - df["oks_ours"]
    
    df["unc_bin"] = pd.qcut(df["cov_det"], q=n_bins, labels=False, duplicates="drop")
    
    grouped = df.groupby("unc_bin").agg(
        mean_unc=("cov_det", "mean"),
        mean_err=("error", "mean"),
        count=("error", "count")
    ).reset_index()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(np.log1p(grouped["mean_unc"]), grouped["mean_err"], s=grouped["count"], alpha=0.6, color="purple")
    plt.plot(np.log1p(grouped["mean_unc"]), grouped["mean_err"], color="purple", linestyle="-", alpha=0.4)
    
    plt.xlabel("Mean Uncertainty (Volume)")
    plt.ylabel("Mean Error (1 - OKS)")
    title_suffix = suffix.replace("_", " ").title()
    plt.title(f"Expected Calibration Error ({title_suffix.strip()})")
    plt.tight_layout()
    plt.savefig(output_dir / f"ece_calibration{suffix}.png", dpi=300)
    plt.close()
    
    max_unc = grouped["mean_unc"].max()
    norm_unc = grouped["mean_unc"] / max_unc if max_unc > 0 else grouped["mean_unc"]
    ece_val = float(np.average(np.abs(grouped["mean_err"] - norm_unc), weights=grouped["count"]))
    spearman_corr = float(df["cov_det"].corr(df["error"], method="spearman")) if len(df) > 1 else np.nan
    return {
        "ece_value": ece_val,
        "spearman_correlation": spearman_corr
    }


def plot_limb_swap_roc(df: pd.DataFrame, output_dir: Path):
    """Compute ROC AUC for detecting topological limb swaps and severe ambiguities."""
    if df.empty or "is_swapped" not in df.columns or "cov_det" not in df.columns:
        return {}
        
    y_true = df["is_swapped"].astype(int)
    if y_true.nunique() < 2:
        return {}
        
    y_scores = df["cov_det"]
    
    # Empirical ROC calculation
    thresholds = np.percentile(y_scores, np.linspace(0, 100, 100))
    tpr, fpr = [], []
    
    pos_count = (y_true == 1).sum()
    neg_count = (y_true == 0).sum()
    
    for t in thresholds:
        tp = ((y_scores >= t) & (y_true == 1)).sum()
        fp = ((y_scores >= t) & (y_true == 0)).sum()
        tpr.append(tp / pos_count if pos_count > 0 else 0)
        fpr.append(fp / neg_count if neg_count > 0 else 0)
        
    sorted_indices = np.argsort(fpr)
    fpr = np.array(fpr)[sorted_indices]
    tpr = np.array(tpr)[sorted_indices]
    
    auc_val = np.trapz(tpr, fpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f"Covariance Volume - AUC: {auc_val:.4f}", color="darkorange", linewidth=2)
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC: Detection of Severe Topological Ambiguity (Limb Swaps)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "limb_swap_roc.png", dpi=300)
    plt.close()
    return {"limb_swap_auroc": float(auc_val)}


def plot_catastrophic_failures(df: pd.DataFrame, output_dir: Path):
    """Analyze and plot rate and weight of catastrophic failures vs degradation."""
    if df.empty or "uniform_weight_mean" not in df.columns or "experiment_name" not in df.columns:
        return {}
        
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="experiment_name", y="uniform_weight_mean", data=df, palette="Reds")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Degradation Level")
    plt.ylabel(r"Uniform Weight ($\pi_{uniform}$)")
    plt.title("Catastrophic Failures vs Degradation")
    plt.tight_layout()
    plt.savefig(output_dir / "catastrophic_failures_boxplot.png", dpi=300)
    plt.close()
    
    stats = {}
    for exp_name, sub in df.groupby("experiment_name"):
        stats[str(exp_name)] = {
            "mean": float(sub["uniform_weight_mean"].mean()),
            "median": float(sub["uniform_weight_mean"].median()),
            "std": float(sub["uniform_weight_mean"].std()) if len(sub) > 1 else 0.0,
            "q3": float(sub["uniform_weight_mean"].quantile(0.75))
        }
    return {"catastrophic_failures_by_degradation": stats}


def plot_ood_absence_roc(df: pd.DataFrame, output_dir: Path, suffix: str = ""):
    """Quantitative OoD / Anomaly Detection Evaluation: AUROC of Keypoint Absence.
    Binary classification: vis == 0 (anomaly, positive=1) vs vis > 0 (normal, negative=0).
    """
    if df.empty or "vis" not in df.columns:
        return {}
        
    df_valid = df[df["vis"].notna()].copy()
    y_true = (df_valid["vis"] == 0).astype(int)
    
    if y_true.nunique() < 2 or y_true.sum() == 0 or (1 - y_true).sum() == 0:
        log.info(f"Insufficient samples of both classes (vis == 0 and vis > 0) for OoD ROC{suffix}.")
        return {}

    predictors = [
        ("cov_det", r"Ours (Volume)", "blue", "auroc_cov_det"),
        ("uniform_weight", r"Ours (Uniform)", "darkorange", "auroc_uniform_weight"),
        ("heatmap_entropy", "DARK (Entropy)", "green", "auroc_heatmap_entropy"),
        ("inv_base_score", r"DARK ($1-P_{DARK}$)", "red", "auroc_inv_base_score")
    ]
    
    plt.figure(figsize=(8, 8))
    res = {
        "n_absent_vis0": int(y_true.sum()),
        "n_present_vis_gt0": int((1 - y_true).sum())
    }
    
    for col, label, color, metric_key in predictors:
        if col == "inv_base_score":
            if "base_score" in df_valid.columns:
                scores = 1.0 - df_valid["base_score"]
            else:
                continue
        elif col in df_valid.columns:
            scores = df_valid[col]
        else:
            continue
            
        mask = ~scores.isna()
        if mask.sum() == 0 or y_true[mask].nunique() < 2:
            continue
            
        fpr, tpr, _ = roc_curve(y_true[mask], scores[mask])
        roc_auc = auc(fpr, tpr)
        res[metric_key] = float(roc_auc)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.4f})", color=color, linewidth=2)
        
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random Guess (AUC = 0.5000)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    title_suffix = suffix.replace("_", " ").title()
    plt.title(f"OoD Absence Detection ROC{title_suffix.strip()}: vis == 0 vs vis > 0")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / f"ood_absence_roc{suffix}.png", dpi=300)
    plt.close()
    return res


def plot_ood_absence_kde(df: pd.DataFrame, output_dir: Path, suffix: str = ""):
    """Qualitative OoD / Anomaly Detection Evaluation: Density Comparison (KDE).
    Compares uncertainty distributions for present (vis > 0) vs absent/occluded (vis == 0) joints.
    """
    if df.empty or "vis" not in df.columns:
        return
        
    df_valid = df[df["vis"].notna()].copy()
    y_true = (df_valid["vis"] == 0).astype(int)
    
    if y_true.nunique() < 2 or y_true.sum() == 0 or (1 - y_true).sum() == 0:
        log.info(f"Insufficient samples of both classes for OoD KDE{suffix}.")
        return

    df_valid["Presence"] = np.where(df_valid["vis"] == 0, "Absent / OoD (vis == 0)", "Present (vis > 0)")
    
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
    
    palette = {
        "Present (vis > 0)": "#1f77b4",       # Blue for in-distribution
        "Absent / OoD (vis == 0)": "#d62728"  # Red for anomalies / OoD
    }
    
    # 1. Multi-panel 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (col, label) in enumerate(predictors):
        ax = axes[i]
        if col not in df_valid.columns or df_valid[col].isna().all():
            ax.set_title(f"{label} (Not Available)")
            continue
            
        df_plot = df_valid.dropna(subset=[col, "Presence"])
        if df_plot.empty:
            continue
            
        try:
            sns.kdeplot(
                data=df_plot, 
                x=col, 
                hue="Presence", 
                fill=True, 
                common_norm=False, 
                palette=palette, 
                alpha=0.4, 
                linewidth=2, 
                ax=ax
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
                ax=ax
            )
            
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(f"Density Distribution: {label}")
        
    title_suffix = suffix.replace("_", " ").title()
    fig.suptitle(f"OoD Absence Detection (KDE Distributions){title_suffix.strip()}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"ood_absence_kde_grid{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Save individual high-resolution figures
    for col, label in predictors:
        if col not in df_valid.columns or df_valid[col].isna().all():
            continue
        df_plot = df_valid.dropna(subset=[col, "Presence"])
        if df_plot.empty:
            continue
            
        plt.figure(figsize=(8, 6))
        try:
            sns.kdeplot(
                data=df_plot, 
                x=col, 
                hue="Presence", 
                fill=True, 
                common_norm=False, 
                palette=palette, 
                alpha=0.4, 
                linewidth=2
            )
        except Exception:
            sns.histplot(
                data=df_plot, 
                x=col, 
                hue="Presence", 
                stat="density", 
                common_norm=False, 
                palette=palette, 
                alpha=0.4
            )
        plt.xlabel(label)
        plt.ylabel("Density")
        plt.title(f"OoD Density Comparison: {label}{title_suffix.strip()}")
        plt.tight_layout()
        plt.savefig(output_dir / f"ood_absence_kde_{col}{suffix}.png", dpi=300)
        plt.close()


def generate_all_plots(
    df: pd.DataFrame, 
    output_dir: Union[str, Path],
    beta: float = 0.1,
    strategy: str = "max_pooling",
    tau: float = 0.5
):
    """Generate all analytical evaluation plots and save to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_summary = {
        "subsets": {},
        "global_topological_and_failures": {}
    }
    
    log.info("Unrolling DataFrame by keypoints...")
    df_kp = _unroll_keypoints(df)
    
    def _get_subset_dict(name: str):
        if name not in metrics_summary["subsets"]:
            metrics_summary["subsets"][name] = {}
        return metrics_summary["subsets"][name]
    
    # OoD Anomaly Detection Analysis for vis == 0
    log.info("Generating OoD Absence Detection Analysis (ROC and KDE)...")
    _get_subset_dict("all").update(plot_ood_absence_roc(df_kp, output_dir, suffix="_all"))
    plot_ood_absence_kde(df_kp, output_dir, suffix="_all")
    
    df_vis = df_kp[df_kp["vis"] == 2]
    df_occ = df_kp[df_kp["vis"] == 1]
    
    log.info("Generating Sparsification Plots and ECE (by visibility)...")
    if not df_vis.empty:
        _get_subset_dict("visible").update(plot_sparsification(df_vis, output_dir, suffix="_visible", beta=beta, strategy=strategy, tau=tau))
        _get_subset_dict("visible").update(plot_ece(df_vis, output_dir, suffix="_visible"))
        
    if not df_occ.empty:
        _get_subset_dict("occluded").update(plot_sparsification(df_occ, output_dir, suffix="_occluded", beta=beta, strategy=strategy, tau=tau))
        _get_subset_dict("occluded").update(plot_ece(df_occ, output_dir, suffix="_occluded"))
        
    if not df_kp.empty:
        _get_subset_dict("all").update(plot_sparsification(df_kp, output_dir, suffix="_all", beta=beta, strategy=strategy, tau=tau))
        _get_subset_dict("all").update(plot_ece(df_kp, output_dir, suffix="_all"))
        
    df_1comp = df_kp[df_kp["n_components"] == 1]
    df_2comp = df_kp[df_kp["n_components"] == 2]
    
    log.info("Generating Sparsification Plots and ECE (by component count)...")
    if not df_1comp.empty:
        _get_subset_dict("1_gaussian").update(plot_sparsification(df_1comp, output_dir, suffix="_1_gaussian", beta=beta, strategy=strategy, tau=tau))
        _get_subset_dict("1_gaussian").update(plot_ece(df_1comp, output_dir, suffix="_1_gaussian"))
        _get_subset_dict("1_gaussian").update(plot_ood_absence_roc(df_1comp, output_dir, suffix="_1_gaussian"))
        plot_ood_absence_kde(df_1comp, output_dir, suffix="_1_gaussian")
        
    if not df_2comp.empty:
        _get_subset_dict("2_gaussians").update(plot_sparsification(df_2comp, output_dir, suffix="_2_gaussians", beta=beta, strategy=strategy, tau=tau))
        _get_subset_dict("2_gaussians").update(plot_ece(df_2comp, output_dir, suffix="_2_gaussians"))
        _get_subset_dict("2_gaussians").update(plot_ood_absence_roc(df_2comp, output_dir, suffix="_2_gaussians"))
        plot_ood_absence_kde(df_2comp, output_dir, suffix="_2_gaussians")
    
    log.info("Generating Limb Swap ROC curves (global)...")
    metrics_summary["global_topological_and_failures"].update(plot_limb_swap_roc(df, output_dir))
    
    log.info("Generating Catastrophic Failures Analysis (global)...")
    metrics_summary["global_topological_and_failures"].update(plot_catastrophic_failures(df, output_dir))
    
    # Save structured JSON
    json_path = output_dir / "evaluation_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
        
    # Flatten and save tabular CSV
    csv_rows = []
    for subset_name, sub_metrics in metrics_summary.get("subsets", {}).items():
        for metric_name, val in sub_metrics.items():
            if isinstance(val, (int, float)):
                csv_rows.append({"category": "subset", "subset_or_exp": subset_name, "metric": metric_name, "value": val})
    for metric_name, val in metrics_summary.get("global_topological_and_failures", {}).items():
        if isinstance(val, (int, float)):
            csv_rows.append({"category": "global", "subset_or_exp": "global", "metric": metric_name, "value": val})
        elif isinstance(val, dict):
            for exp_name, stats_dict in val.items():
                for stat_k, stat_v in stats_dict.items():
                    csv_rows.append({"category": "degradation_failure", "subset_or_exp": exp_name, "metric": f"uniform_weight_{stat_k}", "value": stat_v})
                    
    if csv_rows:
        df_csv = pd.DataFrame(csv_rows)
        csv_path = output_dir / "evaluation_metrics.csv"
        df_csv.to_csv(csv_path, index=False, encoding="utf-8")
        log.info(f"Numeric metrics saved to {json_path} and {csv_path}")
        
    return metrics_summary


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Evaluate uncertainty from parquet results.")
    parser.add_argument("input_path", type=str, help="Path to parquet file or results directory.")
    parser.add_argument("--out", type=str, default=".", help="Output directory for plots.")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta hyperparameter for adaptive uncertainty.")
    parser.add_argument("--strategy", type=str, default="max_pooling", help="Adaptive strategy (max_pooling, gating_k, etc.)")
    parser.add_argument("--tau", type=float, default=0.5, help="Threshold tau for gating_tau strategy.")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    input_path = Path(args.input_path)
    
    from src.experiments.degradation_organizer import run_organized_evaluation_pipeline
    if run_organized_evaluation_pipeline(input_path, out_arg=args.out, beta=args.beta, strategy=args.strategy, tau=args.tau):
        sys.exit(0)
        
    if input_path.is_dir():
        parquets = list(input_path.glob("*.parquet"))
        if not parquets:
            log.error("No parquet files found in directory.")
            sys.exit(1)
        df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    else:
        df = pd.read_parquet(input_path)
        
    generate_all_plots(df, args.out, beta=args.beta, strategy=args.strategy, tau=args.tau)
    log.info(f"Analysis complete. Plots saved in {args.out}")

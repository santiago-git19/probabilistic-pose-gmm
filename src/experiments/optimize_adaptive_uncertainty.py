import json
import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visual styling
sns.set_theme(style="whitegrid")
log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.evaluate_uncertainty import _unroll_keypoints


def compute_ause_fast(error: np.ndarray, uncertainty: np.ndarray) -> float:
    """Optimized and vectorized computation of AUSE."""
    n = len(error)
    if n == 0:
        return 0.0
        
    fractions = np.linspace(0, 1, min(n, 100))
    
    # Sort by error (Oracle) and by uncertainty (Model)
    oracle_errors = error[np.argsort(-error)]
    model_errors = error[np.argsort(-uncertainty)]
    
    # Compute inverse cumulative sums for fast mean computation
    oracle_cum_sum = np.cumsum(oracle_errors[::-1])[::-1]
    model_cum_sum = np.cumsum(model_errors[::-1])[::-1]
    
    # Avoid zero division
    counts = np.arange(n, 0, -1)
    
    oracle_means = oracle_cum_sum / counts
    model_means = model_cum_sum / counts
    
    # Map to fractions 0-1
    indices = np.clip((fractions * n).astype(int), 0, n - 1)
    
    ause = np.trapz(model_means[indices], fractions) - np.trapz(oracle_means[indices], fractions)
    return float(ause)


def optimize_strategies(
    df: pd.DataFrame, 
    beta_candidates: np.ndarray,
    tau_candidates: np.ndarray = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
) -> Dict[str, Any]:
    """Perform grid search for Strategy A (Max-Pooling) and Strategy B (Gating)."""
    error = (1.0 - df["oks_ours"]).values
    error_base = (1.0 - df["oks_base"]).values if "oks_base" in df.columns else error
    
    # Inverse argmax confidence: 1 - P_argmax
    u_base = (1.0 - df["base_score"]).values if "base_score" in df.columns else np.zeros_like(error)
    det_sigma = df["cov_det"].values
    n_comp = df["n_components"].values if "n_components" in df.columns else np.ones_like(error)
    
    # 1. Evaluate pure baselines
    ause_base = compute_ause_fast(error_base, u_base)
    ause_gmm_raw = compute_ause_fast(error, det_sigma)
    
    best_max_pool = {"beta": None, "ause": float("inf"), "history": []}
    best_gating_k = {"beta": None, "ause": float("inf"), "history": []}
    best_gating_tau = {"beta": None, "tau": None, "ause": float("inf")}
    best_softmax = {"beta": None, "ause": float("inf"), "history": []}
    
    log.info(f"Evaluating {len(beta_candidates)} beta candidates...")
    
    for beta in beta_candidates:
        # Exponential mapping to probabilistic [0, 1] space
        u_gmm = 1.0 - np.exp(-beta * det_sigma)
        
        # --- Strategy A: Max-Pooling ---
        u_max_pool = np.maximum(u_base, u_gmm)
        ause_mp = compute_ause_fast(error, u_max_pool)
        best_max_pool["history"].append(ause_mp)
        if ause_mp < best_max_pool["ause"]:
            best_max_pool["ause"] = ause_mp
            best_max_pool["beta"] = beta
            
        # --- Strategy B1: Pure topological gating (K == 2) ---
        u_gating_k = np.where(n_comp == 2, u_gmm, u_base)
        ause_gk = compute_ause_fast(error, u_gating_k)
        best_gating_k["history"].append(ause_gk)
        if ause_gk < best_gating_k["ause"]:
            best_gating_k["ause"] = ause_gk
            best_gating_k["beta"] = beta
            
        # --- Strategy B2: Hybrid gating (K == 2 or u_gmm > tau) ---
        for tau in tau_candidates:
            condition = (n_comp == 2) | (u_gmm > tau)
            u_gating_tau = np.where(condition, u_gmm, u_base)
            ause_gt = compute_ause_fast(error, u_gating_tau)
            if ause_gt < best_gating_tau["ause"]:
                best_gating_tau["ause"] = ause_gt
                best_gating_tau["beta"] = beta
                best_gating_tau["tau"] = tau

        # --- Strategy C: Weighted Softmax ---
        exp_u_base = np.exp(u_base)
        exp_u_gmm = np.exp(u_gmm)
        sum_exp = exp_u_base + exp_u_gmm
        w_base = exp_u_base / sum_exp
        w_gmm = exp_u_gmm / sum_exp
        u_softmax = w_base * u_base + w_gmm * u_gmm
        ause_softmax = compute_ause_fast(error, u_softmax)
        best_softmax["history"].append(ause_softmax)
        if ause_softmax < best_softmax["ause"]:
            best_softmax["ause"] = ause_softmax
            best_softmax["beta"] = beta

    return {
        "baseline_argmax_ause": ause_base,
        "baseline_gmm_raw_ause": ause_gmm_raw,
        "max_pooling": best_max_pool,
        "gating_k": best_gating_k,
        "gating_tau": best_gating_tau,
        "softmax": best_softmax,
        "beta_candidates": beta_candidates
    }


def plot_optimization_curves(results: Dict[str, Any], output_dir: Path, suffix: str = ""):
    """Generate AUSE vs Beta plot to visualize hyperparameter sensitivity."""
    betas = results["beta_candidates"]
    mp_hist = results["max_pooling"]["history"]
    gk_hist = results["gating_k"]["history"]
    sm_hist = results["softmax"]["history"]
    base_ause = results["baseline_argmax_ause"]
    gmm_raw_ause = results["baseline_gmm_raw_ause"]
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(betas, mp_hist, label=f"Ours (Max-Pool) - Best AUSE: {results['max_pooling']['ause']:.4f}", color="magenta", linewidth=2)
    plt.semilogx(betas, gk_hist, label=f"Ours (Gating K=2) - Best AUSE: {results['gating_k']['ause']:.4f}", color="cyan", linewidth=2)
    plt.semilogx(betas, sm_hist, label=f"Ours (Softmax) - Best AUSE: {results['softmax']['ause']:.4f}", color="orange", linewidth=2)
    
    plt.axhline(base_ause, color="red", linestyle="--", label=f"DARK (Base) AUSE: {base_ause:.4f}")
    plt.axhline(gmm_raw_ause, color="blue", linestyle=":", label=f"Ours (Volume) AUSE: {gmm_raw_ause:.4f}")
    
    # Mark optimal points
    plt.scatter([results['max_pooling']['beta']], [results['max_pooling']['ause']], color="magenta", s=80, zorder=5)
    plt.scatter([results['gating_k']['beta']], [results['gating_k']['ause']], color="cyan", s=80, zorder=5)
    plt.scatter([results['softmax']['beta']], [results['softmax']['ause']], color="orange", s=80, zorder=5)
    
    plt.xlabel(r"Sensitivity Hyperparameter $\beta$ (log scale)")
    plt.ylabel("Area Under Sparsification Error (AUSE - lower is better)")
    title_suffix = suffix.replace("_", " ").title().strip()
    plt.title(rf"$\beta$ Hyperparameter Optimization ({title_suffix if title_suffix else 'Global'})")
    plt.legend()
    
    out_file = output_dir / f"ause_vs_beta{suffix}.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Optimization curve saved to {out_file}")


def run_grid_search(input_path: Path, output_dir: Path):
    """Execute hyperparameter grid search and output report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_dir():
        parquets = list(input_path.glob("*.parquet"))
        if not parquets:
            log.error(f"No parquet files found in {input_path}")
            sys.exit(1)
        df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    else:
        df = pd.read_parquet(input_path)
        
    log.info("Unrolling keypoints for optimization...")
    df_kp = _unroll_keypoints(df)
    
    # Strictly filter valid points (vis > 0) for geometric AUSE optimization
    df_kp = df_kp[df_kp["vis"] > 0].dropna(subset=["oks_ours", "cov_det"]).copy()
    
    if df_kp.empty:
        log.error("Not enough valid data in parquet to optimize.")
        return
        
    # Explore betas over wide logarithmic range
    beta_candidates = np.logspace(-6, 4, num=300)
    
    log.info("=== GLOBAL OPTIMIZATION ===")
    res_all = optimize_strategies(df_kp, beta_candidates)
    plot_optimization_curves(res_all, output_dir, suffix="_all")
    
    # Separate optimization for Visible and Occluded
    df_vis = df_kp[df_kp["vis"] == 2]
    df_occ = df_kp[df_kp["vis"] == 1]
    
    if not df_vis.empty:
        log.info("=== VISIBLE KEYPOINTS OPTIMIZATION ===")
        res_vis = optimize_strategies(df_vis, beta_candidates)
        plot_optimization_curves(res_vis, output_dir, suffix="_visible")
        
    if not df_occ.empty:
        log.info("=== OCCLUDED KEYPOINTS OPTIMIZATION ===")
        res_occ = optimize_strategies(df_occ, beta_candidates)
        plot_optimization_curves(res_occ, output_dir, suffix="_occluded")
        
    print("\n" + "="*70)
    print("        ADAPTIVE UNCERTAINTY OPTIMIZATION REPORT        ")
    print("="*70)
    print(f"Total analyzed keypoints: {len(df_kp)}")
    print(f"Baseline Argmax AUSE:      {res_all['baseline_argmax_ause']:.5f}")
    print(f"GMM det(Sigma) Raw AUSE:   {res_all['baseline_gmm_raw_ause']:.5f}")
    print("-" * 70)
    print("STRATEGY A (Max-Pooling): U_adapt = max(1 - P_argmax, U_gmm)")
    print(f"  -> Best Beta: {res_all['max_pooling']['beta']:.6g}")
    print(f"  -> Optimal AUSE: {res_all['max_pooling']['ause']:.5f} (Improvement: {res_all['baseline_argmax_ause'] - res_all['max_pooling']['ause']:.5f})")
    print("-" * 70)
    print("STRATEGY B1 (Pure Topological Gating K=2):")
    print(f"  -> Best Beta: {res_all['gating_k']['beta']:.6g}")
    print(f"  -> Optimal AUSE: {res_all['gating_k']['ause']:.5f} (Improvement: {res_all['baseline_argmax_ause'] - res_all['gating_k']['ause']:.5f})")
    print("-" * 70)
    print("STRATEGY B2 (Hybrid Gating K=2 or U_gmm > tau):")
    print(f"  -> Best Beta: {res_all['gating_tau']['beta']:.6g} | Best Tau: {res_all['gating_tau']['tau']:.4f}")
    print(f"  -> Optimal AUSE: {res_all['gating_tau']['ause']:.5f} (Improvement: {res_all['baseline_argmax_ause'] - res_all['gating_tau']['ause']:.5f})")
    print("-" * 70)
    print("STRATEGY C (Weighted Softmax):")
    print(f"  -> Best Beta: {res_all['softmax']['beta']:.6g}")
    print(f"  -> Optimal AUSE: {res_all['softmax']['ause']:.5f} (Improvement: {res_all['baseline_argmax_ause'] - res_all['softmax']['ause']:.5f})")
    print("="*70 + "\n")
    
    def _extract_clean_summary(res: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "baseline_argmax_ause": float(res["baseline_argmax_ause"]),
            "baseline_gmm_raw_ause": float(res["baseline_gmm_raw_ause"]),
            "max_pooling": {
                "best_beta": float(res["max_pooling"]["beta"]),
                "best_ause": float(res["max_pooling"]["ause"]),
                "ause_improvement": float(res["baseline_argmax_ause"] - res["max_pooling"]["ause"])
            },
            "gating_k": {
                "best_beta": float(res["gating_k"]["beta"]),
                "best_ause": float(res["gating_k"]["ause"]),
                "ause_improvement": float(res["baseline_argmax_ause"] - res["gating_k"]["ause"])
            },
            "gating_tau": {
                "best_beta": float(res["gating_tau"]["beta"]),
                "best_tau": float(res["gating_tau"]["tau"]),
                "best_ause": float(res["gating_tau"]["ause"]),
                "ause_improvement": float(res["baseline_argmax_ause"] - res["gating_tau"]["ause"])
            },
            "softmax": {
                "best_beta": float(res["softmax"]["beta"]),
                "best_ause": float(res["softmax"]["ause"]),
                "ause_improvement": float(res["baseline_argmax_ause"] - res["softmax"]["ause"])
            }
        }
        
    summary_results = {
        "all": _extract_clean_summary(res_all)
    }
    if not df_vis.empty and "res_vis" in locals():
        summary_results["visible"] = _extract_clean_summary(res_vis)
    if not df_occ.empty and "res_occ" in locals():
        summary_results["occluded"] = _extract_clean_summary(res_occ)
        
    json_path = output_dir / "optimization_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
        
    csv_rows = []
    for subset_name, sub_res in summary_results.items():
        csv_rows.append({"subset": subset_name, "strategy": "baseline_argmax", "metric": "ause", "value": sub_res["baseline_argmax_ause"]})
        csv_rows.append({"subset": subset_name, "strategy": "baseline_gmm_raw", "metric": "ause", "value": sub_res["baseline_gmm_raw_ause"]})
        for strat_name in ["max_pooling", "gating_k", "gating_tau", "softmax"]:
            for m_key, m_val in sub_res[strat_name].items():
                csv_rows.append({"subset": subset_name, "strategy": strat_name, "metric": m_key, "value": m_val})
                
    if csv_rows:
        df_csv = pd.DataFrame(csv_rows)
        csv_path = output_dir / "optimization_results.csv"
        df_csv.to_csv(csv_path, index=False, encoding="utf-8")
        log.info(f"Optimization results saved to {json_path} and {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid-Search to optimize beta hyperparameter in adaptive uncertainty.")
    parser.add_argument("input_path", type=str, help="Path to parquet file (or directory with parquets) of benchmark.")
    parser.add_argument("--out", type=str, default=None, help="Directory to save optimization plots (default auto-detected in graficas/).")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    input_path = Path(args.input_path)
    
    from src.experiments.degradation_organizer import run_organized_optimization_pipeline
    if run_organized_optimization_pipeline(input_path, out_arg=args.out):
        sys.exit(0)
        
    out_dir = Path(args.out) if args.out else Path("outputs/adaptive_optimization")
    run_grid_search(input_path, out_dir)

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo visual
sns.set_theme(style="whitegrid")
log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.evaluate_uncertainty import _unroll_keypoints


def compute_ause_fast(error: np.ndarray, uncertainty: np.ndarray) -> float:
    """Versión optimizada y vectorizada del cálculo del AUSE."""
    n = len(error)
    if n == 0:
        return 0.0
        
    fractions = np.linspace(0, 1, min(n, 100))
    
    # Ordenar por error (Oráculo) y por incertidumbre (Modelo)
    oracle_errors = error[np.argsort(-error)]
    model_errors = error[np.argsort(-uncertainty)]
    
    # Calcular sumas acumuladas inversas para acelerar la media
    oracle_cum_sum = np.cumsum(oracle_errors[::-1])[::-1]
    model_cum_sum = np.cumsum(model_errors[::-1])[::-1]
    
    # Evitar división por cero
    counts = np.arange(n, 0, -1)
    
    oracle_means = oracle_cum_sum / counts
    model_means = model_cum_sum / counts
    
    # Mapear a las fracciones 0-1
    indices = np.clip((fractions * n).astype(int), 0, n - 1)
    
    ause = np.trapz(model_means[indices], fractions) - np.trapz(oracle_means[indices], fractions)
    return float(ause)


def optimize_strategies(
    df: pd.DataFrame, 
    beta_candidates: np.ndarray,
    tau_candidates: np.ndarray = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
) -> Dict[str, Any]:
    """Realiza la búsqueda en rejilla para Estrategia A (Max-Pooling) y Estrategia B (Gating)."""
    error = (1.0 - df["oks_ours"]).values
    
    # Inverso de la confianza argmax: 1 - P_argmax
    u_base = (1.0 - df["base_score"]).values if "base_score" in df.columns else np.zeros_like(error)
    det_sigma = df["cov_det"].values
    n_comp = df["n_components"].values if "n_components" in df.columns else np.ones_like(error)
    
    # 1. Evaluar baselines puros
    ause_base = compute_ause_fast(error, u_base)
    ause_gmm_raw = compute_ause_fast(error, det_sigma)
    
    best_max_pool = {"beta": None, "ause": float("inf"), "history": []}
    best_gating_k = {"beta": None, "ause": float("inf"), "history": []}
    best_gating_tau = {"beta": None, "tau": None, "ause": float("inf")}
    
    log.info(f"Evaluando {len(beta_candidates)} candidatos de beta...")
    
    for beta in beta_candidates:
        # Mapeo exponencial al espacio probabilístico [0, 1]
        u_gmm = 1.0 - np.exp(-beta * det_sigma)
        
        # --- Estrategia A: Max-Pooling ---
        u_max_pool = np.maximum(u_base, u_gmm)
        ause_mp = compute_ause_fast(error, u_max_pool)
        best_max_pool["history"].append(ause_mp)
        if ause_mp < best_max_pool["ause"]:
            best_max_pool["ause"] = ause_mp
            best_max_pool["beta"] = beta
            
        # --- Estrategia B1: Gating por topología pura (K == 2) ---
        # Si K==2 (bimodal), confiamos en u_gmm; si K==1 (unimodal), confiamos en u_base
        u_gating_k = np.where(n_comp == 2, u_gmm, u_base)
        ause_gk = compute_ause_fast(error, u_gating_k)
        best_gating_k["history"].append(ause_gk)
        if ause_gk < best_gating_k["ause"]:
            best_gating_k["ause"] = ause_gk
            best_gating_k["beta"] = beta
            
        # --- Estrategia B2: Gating híbrido (K == 2 o u_gmm > tau) ---
        for tau in tau_candidates:
            condition = (n_comp == 2) | (u_gmm > tau)
            u_gating_tau = np.where(condition, u_gmm, u_base)
            ause_gt = compute_ause_fast(error, u_gating_tau)
            if ause_gt < best_gating_tau["ause"]:
                best_gating_tau["ause"] = ause_gt
                best_gating_tau["beta"] = beta
                best_gating_tau["tau"] = tau

    return {
        "baseline_argmax_ause": ause_base,
        "baseline_gmm_raw_ause": ause_gmm_raw,
        "max_pooling": best_max_pool,
        "gating_k": best_gating_k,
        "gating_tau": best_gating_tau,
        "beta_candidates": beta_candidates
    }


def plot_optimization_curves(results: Dict[str, Any], output_dir: Path, suffix: str = ""):
    """Genera gráfica de AUSE vs Beta para visualizar la sensibilidad del hiperparámetro."""
    betas = results["beta_candidates"]
    mp_hist = results["max_pooling"]["history"]
    gk_hist = results["gating_k"]["history"]
    base_ause = results["baseline_argmax_ause"]
    gmm_raw_ause = results["baseline_gmm_raw_ause"]
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(betas, mp_hist, label=f"Estrategia A (Max-Pooling) - Mejor AUSE: {results['max_pooling']['ause']:.4f}", color="magenta", linewidth=2)
    plt.semilogx(betas, gk_hist, label=f"Estrategia B (Gating K=2) - Mejor AUSE: {results['gating_k']['ause']:.4f}", color="cyan", linewidth=2)
    
    plt.axhline(base_ause, color="red", linestyle="--", label=f"Baseline Argmax AUSE: {base_ause:.4f}")
    plt.axhline(gmm_raw_ause, color="blue", linestyle=":", label=f"GMM det(Sigma) Raw AUSE: {gmm_raw_ause:.4f}")
    
    # Marcar los puntos óptimos
    plt.scatter([results['max_pooling']['beta']], [results['max_pooling']['ause']], color="magenta", s=80, zorder=5)
    plt.scatter([results['gating_k']['beta']], [results['gating_k']['ause']], color="cyan", s=80, zorder=5)
    
    plt.xlabel(r"Hiperparámetro de Sensibilidad $\beta$ (escala logarítmica)")
    plt.ylabel("Area Under Sparsification Error (AUSE - menor es mejor)")
    title_suffix = suffix.replace("_", " ").title().strip()
    plt.title(rf"Optimización de Hiperparámetro $\beta$ ({title_suffix if title_suffix else 'Global'})")
    plt.legend()
    
    out_file = output_dir / f"ause_vs_beta{suffix}.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Curva de optimización guardada en {out_file}")


def run_grid_search(input_path: Path, output_dir: Path):
    """Ejecuta la búsqueda de hiperparámetros y muestra informe."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_dir():
        parquets = list(input_path.glob("*.parquet"))
        if not parquets:
            log.error(f"No se encontraron archivos parquet en {input_path}")
            sys.exit(1)
        df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    else:
        df = pd.read_parquet(input_path)
        
    log.info("Desenrollando keypoints para optimización...")
    df_kp = _unroll_keypoints(df)
    
    # Filtrar estrictamente puntos válidos (vis > 0) para optimización AUSE geométrica
    df_kp = df_kp[df_kp["vis"] > 0].dropna(subset=["oks_ours", "cov_det"]).copy()
    
    if df_kp.empty:
        log.error("No hay suficientes datos válidos en el parquet para optimizar.")
        return
        
    # Explorar betas en rango amplio logarítmico
    beta_candidates = np.logspace(-6, 4, num=300)
    
    log.info("=== OPTIMIZACIÓN GLOBAL ===")
    res_all = optimize_strategies(df_kp, beta_candidates)
    plot_optimization_curves(res_all, output_dir, suffix="_all")
    
    # Optimización separada para Visibles y Ocluidos
    df_vis = df_kp[df_kp["vis"] == 2]
    df_occ = df_kp[df_kp["vis"] == 1]
    
    if not df_vis.empty:
        log.info("=== OPTIMIZACIÓN KEYPOINTS VISIBLES ===")
        res_vis = optimize_strategies(df_vis, beta_candidates)
        plot_optimization_curves(res_vis, output_dir, suffix="_visible")
        
    if not df_occ.empty:
        log.info("=== OPTIMIZACIÓN KEYPOINTS OCLUIDOS ===")
        res_occ = optimize_strategies(df_occ, beta_candidates)
        plot_optimization_curves(res_occ, output_dir, suffix="_occluded")
        
    print("\n" + "="*70)
    print("        INFORME DE OPTIMIZACIÓN DE INCERTIDUMBRE ADAPTATIVA        ")
    print("="*70)
    print(f"Total keypoints analizados: {len(df_kp)}")
    print(f"Baseline Argmax AUSE:      {res_all['baseline_argmax_ause']:.5f}")
    print(f"GMM det(Sigma) Raw AUSE:   {res_all['baseline_gmm_raw_ause']:.5f}")
    print("-" * 70)
    print("ESTRATEGIA A (Max-Pooling): U_adapt = max(1 - P_argmax, U_gmm)")
    print(f"  -> Mejor Beta: {res_all['max_pooling']['beta']:.6g}")
    print(f"  -> AUSE Óptimo: {res_all['max_pooling']['ause']:.5f} (Mejora: {res_all['baseline_argmax_ause'] - res_all['max_pooling']['ause']:.5f})")
    print("-" * 70)
    print("ESTRATEGIA B1 (Gating Topológico Puro K=2):")
    print(f"  -> Mejor Beta: {res_all['gating_k']['beta']:.6g}")
    print(f"  -> AUSE Óptimo: {res_all['gating_k']['ause']:.5f} (Mejora: {res_all['baseline_argmax_ause'] - res_all['gating_k']['ause']:.5f})")
    print("-" * 70)
    print("ESTRATEGIA B2 (Gating Híbrido K=2 o U_gmm > tau):")
    print(f"  -> Mejor Beta: {res_all['gating_tau']['beta']:.6g} | Mejor Tau: {res_all['gating_tau']['tau']:.4f}")
    print(f"  -> AUSE Óptimo: {res_all['gating_tau']['ause']:.5f} (Mejora: {res_all['baseline_argmax_ause'] - res_all['gating_tau']['ause']:.5f})")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid-Search para optimizar hiperparámetro beta en incertidumbre adaptativa.")
    parser.add_argument("input_path", type=str, help="Ruta al archivo parquet (o directorio con parquets) del benchmark.")
    parser.add_argument("--out", type=str, default="outputs/adaptive_optimization", help="Directorio para guardar gráficas de optimización.")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    run_grid_search(Path(args.input_path), Path(args.out))

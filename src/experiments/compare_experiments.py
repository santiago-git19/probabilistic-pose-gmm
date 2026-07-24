"""
Script de Comparación de Experimentos (Benchmarking Comparativo).

Permite definir múltiples configuraciones y ejecutarlas secuencialmente
para comparar métricas (A/B Testing, Grid Search manual, etc.).

Principios:
    - **Inmutabilidad**: Cada experimento parte de un ``deepcopy`` de la
      configuración base; los overrides SOLO afectan al experimento actual.
    - **Eficiencia**: El DataLoader y el modelo se instancian una sola vez
      y se reutilizan en todos los experimentos (salvo cambio de modelo).
    - **Robustez**: Si un experimento falla, el script continúa con el
      siguiente y registra el error.

Uso::

    cd <project_root>
    python src/experiments/compare_experiments.py

    # Con debug_limit para pruebas rápidas:
    python src/experiments/compare_experiments.py evaluation.debug_limit=20
"""

from __future__ import annotations

import copy
import datetime
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Add project root to sys.path (so ``src.…`` imports work under Hydra)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.pose_uncertainty.evaluation.runner import EvaluationRunner, _build_dataloader
from src.pose_uncertainty.models.adapters import create_model_adapter
from src.pose_uncertainty.tracking import wandb_run, log_metrics, log_summary_table

log = logging.getLogger(__name__)

# =============================================================================
# ████  DEFINICIÓN DE EXPERIMENTOS  ████
# =============================================================================
# Formato:  "Nombre_Experimento": { "clave.anidada.hydra": valor, ... }
#
# Cada diccionario interno contiene SOLO los parámetros que difieren de
# config.yaml.  Todo lo demás permanece intacto.
#
# ESTUDIO ABLATIVO TTA - Prueba cada componente por separado y combinado
# =============================================================================

'''
EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # BASELINE: Sin TTA
    # =========================================================================
    
    "00_Baseline_NoTTA": {
        "flip.enabled": False,
        "scale.enabled": False,
        "photometric.brightness.enabled": False,
        "photometric.contrast.enabled": False,
        "photometric.noise.enabled": False,
        "photometric.blur.enabled": False,
    },

    # =========================================================================
    # FLIP ONLY
    # =========================================================================
    "01_FlipOnly": {
        "flip.enabled": True,
        "scale.enabled": False,
        "photometric.brightness.enabled": False,
        "photometric.contrast.enabled": False,
        "photometric.noise.enabled": False,
        "photometric.blur.enabled": False,
    },
    
    # =========================================================================
    # SCALE ONLY - Diferentes configuraciones
    # =========================================================================
    "02_ScaleOnly_Conservative": {
        "flip.enabled": False,
        "scale.enabled": True,
        "scale.scales": [0.9, 1.0, 1.1],
        "scale.aggregation": "weighted_mean",
        "scale.sharpness_scale": 5.0,
        "scale.peak_exponent": 1.0,
        "scale.sharpness_exponent": 1.0,
        "photometric.brightness.enabled": False,
        "photometric.contrast.enabled": False,
        "photometric.noise.enabled": False,
        "photometric.blur.enabled": False,
    },
    
    "03_ScaleOnly_Moderate": {
        "flip.enabled": False,
        "scale.enabled": True,
        "scale.scales": [0.85, 0.925, 1.0, 1.075, 1.15],
        "scale.aggregation": "weighted_mean",
        "scale.sharpness_scale": 5.0,
        "scale.peak_exponent": 1.0,
        "scale.sharpness_exponent": 1.0,
        "photometric.brightness.enabled": False,
        "photometric.contrast.enabled": False,
        "photometric.noise.enabled": False,
        "photometric.blur.enabled": False,
    },
    
    "04_ScaleOnly_Aggressive": {
        "flip.enabled": False,
        "scale.enabled": True,
        "scale.scales": [0.75, 0.85, 0.925, 1.0, 1.075, 1.15, 1.25],
        "scale.aggregation": "weighted_mean",
        "scale.sharpness_scale": 5.0,
        "scale.peak_exponent": 1.0,
        "scale.sharpness_exponent": 1.0,
        "photometric.brightness.enabled": False,
        "photometric.contrast.enabled": False,
        "photometric.noise.enabled": False,
        "photometric.blur.enabled": False,
    },

    # Scale con diferentes parámetros de weighting
    "05_ScaleOnly_HighPeakExp": {
        "flip.enabled": False,
        "scale.enabled": True,
        "scale.scales": [0.85, 0.925, 1.0, 1.075, 1.15],
        "scale.aggregation": "weighted_mean",
        "scale.sharpness_scale": 5.0,
        "scale.peak_exponent": 1.5,  # Suppress weak peaks more aggressively
        "scale.sharpness_exponent": 1.0,
        "photometric.brightness.enabled": False,
        "photometric.contrast.enabled": False,
        "photometric.noise.enabled": False,
        "photometric.blur.enabled": False,
    },
    
    "06_ScaleOnly_HighSharpExp": {
        "flip.enabled": False,
        "scale.enabled": True,
        "scale.scales": [0.85, 0.925, 1.0, 1.075, 1.15],
        "scale.aggregation": "weighted_mean",
        "scale.sharpness_scale": 5.0,
        "scale.peak_exponent": 1.0,
        "scale.sharpness_exponent": 1.5,  # Emphasize sharp heatmaps more
        "photometric.brightness.enabled": False,
        "photometric.contrast.enabled": False,
        "photometric.noise.enabled": False,
        "photometric.blur.enabled": False,
    },
    
    "07_ScaleOnly_GentleSharpness": {
        "flip.enabled": False,
        "scale.enabled": True,
        "scale.scales": [0.85, 0.925, 1.0, 1.075, 1.15],
        "scale.aggregation": "weighted_mean",
        "scale.sharpness_scale": 10.0,  # Gentler sigmoid curve
        "scale.peak_exponent": 1.0,
        "scale.sharpness_exponent": 1.0,
        "photometric.brightness.enabled": False,
        "photometric.contrast.enabled": False,
        "photometric.noise.enabled": False,
        "photometric.blur.enabled": False,
    },

    "08_ScaleOnly_MaxConfidence": {
        "flip.enabled": False,
        "scale.enabled": True,
        "scale.scales": [0.85, 0.925, 1.0, 1.075, 1.15],
        "scale.aggregation": "max_confidence",  # Pick best scale per keypoint
        "photometric.brightness.enabled": False,
        "photometric.contrast.enabled": False,
        "photometric.noise.enabled": False,
        "photometric.blur.enabled": False,
    },

    # =========================================================================
    # PHOTOMETRIC ONLY - Cada uno por separado
    # =========================================================================
    "09_PhotometricOnly_Brightness": {
        "flip.enabled": False,
        "scale.enabled": False,
        "photometric.brightness.enabled": True,
        "photometric.brightness.delta": 30.0,
        "photometric.contrast.enabled": False,
        "photometric.noise.enabled": False,
        "photometric.blur.enabled": False,
    },
    
    "10_PhotometricOnly_Contrast": {
        "flip.enabled": False,
        "scale.enabled": False,
        "photometric.brightness.enabled": False,
        "photometric.contrast.enabled": True,
        "photometric.contrast.range": [0.8, 1.2],
        "photometric.noise.enabled": False,
        "photometric.blur.enabled": False,
    },
    
    "11_PhotometricOnly_Noise": {
        "flip.enabled": False,
        "scale.enabled": False,
        "photometric.brightness.enabled": False,
        "photometric.contrast.enabled": False,
        "photometric.noise.enabled": True,
        "photometric.noise.sigma": 10.0,
        "photometric.blur.enabled": False,
    },
    
    "12_PhotometricOnly_Blur": {
        "flip.enabled": False,
        "scale.enabled": False,
        "photometric.brightness.enabled": False,
        "photometric.contrast.enabled": False,
        "photometric.noise.enabled": False,
        "photometric.blur.enabled": True,
        "photometric.blur.kernel_size": 3,
    },
    
    "13_PhotometricOnly_AllCombined": {
        "flip.enabled": False,
        "scale.enabled": False,
        "photometric.brightness.enabled": True,
        "photometric.brightness.delta": 30.0,
        "photometric.contrast.enabled": True,
        "photometric.contrast.range": [0.8, 1.2],
        "photometric.noise.enabled": True,
        "photometric.noise.sigma": 10.0,
        "photometric.blur.enabled": True,
        "photometric.blur.kernel_size": 3,
    },

    # =========================================================================
    # FLIP + SCALE
    # =========================================================================
    "14_FlipScale_Conservative": {
        "flip.enabled": True,
        "scale.enabled": True,
        "scale.scales": [0.9, 1.0, 1.1],
        "scale.aggregation": "weighted_mean",
        "photometric.brightness.enabled": False,
        "photometric.contrast.enabled": False,
        "photometric.noise.enabled": False,
        "photometric.blur.enabled": False,
    },
    
    "15_FlipScale_Moderate": {
        "flip.enabled": True,
        "scale.enabled": True,
        "scale.scales": [0.85, 0.925, 1.0, 1.075, 1.15],
        "scale.aggregation": "weighted_mean",
        "photometric.brightness.enabled": False,
        "photometric.contrast.enabled": False,
        "photometric.noise.enabled": False,
        "photometric.blur.enabled": False,
    },

    # =========================================================================
    # FLIP + PHOTOMETRIC (cada uno)
    # =========================================================================
    "16_FlipPhoto_Brightness": {
        "flip.enabled": True,
        "scale.enabled": False,
        "photometric.brightness.enabled": True,
        "photometric.contrast.enabled": False,
        "photometric.noise.enabled": False,
        "photometric.blur.enabled": False,
    },
    
    "17_FlipPhoto_Contrast": {
        "flip.enabled": True,
        "scale.enabled": False,
        "photometric.brightness.enabled": False,
        "photometric.contrast.enabled": True,
        "photometric.noise.enabled": False,
        "photometric.blur.enabled": False,
    },
    
    "18_FlipPhoto_Noise": {
        "flip.enabled": True,
        "scale.enabled": False,
        "photometric.brightness.enabled": False,
        "photometric.contrast.enabled": False,
        "photometric.noise.enabled": True,
        "photometric.blur.enabled": False,
    },
    
    "19_FlipPhoto_Blur": {
        "flip.enabled": True,
        "scale.enabled": False,
        "photometric.brightness.enabled": False,
        "photometric.contrast.enabled": False,
        "photometric.noise.enabled": False,
        "photometric.blur.enabled": True,
    },
    
    "20_FlipPhoto_AllCombined": {
        "flip.enabled": True,
        "scale.enabled": False,
        "photometric.brightness.enabled": True,
        "photometric.contrast.enabled": True,
        "photometric.noise.enabled": True,
        "photometric.blur.enabled": True,
    },

    # =========================================================================
    # SCALE + PHOTOMETRIC (selección)
    # =========================================================================
    "21_ScalePhoto_Brightness": {
        "flip.enabled": False,
        "scale.enabled": True,
        "scale.scales": [0.85, 0.925, 1.0, 1.075, 1.15],
        "photometric.brightness.enabled": True,
        "photometric.contrast.enabled": False,
        "photometric.noise.enabled": False,
        "photometric.blur.enabled": False,
    },
    
    "22_ScalePhoto_AllCombined": {
        "flip.enabled": False,
        "scale.enabled": True,
        "scale.scales": [0.85, 0.925, 1.0, 1.075, 1.15],
        "photometric.brightness.enabled": True,
        "photometric.contrast.enabled": True,
        "photometric.noise.enabled": True,
        "photometric.blur.enabled": True,
    },

    # =========================================================================
    # FULL STACK: FLIP + SCALE + PHOTOMETRIC
    # =========================================================================
    "23_FullStack_Conservative": {
        "flip.enabled": True,
        "scale.enabled": True,
        "scale.scales": [0.9, 1.0, 1.1],
        "scale.aggregation": "weighted_mean",
        "photometric.brightness.enabled": True,
        "photometric.contrast.enabled": True,
        "photometric.noise.enabled": True,
        "photometric.blur.enabled": True,
    },
    
    "24_FullStack_Moderate": {
        "flip.enabled": True,
        "scale.enabled": True,
        "scale.scales": [0.85, 0.925, 1.0, 1.075, 1.15],
        "scale.aggregation": "weighted_mean",
        "photometric.brightness.enabled": True,
        "photometric.contrast.enabled": True,
        "photometric.noise.enabled": True,
        "photometric.blur.enabled": True,
    },
    
    "25_FullStack_Aggressive": {
        "flip.enabled": True,
        "scale.enabled": True,
        "scale.scales": [0.75, 0.85, 0.925, 1.0, 1.075, 1.15, 1.25],
        "scale.aggregation": "weighted_mean",
        "photometric.brightness.enabled": True,
        "photometric.contrast.enabled": True,
        "photometric.noise.enabled": True,
        "photometric.blur.enabled": True,
    },
}
'''
'''
EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # A) RUIDO GAUSSIANO BLANCO
    # =========================================================================
    "Noise_Sigma_25": {"dataset.noise_sigma": 25.0, "wandb_group": "degradation_noise"},
    "Noise_Sigma_50": {"dataset.noise_sigma": 50.0, "wandb_group": "degradation_noise"},
    "Noise_Sigma_75": {"dataset.noise_sigma": 75.0, "wandb_group": "degradation_noise"},
    "Noise_Sigma_100": {"dataset.noise_sigma": 100.0, "wandb_group": "degradation_noise"},
    "Noise_Sigma_200": {"dataset.noise_sigma": 200.0, "wandb_group": "degradation_noise"},
    "Noise_Sigma_300": {"dataset.noise_sigma": 300.0, "wandb_group": "degradation_noise"},
    "Noise_Sigma_400": {"dataset.noise_sigma": 400.0, "wandb_group": "degradation_noise"},

    # =========================================================================
    # B) REDUCCIÓN DE CONTRASTE
    # =========================================================================
    "Contrast_0.75": {"dataset.contrast_factor": 0.75, "wandb_group": "degradation_contrast"},
    "Contrast_0.50": {"dataset.contrast_factor": 0.50, "wandb_group": "degradation_contrast"},
    "Contrast_0.25": {"dataset.contrast_factor": 0.25, "wandb_group": "degradation_contrast"},
    "Contrast_0.10": {"dataset.contrast_factor": 0.10, "wandb_group": "degradation_contrast"},
    "Contrast_0.05": {"dataset.contrast_factor": 0.05, "wandb_group": "degradation_contrast"},

    # =========================================================================
    # C) GAUSSIAN BLUR (Emborronamiento)
    # =========================================================================
    "Blur_Kernel_21": {"dataset.blur_kernel_size": 21, "wandb_group": "degradation_blur"},
    "Blur_Kernel_31": {"dataset.blur_kernel_size": 31, "wandb_group": "degradation_blur"},
    "Blur_Kernel_41": {"dataset.blur_kernel_size": 41, "wandb_group": "degradation_blur"},
    "Blur_Kernel_51": {"dataset.blur_kernel_size": 51, "wandb_group": "degradation_blur"},
    "Blur_Kernel_61": {"dataset.blur_kernel_size": 61, "wandb_group": "degradation_blur"},
    "Blur_Kernel_71": {"dataset.blur_kernel_size": 71, "wandb_group": "degradation_blur"},

    # =========================================================================
    # D) SMOOTH / LOW PASS FILTER (Suavizado)
    # =========================================================================
    "Smooth_Kernel_10": {"dataset.smooth_kernel_size": 10, "wandb_group": "degradation_smooth"},
    "Smooth_Kernel_20": {"dataset.smooth_kernel_size": 20, "wandb_group": "degradation_smooth"},
    "Smooth_Kernel_30": {"dataset.smooth_kernel_size": 30, "wandb_group": "degradation_smooth"},
    "Smooth_Kernel_40": {"dataset.smooth_kernel_size": 40, "wandb_group": "degradation_smooth"},
}
'''

EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "resolution_0.5": {"dataset.resize_scale": 0.5, "wandb_group": "OCHUMAN_resolution"},
    "resolution_0.25": {"dataset.resize_scale": 0.25, "wandb_group": "OCHUMAN_resolution"},
    "resolution_0.125": {"dataset.resize_scale": 0.125, "wandb_group": "OCHUMAN_resolution"},
}


# =============================================================================
# ████  FIN DEFINICIÓN DE EXPERIMENTOS  ████
# =============================================================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    """Convierte un nombre de experimento en uno seguro para rutas."""
    return (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "-")
        .replace("\\", "-")
    )


def _set_nested(cfg: DictConfig, dotted_key: str, value: Any) -> None:
    """Establece ``cfg[a][b][c] = value`` dada la clave ``'a.b.c'``.

    Crea nodos intermedios si no existen (dentro de ``open_dict``).
    """
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        child = OmegaConf.select(node, part, default=None)
        if child is None:
            # Crear nodo intermedio
            OmegaConf.update(node, part, {})
            child = OmegaConf.select(node, part)
        node = child
    OmegaConf.update(node, parts[-1], value)


def apply_overrides(cfg: DictConfig, overrides: Dict[str, Any]) -> None:
    """Aplica un diccionario de overrides (claves dot-separated) a *cfg*.

    Usa ``open_dict`` para permitir escritura en configuraciones struct y
    ``OmegaConf.update`` para navegar la jerarquía de forma segura.
    """
    with open_dict(cfg):
        for key, val in overrides.items():
            try:
                _set_nested(cfg, key, val)
                log.info("  -> Override: %s = %s", key, val)
            except Exception as exc:
                log.error("  [ERROR] No se pudo aplicar override '%s': %s", key, exc)


def _extract_scalar_metrics(
    df: pd.DataFrame, experiment_name: str, overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Extrae métricas agregadas de un DataFrame de resultados.

    Es tolerante a columnas ausentes (devuelve ``NaN`` si no existen).
    """
    def _safe_mean(col: str) -> float:
        if col in df.columns:
            return float(df[col].mean())
        return float("nan")

    oks_ours = _safe_mean("oks_ours")
    oks_base = _safe_mean("oks_base")
    oks_tta = _safe_mean("oks_tta")

    swaps_base_total = int(df["swaps_base"].sum()) if "swaps_base" in df.columns else 0
    swaps_ours_total = int(df["swaps_ours"].sum()) if "swaps_ours" in df.columns else 0
    swaps_tta_total = int(df["swaps_tta"].sum()) if "swaps_tta" in df.columns else 0
    swaps_corrected = int(df["swaps_corrected"].sum()) if "swaps_corrected" in df.columns else 0
    swaps_introduced = int(df["swaps_introduced"].sum()) if "swaps_introduced" in df.columns else 0
    swaps_corrected_tta = int(df["swaps_corrected_tta"].sum()) if "swaps_corrected_tta" in df.columns else 0
    swaps_introduced_tta = int(df["swaps_introduced_tta"].sum()) if "swaps_introduced_tta" in df.columns else 0

    return {
        "experiment": experiment_name,
        "overrides": json.dumps(overrides, default=str),
        "n_images": len(df),
        "oks_ours_mean": oks_ours,
        "oks_base_mean": oks_base,
        "oks_tta_mean": oks_tta,
        "delta_oks_mean": oks_ours - oks_base if not (np.isnan(oks_ours) or np.isnan(oks_base)) else float("nan"),
        "delta_oks_tta_mean": oks_tta - oks_base if not (np.isnan(oks_tta) or np.isnan(oks_base)) else float("nan"),
        "delta_oks_ours_over_tta_mean": oks_ours - oks_tta if not (np.isnan(oks_ours) or np.isnan(oks_tta)) else float("nan"),
        "nll_mean": _safe_mean("nll"),
        "entropy_mean": _safe_mean("entropy"),
        "covariance_vol_mean": _safe_mean("covariance_vol"),
        "n_components_mean": _safe_mean("n_components"),
        "swaps_base_total": swaps_base_total,
        "swaps_ours_total": swaps_ours_total,
        "swaps_tta_total": swaps_tta_total,
        "swaps_corrected": swaps_corrected,
        "swaps_introduced": swaps_introduced,
        "swaps_corrected_tta": swaps_corrected_tta,
        "swaps_introduced_tta": swaps_introduced_tta,
    }


def _try_free_gpu() -> None:
    """Libera caché de GPU si torch está disponible."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    log.info("=" * 70)
    log.info("  EXPERIMENT COMPARISON BENCHMARK")
    log.info("  %d experiments defined", len(EXPERIMENTS))
    log.info("=" * 70)

    # Directorio raíz del proyecto (absoluto, no depende del CWD de Hydra)
    project_root = _PROJECT_ROOT
    comparison_dir = project_root / "outputs" / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Recurso compartido: DataLoader  (una sola instancia)
    # ------------------------------------------------------------------
    #log.info("[SETUP] Cargando dataset (compartido entre experimentos)...")
    #dataloader = _build_dataloader(cfg)

    # Debug limit (idéntico al que soporta run_benchmark.py)
    debug_limit: Optional[int] = OmegaConf.select(
        cfg, "evaluation.debug_limit", default=None
    )
    if debug_limit is not None:
        log.warning(
            "DEBUG MODE: Cada experimento procesará como máximo %d imágenes.",
            debug_limit,
        )

    # ------------------------------------------------------------------
    # 2) Recurso compartido: Modelo  (se reutiliza si no cambia)
    # ------------------------------------------------------------------
    base_model_name: str = OmegaConf.select(cfg, "model.name", default="hrnet_w32")
    base_device: str = OmegaConf.select(cfg, "model.device", default="cuda")
    log.info("[SETUP] Cargando modelo '%s' en '%s' ...", base_model_name, base_device)
    shared_model = create_model_adapter(base_model_name, device=base_device)

    # ------------------------------------------------------------------
    # 3) Bucle de experimentos
    # ------------------------------------------------------------------
    results_summary: List[Dict[str, Any]] = []
    timings: Dict[str, float] = {}

    for exp_idx, (exp_name, overrides) in enumerate(
        tqdm(EXPERIMENTS.items(), desc="Experimentos", unit="exp"),
        start=1
    ):
        log.info("")
        log.info("-" * 60)
        log.info(
            ">>> [%d/%d] EXPERIMENT: %s", exp_idx, len(EXPERIMENTS), exp_name,
        )
        log.info("-" * 60)

        # ---- Clonar config de forma segura ----------------------------
        current_cfg = copy.deepcopy(cfg)

        # Directorio de salida aislado para este experimento
        safe_name = _sanitize_name(exp_name)
        exp_output = comparison_dir / safe_name
        with open_dict(current_cfg):
            OmegaConf.update(current_cfg, "logging.output_dir", str(exp_output))

        # ---- Extraer grupo W&B y limpiar overrides --------------------
        # Sacamos 'wandb_group' para no inyectarlo en la config de Hydra
        clean_overrides = overrides.copy()
        run_group = clean_overrides.pop("wandb_group", "compare_experiments")

        # ---- Aplicar overrides exclusivos de este experimento ---------
        apply_overrides(current_cfg, clean_overrides)

        # ---- Ejecutar evaluación --------------------------------------
        t0 = time.perf_counter()

        # ---- Crear dataloader fresco con los parámetros fotométricos actuales
        log.info("  -> Instanciando DataLoader para %s...", exp_name)
        current_dataloader = _build_dataloader(current_cfg)

        # --- wandb: un run por experimento (try/finally garantiza finish) --
        with wandb_run(
            current_cfg,
            name=exp_name,
            tags=["degradation", cfg.dataset.name, base_model_name],
            group=run_group,  # <- Aquí usamos el grupo dinámico
            job_type="evaluation",
        ):
            try:
                runner = EvaluationRunner(current_cfg, dataloader=current_dataloader)

                # Reutilizar modelo si la config de modelo no ha cambiado
                exp_model_name: str = OmegaConf.select(
                    current_cfg, "model.name", default="hrnet_w32"
                )
                if exp_model_name == base_model_name:
                    runner.model = shared_model
                    log.info("  (reutilizando modelo compartido '%s')", base_model_name)

                # Debug limit: envolver dataloader con islice
                if debug_limit is not None:
                    import itertools
                    runner.dataloader = itertools.islice(current_dataloader, debug_limit)

                # Solo mass evaluation (Stage 1)
                df = runner.run_mass_evaluation()

                elapsed = time.perf_counter() - t0
                timings[exp_name] = elapsed

                if df is not None and not df.empty:
                    metrics = _extract_scalar_metrics(df, exp_name, overrides)
                    metrics["elapsed_s"] = round(elapsed, 2)
                    results_summary.append(metrics)

                    # --- wandb: log métricas escalares del experimento ---
                    log_metrics({
                        "oks_ours_mean": metrics["oks_ours_mean"],
                        "oks_base_mean": metrics["oks_base_mean"],
                        "oks_tta_mean": metrics["oks_tta_mean"],
                        "delta_oks_mean": metrics["delta_oks_mean"],
                        "delta_oks_tta_mean": metrics["delta_oks_tta_mean"],
                        "delta_oks_ours_over_tta_mean": metrics["delta_oks_ours_over_tta_mean"],
                        "delta_oks_proportional": metrics["delta_oks_mean"] / metrics["oks_base_mean"] if metrics["oks_base_mean"] > 0 else 0.0,
                        "nll_mean": metrics["nll_mean"],
                        "entropy_mean": metrics["entropy_mean"],
                        "covariance_vol_mean": metrics["covariance_vol_mean"],
                        "n_components_mean": metrics["n_components_mean"],
                        "swaps_base_total": metrics["swaps_base_total"],
                        "swaps_tta_total": metrics["swaps_tta_total"],
                        "swaps_ours_total": metrics["swaps_ours_total"],
                        "swaps_corrected": metrics["swaps_corrected"],
                        "swaps_introduced": metrics["swaps_introduced"],
                        "swaps_corrected_tta": metrics["swaps_corrected_tta"],
                        "swaps_introduced_tta": metrics["swaps_introduced_tta"],
                        "n_images": metrics["n_images"],
                        "elapsed_s": metrics["elapsed_s"],
                    })

                    log.info(
                        "  [OK] %s: OKS=%.4f (Δ=%+.4f) | NLL=%.4f | %.1fs",
                        exp_name,
                        metrics["oks_ours_mean"],
                        metrics["delta_oks_mean"],
                        metrics["nll_mean"],
                        elapsed,
                    )
                else:
                    log.warning("  [WARN] %s devolvió un DataFrame vacío.", exp_name)

            except Exception:
                log.exception("  [FAIL] Error crítico en experimento '%s'", exp_name)

            finally:
                # Limpieza de memoria (sin destruir el modelo compartido)
                if "runner" in dir():
                    del runner  # pragma: no cover
                gc.collect()
                _try_free_gpu()

    # ------------------------------------------------------------------
    # 4) Resumen final
    # ------------------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info("  BENCHMARK COMPLETADO")
    log.info("=" * 70)

    if not results_summary:
        log.error("Ningún experimento produjo resultados.")
        return

    df_results = pd.DataFrame(results_summary)

    # Ordenar por mejor OKS
    if "oks_ours_mean" in df_results.columns:
        df_results = df_results.sort_values("oks_ours_mean", ascending=False)

    # Mostrar tabla en consola
    display_cols = [
        "experiment", "n_images", "oks_ours_mean", "oks_base_mean",
        "delta_oks_mean", "nll_mean", "entropy_mean", "elapsed_s",
    ]
    display_cols = [c for c in display_cols if c in df_results.columns]
    print("\n" + "=" * 70)
    print("  RESULTADOS COMPARATIVOS")
    print("=" * 70)
    print(df_results[display_cols].to_string(index=False))
    print("=" * 70)

    # Guardar CSV (ruta absoluta, independiente del CWD de Hydra)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = comparison_dir / f"comparison_results_{timestamp}.csv"
    df_results.to_csv(csv_path, index=False)
    log.info("CSV guardado en: %s", csv_path)

    # Enlace simbólico / copia al nombre fijo para fácil acceso
    latest_path = comparison_dir / "comparison_results_latest.csv"
    try:
        if latest_path.exists():
            latest_path.unlink()
        df_results.to_csv(latest_path, index=False)
        log.info("Copia en:       %s", latest_path)
    except OSError:
        pass  # no es crítico

    # --- wandb: log tabla resumen global en un run dedicado ---------------
    with wandb_run(
        cfg,
        name="comparison_summary",
        tags=["summary", cfg.dataset.name],
        group="compare_experiments",
        job_type="summary",
    ):
        log_summary_table("comparison_results", df_results)
        log_metrics({
            "best_oks": float(df_results["oks_ours_mean"].max()),
            "best_experiment": df_results.iloc[0]["experiment"],
            "total_experiments": len(df_results),
        })


if __name__ == "__main__":
    main()

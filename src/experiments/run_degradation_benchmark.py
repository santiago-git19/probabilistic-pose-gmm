"""
Benchmark de Degradación y Evaluación de Incertidumbre

Este script:
1. Ejecuta iterativamente el modelo bajo niveles incrementales de degradación sintética (ruido, blur, contraste).
2. Recolecta las métricas GMM ($\Sigma$, $W_2^2$, $D_{KL}$, $\pi_{uniforme}$) para cada imagen y articulación.
3. Al finalizar, invoca a `evaluate_uncertainty.py` para generar AUSE, ECE, ROC y analizar fallos catastróficos.
"""

import sys
import copy
import logging
import itertools
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.pose_uncertainty.evaluation.runner import EvaluationRunner, _build_dataloader
from src.pose_uncertainty.models.adapters import create_model_adapter
from src.experiments.evaluate_uncertainty import generate_all_plots

log = logging.getLogger(__name__)

DEGRADATION_EXPERIMENTS = {
    "00_Baseline_Clean": {
        "dataset.noise_sigma": 0.0,
        "dataset.blur_sigma": 0.0,
        "dataset.contrast_factor": 1.0,
    },
    "01_Resize_Low": {
        "dataset.resize_scale": 0.5,
    },
    "02_Resize_Medium": {
        "dataset.resize_scale": 0.3,
    },
    "03_Resize_High": {
        "dataset.resize_scale": 0.1,
    },
    "04_Resize_Extreme": {
        "dataset.resize_scale": 0.05,
    },
    "05_Resize_Catastrophic": {
        "dataset.resize_scale": 0.01,
    },
}

@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    log.info("Iniciando Benchmark de Degradación Controlada")
    
    # Resolve output directory
    output_dir_str = OmegaConf.select(cfg, "logging.output_dir", default="outputs")
    base_out_dir = Path(output_dir_str) / "degradation_benchmark"
    base_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config to save inside benchmark folder
    with open_dict(cfg):
        cfg.logging.output_dir = str(base_out_dir)

    # Initialize model once
    model_name = OmegaConf.select(cfg, "model.name", default="hrnet_w32")
    device = OmegaConf.select(cfg, "model.device", default="cuda")
    model = create_model_adapter(model_name, device=device)
    
    all_results = []
    
    for exp_name, exp_overrides in DEGRADATION_EXPERIMENTS.items():
        log.info(f"=== Ejecutando experimento: {exp_name} ===")
        
        # Deepcopy config for isolation
        exp_cfg = copy.deepcopy(cfg)
        with open_dict(exp_cfg):
            for k, v in exp_overrides.items():
                OmegaConf.update(exp_cfg, k, v)
        
        # Re-build dataloader for this degradation
        dataloader = _build_dataloader(exp_cfg)
        
        debug_limit = OmegaConf.select(exp_cfg, "evaluation.debug_limit", default=None)
        if debug_limit is not None:
            dataloader = itertools.islice(dataloader, debug_limit)
            log.info(f"Limitando dataloader a {debug_limit} muestras por debug_limit.")
        
        # Run mass evaluation
        runner = EvaluationRunner(exp_cfg, dataloader=dataloader)
        runner.model = model # Re-use model
        
        try:
            df = runner.run_mass_evaluation()
            if not df.empty:
                df["experiment_name"] = exp_name
                for key, val in exp_overrides.items():
                    df[key] = val
                    
                parquet_path = base_out_dir / f"{exp_name}_results.parquet"
                df.to_parquet(parquet_path, index=False)
                log.info(f"Guardado {exp_name} en {parquet_path}")
                all_results.append(df)
        except Exception as e:
            log.error(f"Fallo en experimento {exp_name}: {e}")

    if not all_results:
        log.error("No se generaron resultados en ningún experimento.")
        return
        
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_parquet = base_out_dir / "all_degradations_combined.parquet"
    combined_df.to_parquet(combined_parquet, index=False)
    
    log.info("Ejecución finalizada. Generando gráficas de incertidumbre y diagnósticos...")
    generate_all_plots(combined_df, base_out_dir)
    log.info("Proceso completado exitosamente.")

if __name__ == "__main__":
    main()

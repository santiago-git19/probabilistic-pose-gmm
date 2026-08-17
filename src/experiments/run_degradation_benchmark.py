"""Controlled Degradation Benchmark and Uncertainty Profiling Runner.

This script:
1. Iteratively runs the model under incremental synthetic degradation levels (noise, blur, resolution scaling).
2. Collects GMM metrics (Sigma, W2, D_KL, pi_uniform) for each image and joint.
3. Automatically triggers organized evaluation and adaptive uncertainty optimization pipelines,
   generating AUSE, ECE, OoD ROC, and unified CSV/JSON master tables.
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
from src.experiments.degradation_organizer import run_organized_evaluation_pipeline, run_organized_optimization_pipeline

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
        "dataset.resize_scale": 0.25,
    },
    "03_Resize_High": {
        "dataset.resize_scale": 0.125,
    },
    "04_Resize_Extreme": {
        "dataset.resize_scale": 0.0625,
    }
}

@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    log.info("Starting Controlled Degradation Benchmark...")
    
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
        log.info(f"=== Running experiment: {exp_name} ===")
        
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
            log.info(f"Limiting dataloader to {debug_limit} samples via debug_limit.")
        
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
                log.info(f"Saved {exp_name} to {parquet_path}")
                all_results.append(df)
        except Exception as e:
            log.error(f"Failure in experiment {exp_name}: {e}")

    if not all_results:
        log.error("No results generated across any degradation experiments.")
        return
        
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_parquet = base_out_dir / "all_degradations_combined.parquet"
    combined_df.to_parquet(combined_parquet, index=False)
    
    log.info("Execution complete. Organizing and executing uncertainty evaluation and adaptive optimization...")
    beta = OmegaConf.select(cfg, "adaptive_uncertainty.beta", default=42.2103)
    strategy = OmegaConf.select(cfg, "adaptive_uncertainty.strategy", default="max_pooling")
    tau = OmegaConf.select(cfg, "adaptive_uncertainty.tau", default=0.5)
    
    log.info("--> 1/2: Executing evaluation pipeline (evaluate_uncertainty)...")
    run_organized_evaluation_pipeline(base_out_dir, beta=beta, strategy=strategy, tau=tau)
    
    log.info("--> 2/2: Executing adaptive optimization pipeline (optimize_adaptive_uncertainty)...")
    run_organized_optimization_pipeline(base_out_dir)
    
    log.info(f"Process completed successfully! All plots and master summary CSV/JSON files organized in: {base_out_dir / 'graficas'}")

if __name__ == "__main__":
    main()

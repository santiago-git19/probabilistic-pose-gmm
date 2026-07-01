"""Run complete evaluation pipeline: Mass Evaluation -> Diagnostics -> Deep Profiling.

Usage:
    # Full evaluation
    python src/experiments/run_benchmark.py
    
    # Quick test (limit to 10 images)
    python src/experiments/run_benchmark.py evaluation.debug_limit=10
    
    # Use different model
    python src/experiments/run_benchmark.py model=resnet50
    
    # Force re-run even if parquet exists
    python src/experiments/run_benchmark.py force_rerun=true
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.pose_uncertainty.evaluation.runner import EvaluationRunner
from src.pose_uncertainty.evaluation.diagnostics import select_focus_groups
from src.pose_uncertainty.tracking import wandb_run, log_metrics

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Three-stage evaluation pipeline:
    1. Mass Evaluation: scalar metrics for the full dataset -> results_metadata.parquet
    2. Diagnostics: select focus groups (Wins, Regressions, High_Uncertainty, Edge_Cases)
    3. Deep Profiling: capture full artefacts (images, heatmaps, GMM params) for selected IDs
    """
    log.info("=" * 70)
    log.info("ROBUST POSE ESTIMATION - EVALUATION PIPELINE")
    log.info("=" * 70)
    log.info("Model: %s", cfg.model.name)
    log.info("Dataset: %s", cfg.dataset.name)
    log.info("Output: %s", OmegaConf.select(cfg, "logging.output_dir"))
    
    # Handle debug limit
    debug_limit = OmegaConf.select(cfg, "evaluation.debug_limit", default=None)
    if debug_limit is not None:
        log.warning("DEBUG MODE: Limited to %d images", debug_limit)

    # --- wandb: un solo run para todo el benchmark -----------------------
    
    with wandb_run(
        cfg,
        name=f"benchmark_{cfg.model.name}_{cfg.dataset.name}",
        tags=["benchmark", cfg.dataset.name, cfg.model.name],
        job_type="benchmark",
    ):
    
        # ---------------------------------------------------------------------
        # Initialize Runner (builds model + dataloader internally)
        # ---------------------------------------------------------------------
        log.info("\n[1/3] Initializing Runner...")
        runner = EvaluationRunner(cfg)
        parquet_path = runner.output_dir / "results_metadata.parquet"
        
        # ---------------------------------------------------------------------
        # STAGE 1: Mass Evaluation (scalars only)
        # ---------------------------------------------------------------------
        force_rerun = cfg.get("force_rerun", False)
        
        if parquet_path.exists() and not force_rerun:
            log.info("\n[2/3] Mass Evaluation: SKIPPED (parquet exists)")
            log.info("      Use force_rerun=true to override")
            # Cargar parquet existente para poder loggear métricas igualmente
            import pandas as pd
            df = pd.read_parquet(parquet_path)
        else:
            log.info("\n[2/3] Mass Evaluation: RUNNING...")
            log.info("      This may take several minutes...")
            
            # Apply debug limit if set
            if debug_limit is not None:
                import itertools
                runner.dataloader = itertools.islice(runner.dataloader, debug_limit)
            
            df = runner.run_mass_evaluation()
            log.info("      [OK] Saved %d results to %s", len(df), parquet_path.name)

        # --- wandb: log métricas escalares agregadas ----------------------
        if df is not None and not df.empty:
            oks_base_mean = float(df["oks_base"].mean()) if "oks_base" in df.columns else 0.0
            delta_oks_mean = float(df["delta_oks"].mean()) if "delta_oks" in df.columns else 0.0
            
            swaps_base_total = int(df["swaps_base"].sum()) if "swaps_base" in df.columns else 0
            swaps_ours_total = int(df["swaps_ours"].sum()) if "swaps_ours" in df.columns else 0
            swaps_corrected = int(df["swaps_corrected"].sum()) if "swaps_corrected" in df.columns else 0
            swaps_introduced = int(df["swaps_introduced"].sum()) if "swaps_introduced" in df.columns else 0
                
            log_metrics({
                "oks_ours_mean": float(df["oks_ours"].mean()) if "oks_ours" in df.columns else 0.0,
                "oks_base_mean": oks_base_mean,
                "delta_oks_mean": delta_oks_mean,
                "delta_oks_proportional": delta_oks_mean / oks_base_mean if oks_base_mean > 0 else 0.0,
                "nll_mean": float(df["nll"].mean()) if "nll" in df.columns else 0.0,
                "entropy_mean": float(df["entropy"].mean()) if "entropy" in df.columns else 0.0,
                "covariance_vol_mean": float(df["covariance_vol"].mean()) if "covariance_vol" in df.columns else 0.0,
                "swaps_base_total": swaps_base_total,
                "swaps_ours_total": swaps_ours_total,
                "swaps_corrected": swaps_corrected,
                "swaps_introduced": swaps_introduced,
                "n_images": len(df),
            })
        
        # ---------------------------------------------------------------------
        # STAGE 2: Diagnostics (select focus groups)
        # ---------------------------------------------------------------------
        log.info("\n[3/3] Diagnostics: Selecting Focus Groups...")
        
        # Extract only the needed config sections to avoid interpolation errors
        config_dict = {
            'evaluation': OmegaConf.to_container(cfg.evaluation, resolve=True),
            'dataset': OmegaConf.to_container(cfg.dataset, resolve=True),
            'paths': OmegaConf.to_container(cfg.paths, resolve=True)
        }
        focus_groups = select_focus_groups(str(parquet_path), config_dict)
        
        total_selected = sum(len(ids) for ids in focus_groups.values())
        log.info("      Selected %d images across %d groups:", total_selected, len(focus_groups))
        for group_name, ids in focus_groups.items():
            log.info("        - %-30s: %3d images", group_name, len(ids))

        # --- wandb: log número de imágenes por focus group -----------------
        log_metrics(
            {f"focus_group/{gn}": len(ids) for gn, ids in focus_groups.items()}
        )
        
        # ---------------------------------------------------------------------
        # STAGE 3: Deep Profiling (full artefacts for selected images)
        # ---------------------------------------------------------------------
        if total_selected == 0:
            log.warning("\n[WARNING] No images selected for Deep Profiling.")
            log.warning("   Try adjusting thresholds in diagnostics.py or running more samples.")
            return
    
        log.info("\n[4/4] Deep Profiling: Capturing full artefacts...")
        log.info("      This will save ~%.1f MB per image (compressed)", 2.5)
        
        saved_paths = runner.run_deep_profiling(focus_groups)
        log.info("      [OK] Saved %d analysis packets", len(saved_paths))

        # --- wandb: log deep profiling stats ---
        log_metrics({"deep_profiling/n_packets": len(saved_paths)})
        
        # ---------------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------------
        log.info("\n" + "=" * 70)
        log.info("EVALUATION COMPLETE")
        log.info("=" * 70)
        log.info("Output directory: %s", runner.output_dir)
        log.info("  - results_metadata.parquet  <- scalar metrics")
        log.info("  - focus_groups_ids.json     <- selected IDs")
        log.info("  - *.pkl.gz                  <- deep analysis packets")
        log.info("\nNext step: Launch FiftyOne visualization")
        log.info("  python src/experiments/launch_viz.py")
        log.info("=" * 70)


if __name__ == "__main__":
    main()
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
    
    # -------------------------------------------------------------------------
    # Initialize Runner (builds model + dataloader internally)
    # -------------------------------------------------------------------------
    log.info("\n[1/3] Initializing Runner...")
    runner = EvaluationRunner(cfg)
    parquet_path = runner.output_dir / "results_metadata.parquet"
    
    # -------------------------------------------------------------------------
    # STAGE 1: Mass Evaluation (scalars only)
    # -------------------------------------------------------------------------
    force_rerun = cfg.get("force_rerun", False)
    
    if parquet_path.exists() and not force_rerun:
        log.info("\n[2/3] Mass Evaluation: SKIPPED (parquet exists)")
        log.info("      Use force_rerun=true to override")
    else:
        log.info("\n[2/3] Mass Evaluation: RUNNING...")
        log.info("      This may take several minutes...")
        
        # Apply debug limit if set
        if debug_limit is not None:
            import itertools
            runner.dataloader = itertools.islice(runner.dataloader, debug_limit)
        
        df = runner.run_mass_evaluation()
        log.info("      [OK] Saved %d results to %s", len(df), parquet_path.name)
    
    # -------------------------------------------------------------------------
    # STAGE 2: Diagnostics (select focus groups)
    # -------------------------------------------------------------------------
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
        log.info("        - %-20s: %3d images", group_name, len(ids))
    
    # -------------------------------------------------------------------------
    # STAGE 3: Deep Profiling (full artefacts for selected images)
    # -------------------------------------------------------------------------
    if total_selected == 0:
        log.warning("\n[WARNING] No images selected for Deep Profiling.")
        log.warning("   Try adjusting thresholds in diagnostics.py or running more samples.")
        return
    
    log.info("\n[4/4] Deep Profiling: Capturing full artefacts...")
    log.info("      This will save ~%.1f MB per image (compressed)", 2.5)
    
    saved_paths = runner.run_deep_profiling(focus_groups)
    log.info("      [OK] Saved %d analysis packets", len(saved_paths))
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
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
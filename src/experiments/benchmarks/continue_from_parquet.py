"""Continue evaluation from existing parquet file.

Usage:
    python src/experiments/benchmarks/continue_from_parquet.py parquet_path="outputs/2026-02-17/23-00-36/results_metadata.parquet"
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.pose_uncertainty.evaluation.runner import EvaluationRunner
from src.pose_uncertainty.evaluation.diagnostics import select_focus_groups

log = logging.getLogger(__name__)


@hydra.main(config_path="../../../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Continue evaluation and deep profiling from existing parquet file."""
    
    # Path to existing parquet (configurable via Hydra override)
    parquet_arg = cfg.get("parquet_path", None)
    if parquet_arg:
        parquet_path = Path(parquet_arg)
    else:
        # Search for latest parquet file under outputs/ if not specified
        parquets = sorted(Path("outputs").glob("**/results_metadata.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
        if parquets:
            parquet_path = parquets[0]
            log.info("No 'parquet_path' specified. Automatically selected latest run: %s", parquet_path)
        else:
            log.error("No parquet file specified and none found in 'outputs/'. Please supply parquet_path=...")
            return
    
    if not parquet_path.exists():
        log.error("Parquet file not found: %s", parquet_path)
        return
    
    log.info("=" * 70)
    log.info("CONTINUING EVALUATION FROM EXISTING PARQUET")
    log.info("=" * 70)
    log.info("Parquet: %s", parquet_path)
    
    # -------------------------------------------------------------------------
    # STAGE 1: Diagnostics (select focus groups)
    # -------------------------------------------------------------------------
    log.info("\n[1/2] Diagnostics: Selecting Focus Groups...")
    
    # Extract only the needed config sections
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
    
    if total_selected == 0:
        log.warning("\n[WARNING] No images selected for Deep Profiling.")
        return
    
    # -------------------------------------------------------------------------
    # STAGE 2: Deep Profiling (full artefacts for selected images)
    # -------------------------------------------------------------------------
    log.info("\n[2/2] Deep Profiling: Capturing full artefacts...")
    
    # Initialize runner (needed for deep profiling)
    runner = EvaluationRunner(cfg)
    runner.output_dir = parquet_path.parent  # Use same output directory
    
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
    log.info("  python src/experiments/visualizations/launch_viz.py data_dir=%s", runner.output_dir)
    log.info("=" * 70)


if __name__ == "__main__":
    main()

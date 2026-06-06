"""Launch FiftyOne web UI to visualize Deep Profiling results.

Usage:
    # Launch visualization (will auto-detect latest output)
    python src/experiments/launch_viz.py
    
    # Specify output directory manually
    python src/experiments/launch_viz.py data_dir="outputs/2026-02-09/14-30-00"
    
    # Use different dataset name
    python src/experiments/launch_viz.py dataset_name="MyCustomEval"
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.pose_uncertainty.visualization.fiftyone_loader import create_evaluation_dataset

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Load .pkl.gz packets from Deep Profiling and visualize in FiftyOne.
    
    The web app allows:
    - Filtering by sample_type (Wins, Regressions, High_Uncertainty, Edge_Cases)
    - Sorting by metrics (oks_delta, entropy, nll)
    - Side-by-side comparison of GT / baseline / ours predictions
    - Uncertainty ellipses overlaid on keypoints
    """
    log.info("=" * 70)
    log.info("FIFTYONE VISUALIZATION LAUNCHER")
    log.info("=" * 70)
    
    # ---- Determine data directory ---------------------------------------------
    data_dir_cfg = "outputs\\2026-06-04\\12-33-24"#"outputs\\2026-02-23\\22-58-26"#"outputs\\2026-02-22\\21-53-46"#"outputs\\2026-02-18\\18-51-15"#"outputs\\2026-02-17\\23-00-36"#"outputs\\2026-02-16\\16-19-51"#"outputs\\2026-02-16\\15-47-00"#"outputs\\2026-02-16\\01-16-24"#"outputs\\2026-02-12\\18-08-52" # "outputs\\2026-02-12\\16-35-10"#"outputs\\2026-02-11\\12-40-42"# cfg.get("data_dir", None)
    
    if data_dir_cfg is not None:
        # User explicitly provided path
        data_dir = Path(data_dir_cfg)
    else:
        # Auto-detect: use logging.output_dir from config
        output_dir_str = OmegaConf.select(cfg, "logging.output_dir")
        if output_dir_str is None:
            log.error("No output directory found in config.")
            log.error("Either run run_benchmark.py first, or specify: data_dir=<path>")
            sys.exit(1)
        data_dir = Path(output_dir_str)
    
    # Check existence
    if not data_dir.exists():
        log.error("Directory not found: %s", data_dir)
        log.error("Please run run_benchmark.py first to generate Deep Profiling data.")
        sys.exit(1)
    
    # Check for .pkl.gz files
    pkl_files = list(data_dir.glob("*.pkl.gz"))
    if not pkl_files:
        log.error("No .pkl.gz files found in %s", data_dir)
        log.error("Make sure Deep Profiling completed successfully.")
        sys.exit(1)
    
    log.info("Data directory: %s", data_dir)
    log.info("Found %d analysis packets", len(pkl_files))
    
    # ---- Launch FiftyOne --------------------------------------------------
    dataset_name = cfg.get("dataset_name", f"TFG_Eval_{cfg.model.name}_resol_0.125")
    overwrite_flag = cfg.get("overwrite", True)
    log.info("Creating FiftyOne dataset: %s", dataset_name)
    log.info("This may take a minute (decompressing images)...\n")
    
    try:
        dataset = create_evaluation_dataset(
            data_dir=str(data_dir),
            dataset_name=dataset_name,
            overwrite=overwrite_flag
        )
        log.info("\n" + "=" * 70)
        log.info("[OK] FiftyOne app launched successfully!")
        log.info("=" * 70)
        log.info("The web UI should open in your browser.")
        log.info("If not, navigate to: http://localhost:5151")
        log.info("\nUseful filters:")
        log.info("  - Tags -> Select sample_type (Wins, Regressions, ...)")
        log.info("  - Sort by: oks_delta, entropy, nll")
        log.info("  - View fields: ground_truth (green), prediction_base (red), prediction_ours (blue)")
        log.info("\nPress Ctrl+C to stop the server.")
        log.info("=" * 70)
        
        # Keep script alive so FiftyOne server stays up
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        log.info("\nShutting down FiftyOne server...")
    except Exception as e:
        log.exception("Error launching FiftyOne: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
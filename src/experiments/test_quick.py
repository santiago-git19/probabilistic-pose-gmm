"""Quick test script to verify the evaluation pipeline works end-to-end.

This runs a minimal test on 5 images to check:
1. Model loading
2. TTA + sampling + GMM pipeline
3. Metrics computation
4. Storage compression/decompression

Usage:
    python src/experiments/test_quick.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.pose_uncertainty.evaluation.runner import EvaluationRunner
from src.pose_uncertainty.evaluation.diagnostics import select_focus_groups
from src.pose_uncertainty.evaluation import storage

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Quick smoke test: process 5 images, save 2 deep packets, verify load.
    """
    log.info("=" * 70)
    log.info("QUICK TEST - Evaluation Pipeline")
    log.info("=" * 70)
    
    # Force small test set
    OmegaConf.update(cfg, "evaluation.debug_limit", 5, force_add=True)
    OmegaConf.update(cfg, "evaluation.n_samples_per_group", 2, force_add=True)
    
    log.info("\n[TEST 1/4] Initialize Runner...")
    try:
        runner = EvaluationRunner(cfg)
        log.info("✓ Runner initialized")
    except Exception as e:
        log.error("✗ Failed to initialize runner: %s", e)
        raise
    
    log.info("\n[TEST 2/4] Run Mass Evaluation (5 images)...")
    try:
        import itertools
        runner.dataloader = itertools.islice(runner.dataloader, 5)
        df = runner.run_mass_evaluation()
        log.info("✓ Mass evaluation complete: %d rows", len(df))
        assert len(df) > 0, "No results generated"
        assert "oks_base" in df.columns, "Missing oks_base column"
        assert "oks_ours" in df.columns, "Missing oks_ours column"
    except Exception as e:
        log.error("✗ Mass evaluation failed: %s", e)
        raise
    
    log.info("\n[TEST 3/4] Select Focus Groups...")
    try:
        parquet_path = runner.output_dir / "results_metadata.parquet"
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        focus_groups = select_focus_groups(str(parquet_path), config_dict)
        log.info("✓ Focus groups selected: %s", 
                 {k: len(v) for k, v in focus_groups.items()})
        total = sum(len(v) for v in focus_groups.values())
        assert total > 0, "No images selected for deep profiling"
    except Exception as e:
        log.error("✗ Focus group selection failed: %s", e)
        raise
    
    log.info("\n[TEST 4/4] Run Deep Profiling + Storage...")
    try:
        saved_paths = runner.run_deep_profiling(focus_groups)
        log.info("✓ Deep profiling complete: %d packets saved", len(saved_paths))
        assert len(saved_paths) > 0, "No packets saved"
        
        # Test load
        test_path = saved_paths[0]
        loaded_packet = storage.load_deep_analysis(str(test_path))
        log.info("✓ Storage load verified")
        
        # Check packet structure
        assert "meta" in loaded_packet, "Missing meta"
        assert "metrics" in loaded_packet, "Missing metrics"
        assert "tta_data" in loaded_packet, "Missing tta_data"
        assert "gmm_model" in loaded_packet, "Missing gmm_model"
        log.info("✓ Packet structure valid")
        
    except Exception as e:
        log.error("✗ Deep profiling/storage failed: %s", e)
        raise
    
    log.info("\n" + "=" * 70)
    log.info("ALL TESTS PASSED ✓")
    log.info("=" * 70)
    log.info("Output directory: %s", runner.output_dir)
    log.info("\nYou can now run the full evaluation with:")
    log.info("  python src/experiments/run_benchmark.py")
    log.info("=" * 70)


if __name__ == "__main__":
    main()

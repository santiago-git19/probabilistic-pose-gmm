"""Comparative Experiment Benchmarking Script.

Allows defining multiple configurations and executing them sequentially
to compare metrics (A/B testing, manual ablation sweeps, etc.).

Design Principles:
    - Immutability: Each experiment starts from a deepcopy of the base
      configuration; overrides only affect the current experiment.
    - Efficiency: DataLoader and model are instantiated once and reused
      across experiments (unless model architecture changes).
    - Robustness: If an experiment fails, the script logs the error and
      continues with subsequent runs.

Usage::

    cd <project_root>
    python src/experiments/ablations/compare_experiments.py

    # Quick test with debug limit:
    python src/experiments/ablations/compare_experiments.py evaluation.debug_limit=20
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
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.pose_uncertainty.evaluation.runner import EvaluationRunner, _build_dataloader
from src.pose_uncertainty.models.adapters import create_model_adapter
from src.pose_uncertainty.tracking import wandb_run, log_metrics, log_summary_table

log = logging.getLogger(__name__)

# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================
# Format: "Experiment_Name": { "nested.hydra.key": value, ... }
#
# Each sub-dictionary contains ONLY parameters differing from config.yaml.
# Everything else remains intact.
#
# TTA ABLATION STUDY - Evaluates individual and combined components
# =============================================================================

EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "resolution_0.5": {"dataset.resize_scale": 0.5, "wandb_group": "OCHUMAN_resolution"},
    "resolution_0.25": {"dataset.resize_scale": 0.25, "wandb_group": "OCHUMAN_resolution"},
    "resolution_0.125": {"dataset.resize_scale": 0.125, "wandb_group": "OCHUMAN_resolution"},
}


# =============================================================================
# ████  END OF EXPERIMENT DEFINITIONS  ████
# =============================================================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    """Sanitize experiment name for filesystem paths."""
    return (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "-")
        .replace("\\", "-")
    )


def _set_nested(cfg: DictConfig, dotted_key: str, value: Any) -> None:
    """Set ``cfg[a][b][c] = value`` given dot-separated key ``'a.b.c'``.

    Creates intermediate nodes if they do not exist (within ``open_dict``).
    """
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        child = OmegaConf.select(node, part, default=None)
        if child is None:
            # Create intermediate dictionary node
            OmegaConf.update(node, part, {})
            child = OmegaConf.select(node, part)
        node = child
    OmegaConf.update(node, parts[-1], value)


def apply_overrides(cfg: DictConfig, overrides: Dict[str, Any]) -> None:
    """Apply dictionary of dot-separated overrides to Hydra configuration *cfg*.

    Uses ``open_dict`` to allow mutations on structured configs and
    ``OmegaConf.update`` to safely traverse the hierarchy.
    """
    with open_dict(cfg):
        for key, val in overrides.items():
            try:
                _set_nested(cfg, key, val)
                log.info("  -> Override: %s = %s", key, val)
            except Exception as exc:
                log.error("  [ERROR] Failed to apply override '%s': %s", key, exc)


def _extract_scalar_metrics(
    df: pd.DataFrame, experiment_name: str, overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract aggregate scalar metrics from an evaluation results DataFrame.

    Tolerates missing columns gracefully (returns ``NaN``).
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
    """Release GPU cache if PyTorch CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../../../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    log.info("=" * 70)
    log.info("  EXPERIMENT COMPARISON BENCHMARK")
    log.info("  %d experiments defined", len(EXPERIMENTS))
    log.info("=" * 70)

    # Project root directory (absolute path)
    project_root = _PROJECT_ROOT
    comparison_dir = project_root / "outputs" / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Shared resource: DataLoader (single instance if configs match)
    # ------------------------------------------------------------------
    # Debug limit (identical to run_benchmark.py)
    debug_limit: Optional[int] = OmegaConf.select(
        cfg, "evaluation.debug_limit", default=None
    )
    if debug_limit is not None:
        log.warning(
            "DEBUG MODE: Each experiment will process at most %d images.",
            debug_limit,
        )

    # ------------------------------------------------------------------
    # 2) Shared resource: Model adapter (reused across runs)
    # ------------------------------------------------------------------
    base_model_name: str = OmegaConf.select(cfg, "model.name", default="hrnet_w32")
    base_device: str = OmegaConf.select(cfg, "model.device", default="cuda")
    log.info("[SETUP] Loading model '%s' on '%s' ...", base_model_name, base_device)
    shared_model = create_model_adapter(base_model_name, device=base_device)

    # ------------------------------------------------------------------
    # 3) Experiment Loop
    # ------------------------------------------------------------------
    results_summary: List[Dict[str, Any]] = []
    timings: Dict[str, float] = {}

    for exp_idx, (exp_name, overrides) in enumerate(
        tqdm(EXPERIMENTS.items(), desc="Experiments", unit="exp"),
        start=1
    ):
        log.info("")
        log.info("-" * 60)
        log.info(
            ">>> [%d/%d] EXPERIMENT: %s", exp_idx, len(EXPERIMENTS), exp_name,
        )
        log.info("-" * 60)

        # ---- Safely clone base config ---------------------------------
        current_cfg = copy.deepcopy(cfg)

        # Isolated output directory for this experiment
        safe_name = _sanitize_name(exp_name)
        exp_output = comparison_dir / safe_name
        with open_dict(current_cfg):
            OmegaConf.update(current_cfg, "logging.output_dir", str(exp_output))

        # ---- Extract W&B group and sanitize overrides ----------------
        clean_overrides = overrides.copy()
        run_group = clean_overrides.pop("wandb_group", "compare_experiments")

        # ---- Apply exclusive overrides for this experiment ------------
        apply_overrides(current_cfg, clean_overrides)

        # ---- Execute evaluation ---------------------------------------
        t0 = time.perf_counter()

        # ---- Build fresh dataloader for current augmentation/degradation settings
        log.info("  -> Instantiating DataLoader for %s...", exp_name)
        current_dataloader = _build_dataloader(current_cfg)

        # --- W&B run tracking (try/finally ensures clean termination) --
        with wandb_run(
            current_cfg,
            name=exp_name,
            tags=["degradation", cfg.dataset.name, base_model_name],
            group=run_group,
            job_type="evaluation",
        ):
            try:
                runner = EvaluationRunner(current_cfg, dataloader=current_dataloader)

                # Reuse shared model if model config is identical
                exp_model_name: str = OmegaConf.select(
                    current_cfg, "model.name", default="hrnet_w32"
                )
                if exp_model_name == base_model_name:
                    runner.model = shared_model
                    log.info("  (reusing shared model '%s')", base_model_name)

                # Debug limit: wrap dataloader with islice
                if debug_limit is not None:
                    import itertools
                    runner.dataloader = itertools.islice(current_dataloader, debug_limit)

                # Mass evaluation stage
                df = runner.run_mass_evaluation()

                elapsed = time.perf_counter() - t0
                timings[exp_name] = elapsed

                if df is not None and not df.empty:
                    metrics = _extract_scalar_metrics(df, exp_name, overrides)
                    metrics["elapsed_s"] = round(elapsed, 2)
                    results_summary.append(metrics)

                    # --- W&B scalar metrics log ---
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
                    log.warning("  [WARN] %s returned an empty DataFrame.", exp_name)

            except Exception:
                log.exception("  [FAIL] Critical failure in experiment '%s'", exp_name)

            finally:
                # Memory cleanup (without destroying shared model)
                if "runner" in dir():
                    del runner  # pragma: no cover
                gc.collect()
                _try_free_gpu()

    # ------------------------------------------------------------------
    # 4) Final Summary
    # ------------------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info("  BENCHMARK COMPLETED")
    log.info("=" * 70)

    if not results_summary:
        log.error("No experiment produced valid results.")
        return

    df_results = pd.DataFrame(results_summary)

    # Sort by highest OKS
    if "oks_ours_mean" in df_results.columns:
        df_results = df_results.sort_values("oks_ours_mean", ascending=False)

    # Console summary table
    display_cols = [
        "experiment", "n_images", "oks_ours_mean", "oks_base_mean",
        "delta_oks_mean", "nll_mean", "entropy_mean", "elapsed_s",
    ]
    display_cols = [c for c in display_cols if c in df_results.columns]
    print("\n" + "=" * 70)
    print("  COMPARATIVE RESULTS SUMMARY")
    print("=" * 70)
    print(df_results[display_cols].to_string(index=False))
    print("=" * 70)

    # Save CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = comparison_dir / f"comparison_results_{timestamp}.csv"
    df_results.to_csv(csv_path, index=False)
    log.info("CSV saved to: %s", csv_path)

    # Fixed latest copy
    latest_path = comparison_dir / "comparison_results_latest.csv"
    try:
        if latest_path.exists():
            latest_path.unlink()
        df_results.to_csv(latest_path, index=False)
        log.info("Latest copy:  %s", latest_path)
    except OSError:
        pass

    # --- W&B global summary table in dedicated run ---
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

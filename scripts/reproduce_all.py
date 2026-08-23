"""
Master One-Click Paper Reproduction Suite (FAIR Compliant).

Orchestrates the complete reproduction pipeline for all qualitative figures,
uncertainty calibration curves, ablation studies, and computational benchmarks.

Usage:
    # Reproduce everything with one command:
    poetry run python scripts/reproduce_all.py --all

    # Download/verify pre-computed benchmark data:
    poetry run python scripts/reproduce_all.py --download-data

    # Reproduce only qualitative figures (Figs 1-5 + Methodology):
    poetry run python scripts/reproduce_all.py --figures

    # Reproduce only quantitative uncertainty figures (ECE, OoD KDE, 3x3 Matrices):
    poetry run python scripts/reproduce_all.py --uncertainty

    # Run computational cost benchmark (FPS, Latency, VRAM):
    poetry run python scripts/reproduce_all.py --benchmarks
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_EXPERIMENTS = PROJECT_ROOT / "src" / "experiments"


def run_stage(title: str, cmd: List[str]) -> bool:
    """Execute a reproduction stage subprocess with timing and error handling."""
    logger.info("=" * 80)
    logger.info("STAGE: %s", title.upper())
    logger.info("COMMAND: %s", " ".join(cmd))
    logger.info("=" * 80)

    t0 = time.perf_counter()
    try:
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        elapsed = time.perf_counter() - t0
        logger.info("[SUCCESS] Finished '%s' in %.2f seconds.\n", title, elapsed)
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - t0
        logger.error("[FAILED] '%s' exited with code %d after %.2f seconds: %s\n", title, e.returncode, elapsed, e)
        return False
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error("[ERROR] Unexpected exception in '%s': %s\n", title, e)
        return False


def ensure_benchmark_data() -> bool:
    """Ensure benchmark parquet data is available in outputs/data/."""
    from scripts.download_benchmark_data import check_data_complete, download_and_extract, DATA_ROOT

    if check_data_complete(DATA_ROOT):
        logger.info("[DATA CHECK] All required benchmark parquet files found in %s", DATA_ROOT)
        return True
    
    logger.info("[DATA CHECK] Missing benchmark parquet files. Running automated downloader...")
    return download_and_extract(target_dir=DATA_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Master One-Click Reproduction Pipeline for Robust Pose TTA.")
    parser.add_argument("--all", action="store_true", help="Run the full reproduction pipeline (data, figures, uncertainty, benchmarks)")
    parser.add_argument("--download-data", action="store_true", help="Ensure all pre-computed benchmark parquets are present")
    parser.add_argument("--figures", action="store_true", help="Reproduce qualitative figures (Figs 1-5 and Methodology)")
    parser.add_argument("--uncertainty", action="store_true", help="Reproduce quantitative uncertainty figures (ECE, OoD KDE, 3x3 Matrices, Beta, Boxplots)")
    parser.add_argument("--benchmarks", action="store_true", help="Run computational cost and runtime latency benchmarking")
    parser.add_argument("--test", action="store_true", help="Run pytest unit and integration test suite")
    parser.add_argument("--python-exec", type=str, default=sys.executable, help="Python executable to use (defaults to current)")

    args = parser.parse_args()
    py_exec = args.python_exec

    # If no flags specified, show help or run all
    if not any([args.all, args.download_data, args.figures, args.uncertainty, args.benchmarks, args.test]):
        logger.info("No specific stage flag provided. Defaulting to --figures and --uncertainty.")
        args.figures = True
        args.uncertainty = True

    overall_start = time.perf_counter()
    stages_run: List[Tuple[str, bool]] = []

    # 1. Tests
    if args.all or args.test:
        success = run_stage("Test Suite (PyTest)", [py_exec, "-m", "pytest", "src/tests", "--ignore=models/mmpose", "-q"])
        stages_run.append(("PyTest Suite", success))

    # 2. Benchmark Parquet Data
    if args.all or args.download_data or args.uncertainty:
        data_ok = ensure_benchmark_data()
        stages_run.append(("Benchmark Data Verification", data_ok))
        if not data_ok:
            logger.warning("[WARNING] Benchmark data is incomplete. Some quantitative plots may be skipped.")

    # 3. Qualitative Figures
    if args.all or args.figures:
        # Methodology Figure
        s1 = run_stage(
            "Methodology Pipeline Figure (Block 1-4 Overview)",
            [py_exec, str(SRC_EXPERIMENTS / "visualizations" / "generate_methodology_pipeline_figure.py")]
        )
        stages_run.append(("Methodology Pipeline Figure", s1))

        # Qualitative Figures 1 to 5
        s2 = run_stage(
            "Qualitative Paper Figures (Figs 1-5)",
            [py_exec, str(SRC_EXPERIMENTS / "visualizations" / "generate_all_paper_figures.py")]
        )
        stages_run.append(("Qualitative Figures 1-5", s2))

    # 4. Quantitative Uncertainty Figures
    if args.all or args.uncertainty:
        # ECE Calibration Plots
        s3 = run_stage(
            "ECE Calibration Plots",
            [py_exec, str(SRC_EXPERIMENTS / "uncertainty" / "generate_ece_plots.py")]
        )
        stages_run.append(("ECE Calibration Plots", s3))

        # OoD KDE Distribution Plots
        s4 = run_stage(
            "OoD KDE Anomaly Detection Plots",
            [py_exec, str(SRC_EXPERIMENTS / "uncertainty" / "generate_ood_kde_plots.py")]
        )
        stages_run.append(("OoD KDE Plots", s4))

        # 3x3 Matrices (ROC, AUSE, Correlation)
        s5 = run_stage(
            "Publication 3x3 Matrices",
            [py_exec, str(SRC_EXPERIMENTS / "uncertainty" / "plot_3x3_matrices.py"), "--models", "hrnet_w32", "resnet50", "vitpose_small"]
        )
        stages_run.append(("3x3 Publication Matrices", s5))

        # Beta Hyperparameter Optimization
        s6 = run_stage(
            "Beta Optimization 1x3 Composite Figure",
            [py_exec, str(SRC_EXPERIMENTS / "uncertainty" / "plot_1x3_beta_optimization.py")]
        )
        stages_run.append(("Beta Optimization Figure", s6))

        # Uniform Weight Boxplot
        s7 = run_stage(
            "Uniform Weight Boxplot 1x3 Figure",
            [py_exec, str(SRC_EXPERIMENTS / "uncertainty" / "plot_1x3_uniform_weight_boxplot.py")]
        )
        stages_run.append(("Uniform Weight Boxplot", s7))

        # Baseline Precision vs Ours Across Degradations
        s8 = run_stage(
            "Phase I Baseline Precision vs Ours Degradation Plot",
            [py_exec, str(SRC_EXPERIMENTS / "uncertainty" / "plot_baseline_degradation_precision.py")]
        )
        stages_run.append(("Baseline Precision Degradation Plot", s8))

        # OoD AUROC Evolution Across Degradations
        s9 = run_stage(
            "OoD AUROC Degradation Evolution 1x3 Plots",
            [py_exec, str(SRC_EXPERIMENTS / "uncertainty" / "plot_ood_degradation_evolution.py"), "--models", "hrnet_w32", "resnet50", "vitpose_small"]
        )
        stages_run.append(("OoD AUROC Degradation Evolution Plots", s9))

    # 5. Computational Cost Benchmark
    if args.all or args.benchmarks:
        s10 = run_stage(
            "Computational Cost & Latency Benchmark",
            [py_exec, str(SRC_EXPERIMENTS / "benchmarks" / "benchmark_computational_cost.py"), "--models", "hrnet_w32", "resnet50", "vitpose_small", "--num-trials", "15"]
        )
        stages_run.append(("Computational Cost Benchmark", s10))

    # Summary Report
    total_elapsed = time.perf_counter() - overall_start
    print("\n" + "=" * 80)
    print(" PAPER REPRODUCTION SUMMARY REPORT")
    print("=" * 80)
    for name, success in stages_run:
        status_str = "[PASSED]" if success else "[FAILED]"
        print(f"  {status_str:<10} | {name}")
    print("-" * 80)
    print(f"Total Execution Time: {total_elapsed:.2f} seconds ({total_elapsed / 60:.2f} minutes)")
    print(f"All generated artifacts saved to: {PROJECT_ROOT / 'outputs' / 'figures'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

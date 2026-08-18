"""Degradation Plot Organizer Script.

Generates uncertainty evaluation plots for each degradation level (resolution),
organizes all outputs into hierarchical subdirectories by degradation and metric
(AUC_*, ECE_*), and copies global catastrophic failure boxplots to root.

Uses evaluation routines from evaluate_uncertainty.py.
"""
import argparse
import logging
import shutil
import sys
from pathlib import Path
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.uncertainty.evaluate_uncertainty import generate_all_plots

log = logging.getLogger(__name__)

def generate_organized_plots(input_dir: Path, beta: float = 42.2103, strategy: str = "max_pooling", tau: float = 0.5):
    """Generate hierarchical plots for all degradations and organize into metric folders."""
    input_dir = Path(input_dir).resolve()
    if not input_dir.exists():
        log.error(f"Directory {input_dir} does not exist.")
        return

    graficas_dir = input_dir / "graficas"
    graficas_dir.mkdir(parents=True, exist_ok=True)

    # 1. Search individual degradation parquet files
    parquets = sorted(list(input_dir.glob("*_results.parquet")))
    if not parquets:
        parquets = [p for p in sorted(list(input_dir.glob("*.parquet"))) if "metadata" not in p.name and "combined" not in p.name]

    if not parquets:
        log.warning("No individual degradation parquet files found.")
    else:
        log.info(f"Found {len(parquets)} individual parquet files. Generating plots in {graficas_dir} ...")

    for p in parquets:
        deg_name = p.stem.replace("_results", "")
        log.info(f"--> Processing resolution / degradation: {deg_name} ({p.name})")
        
        df = pd.read_parquet(p)
        if df.empty:
            log.warning(f"File {p.name} is empty. Skipping.")
            continue
            
        subfolder = graficas_dir / deg_name
        subfolder.mkdir(parents=True, exist_ok=True)
        
        # Generate plots using evaluate_uncertainty.py
        generate_all_plots(df, subfolder, beta=beta, strategy=strategy, tau=tau)
        
        # Rename files in subfolder to include degradation prefix
        for img_path in list(subfolder.glob("*.png")):
            if not img_path.name.startswith(f"{deg_name}_"):
                new_name = f"{deg_name}_{img_path.name}"
                img_path.rename(subfolder / new_name)

    # 2. Process general (combined) dataset
    combined_parquet = input_dir / "all_degradations_combined.parquet"
    if combined_parquet.exists():
        log.info(f"--> Processing general dataset: General ({combined_parquet.name})")
        df_general = pd.read_parquet(combined_parquet)
        general_folder = graficas_dir / "General"
        general_folder.mkdir(parents=True, exist_ok=True)
        
        if not df_general.empty:
            generate_all_plots(df_general, general_folder, beta=beta, strategy=strategy, tau=tau)
            
            for img_path in list(general_folder.glob("*.png")):
                if not img_path.name.startswith("General_"):
                    new_name = f"General_{img_path.name}"
                    img_path.rename(general_folder / new_name)
    else:
        log.warning(f"Did not find {combined_parquet.name} to generate 'General' plots.")

    # 3. Organize into metric subfolders (AUC_*, ECE_*)
    metric_folders = {
        "AUC_1": "sparsification_curve_1_gaussian.png",
        "AUC_2": "sparsification_curve_2_gaussians.png",
        "AUC_all": "sparsification_curve_all.png",
        "AUC_ocluded": "sparsification_curve_occluded.png",
        "AUC_visible": "sparsification_curve_visible.png",
        "ECE_1": "ece_calibration_1_gaussian.png",
        "ECE_2": "ece_calibration_2_gaussians.png",
        "ECE_all": "ece_calibration_all.png",
        "ECE_ocluded": "ece_calibration_occluded.png",
        "ECE_visible": "ece_calibration_visible.png",
    }

    log.info("--> Organizing plots into metric folders AUC_* and ECE_* ...")
    for folder_name, pattern in metric_folders.items():
        metric_dir = graficas_dir / folder_name
        metric_dir.mkdir(parents=True, exist_ok=True)
        
        for subfolder in graficas_dir.iterdir():
            if subfolder.is_dir() and subfolder.name not in metric_folders:
                for img_path in subfolder.glob("*.png"):
                    if img_path.name.endswith(pattern):
                        shutil.copy2(img_path, metric_dir / img_path.name)

    # 4. Copy catastrophic failures boxplot to root
    general_folder = graficas_dir / "General"
    if general_folder.exists():
        for boxplot_path in general_folder.glob("*catastrophic_failures_boxplot.png"):
            shutil.copy2(boxplot_path, graficas_dir / boxplot_path.name)

    log.info(f"\n¡Proceso completado! Todas las gráficas han sido generadas y organizadas en:\n{graficas_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and organize uncertainty evaluation plots across degradation levels.")
    parser.add_argument("input_dir", nargs="?", default=r"outputs/degradation_benchmark", help="Directory containing parquet files.")
    parser.add_argument("--beta", type=float, default=42.2103, help="Beta hyperparameter for adaptive uncertainty.")
    parser.add_argument("--strategy", type=str, default="max_pooling", help="Adaptive strategy (max_pooling, gating_k, etc.)")
    parser.add_argument("--tau", type=float, default=0.5, help="Threshold tau for gating_tau strategy.")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_organized_plots(args.input_dir, beta=args.beta, strategy=args.strategy, tau=args.tau)

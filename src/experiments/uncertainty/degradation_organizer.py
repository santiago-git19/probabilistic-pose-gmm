"""Hierarchical Output Organizer for Degradation Benchmarking and Adaptive Optimization.

Structures evaluation outputs (plots, CSVs, and JSONs) across resolution tiers
and generates consolidated master tables and cross-metric directories.
"""
import json
import logging
import shutil
from pathlib import Path
from typing import Tuple, List, Dict, Any

import pandas as pd

log = logging.getLogger(__name__)

EVAL_METRIC_FOLDERS = {
    # Sparsification (AUC_*)
    "AUC_all": "*_sparsification_curve_all.png",
    "AUC_visible": "*_sparsification_curve_visible.png",
    "AUC_ocluded": "*_sparsification_curve_occluded.png",
    "AUC_1": "*_sparsification_curve_1_gaussian.png",
    "AUC_2": "*_sparsification_curve_2_gaussians.png",
    # Calibration (ECE_*)
    "ECE_all": "*_ece_calibration_all.png",
    "ECE_visible": "*_ece_calibration_visible.png",
    "ECE_ocluded": "*_ece_calibration_occluded.png",
    "ECE_1": "*_ece_calibration_1_gaussian.png",
    "ECE_2": "*_ece_calibration_2_gaussians.png",
    # Out-of-Distribution (OoD) ROC
    "OoD_ROC_all": "*_ood_absence_roc_all.png",
    "OoD_ROC_1": "*_ood_absence_roc_1_gaussian.png",
    "OoD_ROC_2": "*_ood_absence_roc_2_gaussians.png",
    # Out-of-Distribution (OoD) KDE (Grid)
    "OoD_KDE_grid_all": "*_ood_absence_kde_grid_all.png",
    "OoD_KDE_grid_1": "*_ood_absence_kde_grid_1_gaussian.png",
    "OoD_KDE_grid_2": "*_ood_absence_kde_grid_2_gaussians.png",
    # Limb Swaps
    "Limb_Swaps": "*_limb_swap_roc.png",
}

OPT_METRIC_FOLDERS = {
    "AUSE_vs_Beta_all": "*_ause_vs_beta_all.png",
    "AUSE_vs_Beta_visible": "*_ause_vs_beta_visible.png",
    "AUSE_vs_Beta_occluded": "*_ause_vs_beta_occluded.png",
}


def get_graficas_dir(input_path: Path, out_arg: Path = None) -> Path:
    """Determine standardized path to plots directory."""
    if out_arg and str(out_arg) != "." and out_arg is not None:
        g_dir = out_arg if out_arg.name in ("plots", "graficas") else out_arg / "graficas"
    else:
        base = input_path if input_path.is_dir() else input_path.parent
        g_dir = base if base.name in ("plots", "graficas") else base / "graficas"
    g_dir.mkdir(parents=True, exist_ok=True)
    return g_dir


def get_deg_info(parquet_path: Path) -> Tuple[str, str]:
    """Extract experiment subfolder name and file prefix from parquet path."""
    name = parquet_path.stem
    if "combined" in name or "General" in name:
        return "06_General", "General"
    else:
        deg_name = name.replace("_results", "")
        return deg_name, deg_name


def prefix_folder_files(subfolder: Path, prefix: str):
    """Ensure all files in an experiment subfolder are prefixed by resolution name."""
    for file_path in list(subfolder.iterdir()):
        if file_path.is_file() and not file_path.name.startswith(f"{prefix}_"):
            new_name = f"{prefix}_{file_path.name}"
            target_path = subfolder / new_name
            if target_path.exists():
                target_path.unlink()
            file_path.rename(target_path)


def organize_metric_folders_and_masters(graficas_dir: Path):
    """
    Group plots into transversal metric folders (AUC_*, ECE_*, OoD_*, ...),
    copy CSV and JSON to Metrics_CSV/ y Metrics_JSON/,and build combined master summary files.
    """
    all_metric_patterns = {**EVAL_METRIC_FOLDERS, **OPT_METRIC_FOLDERS}
    
    # 1. Distribute plots into metric subfolders
    for folder_name, pattern in all_metric_patterns.items():
        metric_dir = graficas_dir / folder_name
        metric_dir.mkdir(parents=True, exist_ok=True)
        
        for subfolder in graficas_dir.iterdir():
            if subfolder.is_dir() and subfolder.name not in all_metric_patterns and subfolder.name not in ["Metrics_CSV", "Metrics_JSON"]:
                for img_path in subfolder.glob("*.png"):
                    if img_path.match(pattern) or img_path.name.endswith(pattern.replace("*", "")):
                        shutil.copy2(img_path, metric_dir / img_path.name)
                        
    # 2. Distribute numerical CSV and JSON files
    metrics_csv_dir = graficas_dir / "Metrics_CSV"
    metrics_json_dir = graficas_dir / "Metrics_JSON"
    metrics_csv_dir.mkdir(parents=True, exist_ok=True)
    metrics_json_dir.mkdir(parents=True, exist_ok=True)
    
    for subfolder in graficas_dir.iterdir():
        if subfolder.is_dir() and subfolder.name not in all_metric_patterns and subfolder.name not in ["Metrics_CSV", "Metrics_JSON"]:
            for csv_path in subfolder.glob("*.csv"):
                shutil.copy2(csv_path, metrics_csv_dir / csv_path.name)
            for json_path in subfolder.glob("*.json"):
                shutil.copy2(json_path, metrics_json_dir / json_path.name)
                
    # 3. Build consolidated master tables
    _build_combined_masters(graficas_dir)
    
    # 4. Copy general boxplot to root if present
    general_folder = graficas_dir / "06_General"
    if general_folder.exists():
        for boxplot_path in general_folder.glob("*catastrophic_failures_boxplot.png"):
            shutil.copy2(boxplot_path, graficas_dir / boxplot_path.name)


def _build_combined_masters(graficas_dir: Path):
    """Generate global consolidated CSVs and JSONs aggregating all degradation levels."""
    metrics_csv_dir = graficas_dir / "Metrics_CSV"
    if metrics_csv_dir.exists():
        eval_csvs = sorted(list(metrics_csv_dir.glob("*_evaluation_metrics.csv")))
        if eval_csvs:
            dfs = []
            for c in eval_csvs:
                deg = c.name.replace("_evaluation_metrics.csv", "")
                df_sub = pd.read_csv(c)
                df_sub.insert(0, "degradation_level", deg)
                dfs.append(df_sub)
            df_all_eval = pd.concat(dfs, ignore_index=True)
            df_all_eval.to_csv(graficas_dir / "all_evaluation_metrics_combined.csv", index=False, encoding="utf-8")
            
        opt_csvs = sorted(list(metrics_csv_dir.glob("*_optimization_results.csv")))
        if opt_csvs:
            dfs = []
            for c in opt_csvs:
                deg = c.name.replace("_optimization_results.csv", "")
                df_sub = pd.read_csv(c)
                df_sub.insert(0, "degradation_level", deg)
                dfs.append(df_sub)
            df_all_opt = pd.concat(dfs, ignore_index=True)
            df_all_opt.to_csv(graficas_dir / "all_optimization_results_combined.csv", index=False, encoding="utf-8")
            
    metrics_json_dir = graficas_dir / "Metrics_JSON"
    if metrics_json_dir.exists():
        eval_jsons = sorted(list(metrics_json_dir.glob("*_evaluation_metrics.json")))
        if eval_jsons:
            combined_json = {}
            for j in eval_jsons:
                deg = j.name.replace("_evaluation_metrics.json", "")
                try:
                    with open(j, "r", encoding="utf-8") as f:
                        combined_json[deg] = json.load(f)
                except Exception as e:
                    log.warning(f"Could not read JSON from {j}: {e}")
            with open(graficas_dir / "all_evaluation_metrics_combined.json", "w", encoding="utf-8") as f:
                json.dump(combined_json, f, indent=2, ensure_ascii=False)
                
        opt_jsons = sorted(list(metrics_json_dir.glob("*_optimization_results.json")))
        if opt_jsons:
            combined_json = {}
            for j in opt_jsons:
                deg = j.name.replace("_optimization_results.json", "")
                try:
                    with open(j, "r", encoding="utf-8") as f:
                        combined_json[deg] = json.load(f)
                except Exception as e:
                    log.warning(f"Could not read JSON from {j}: {e}")
            with open(graficas_dir / "all_optimization_results_combined.json", "w", encoding="utf-8") as f:
                json.dump(combined_json, f, indent=2, ensure_ascii=False)


def run_organized_evaluation_pipeline(input_path: Path, out_arg: str = None, beta: float = 0.1, strategy: str = "max_pooling", tau: float = 0.5) -> bool:
    """Execute hierarchical evaluation organized by resolution tier."""
    from src.experiments.uncertainty.evaluate_uncertainty import generate_all_plots
    input_path = Path(input_path).resolve()
    out_path = Path(out_arg) if out_arg and str(out_arg) != "." else None
    
    if input_path.is_dir():
        parquets = sorted(list(input_path.glob("*_results.parquet")))
        if not parquets:
            parquets = [p for p in sorted(list(input_path.glob("*.parquet"))) if "metadata" not in p.name and "combined" not in p.name]
            
        if parquets:
            graficas_dir = get_graficas_dir(input_path, out_path)
            log.info(f"Degradation benchmark mode detected. Processing {len(parquets)} files in {graficas_dir}...")
            
            for p in parquets:
                deg_folder, prefix = get_deg_info(p)
                subfolder = graficas_dir / deg_folder
                subfolder.mkdir(parents=True, exist_ok=True)
                log.info(f"--> Evaluating resolution: {prefix} ({p.name})")
                df_sub = pd.read_parquet(p)
                if not df_sub.empty:
                    generate_all_plots(df_sub, subfolder, beta=beta, strategy=strategy, tau=tau)
                    prefix_folder_files(subfolder, prefix)
                    
            combined_p = input_path / "all_degradations_combined.parquet"
            if combined_p.exists():
                deg_folder, prefix = get_deg_info(combined_p)
                subfolder = graficas_dir / deg_folder
                subfolder.mkdir(parents=True, exist_ok=True)
                log.info(f"--> Evaluating global dataset: {prefix} ({combined_p.name})")
                df_general = pd.read_parquet(combined_p)
                if not df_general.empty:
                    generate_all_plots(df_general, subfolder, beta=beta, strategy=strategy, tau=tau)
                    prefix_folder_files(subfolder, prefix)
                    
            organize_metric_folders_and_masters(graficas_dir)
            log.info(f"\nEvaluation finished! Organized plots, CSVs, and JSONs saved to:\n{graficas_dir}")
            return True
    elif input_path.is_file() and (input_path.name.endswith("_results.parquet") or "combined" in input_path.name or "General" in input_path.name or (input_path.parent.name == "degradation_benchmark") or (out_path and out_path.name in ("plots", "graficas"))):
        graficas_dir = get_graficas_dir(input_path, out_path)
        deg_folder, prefix = get_deg_info(input_path)
        subfolder = graficas_dir / deg_folder
        subfolder.mkdir(parents=True, exist_ok=True)
        log.info(f"--> Evaluating degradation file: {prefix} ({input_path.name})")
        df_sub = pd.read_parquet(input_path)
        if not df_sub.empty:
            generate_all_plots(df_sub, subfolder, beta=beta, strategy=strategy, tau=tau)
            prefix_folder_files(subfolder, prefix)
            organize_metric_folders_and_masters(graficas_dir)
            log.info(f"\nOrganized evaluation of {prefix} completed in:\n{subfolder}")
            return True
            
    return False


def run_organized_optimization_pipeline(input_path: Path, out_arg: str = None) -> bool:
    """Execute adaptive uncertainty optimization organized by resolution tier."""
    from src.experiments.uncertainty.optimize_adaptive_uncertainty import run_grid_search
    input_path = Path(input_path).resolve()
    out_path = Path(out_arg) if out_arg and str(out_arg) != "." else None
    
    if input_path.is_dir():
        parquets = sorted(list(input_path.glob("*_results.parquet")))
        if not parquets:
            parquets = [p for p in sorted(list(input_path.glob("*.parquet"))) if "metadata" not in p.name and "combined" not in p.name]
            
        if parquets:
            graficas_dir = get_graficas_dir(input_path, out_path)
            log.info(f"Degradation optimization benchmark mode detected. Processing {len(parquets)} files in {graficas_dir}...")
            
            for p in parquets:
                deg_folder, prefix = get_deg_info(p)
                subfolder = graficas_dir / deg_folder
                subfolder.mkdir(parents=True, exist_ok=True)
                log.info(f"--> Optimizing resolution: {prefix} ({p.name})")
                run_grid_search(p, subfolder)
                prefix_folder_files(subfolder, prefix)
                
            combined_p = input_path / "all_degradations_combined.parquet"
            if combined_p.exists():
                deg_folder, prefix = get_deg_info(combined_p)
                subfolder = graficas_dir / deg_folder
                subfolder.mkdir(parents=True, exist_ok=True)
                log.info(f"--> Optimizing global dataset: {prefix} ({combined_p.name})")
                run_grid_search(combined_p, subfolder)
                prefix_folder_files(subfolder, prefix)
                
            organize_metric_folders_and_masters(graficas_dir)
            log.info(f"\nOrganized optimization completed! Saved to:\n{graficas_dir}")
            return True
    elif input_path.is_file() and (input_path.name.endswith("_results.parquet") or "combined" in input_path.name or "General" in input_path.name or (input_path.parent.name == "degradation_benchmark") or (out_path and out_path.name in ("plots", "graficas"))):
        graficas_dir = get_graficas_dir(input_path, out_path)
        deg_folder, prefix = get_deg_info(input_path)
        subfolder = graficas_dir / deg_folder
        subfolder.mkdir(parents=True, exist_ok=True)
        log.info(f"--> Optimizing file: {prefix} ({input_path.name})")
        run_grid_search(input_path, subfolder)
        prefix_folder_files(subfolder, prefix)
        organize_metric_folders_and_masters(graficas_dir)
        log.info(f"\nOrganized optimization of {prefix} completed in:\n{subfolder}")
        return True
        
    return False

if __name__ == "__main__":
    import argparse
    import sys
    
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
        
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="Organizer for degradation evaluation and adaptive optimization.")
    parser.add_argument("input_path", type=str, help="Path to degradation_benchmark directory or a parquet results file.")
    parser.add_argument("--out", type=str, default=None, help="Output directory (optional).")
    parser.add_argument("--optimize", action="store_true", help="Run optimization pipeline instead of evaluation pipeline.")
    parser.add_argument("--force", action="store_true", help="Force regeneration of results (overwrite existing).")
    
    args = parser.parse_args()
    
    # Enable force by clearing existing jsons if requested
    if args.force:
        target_dir = get_graficas_dir(Path(args.input_path), Path(args.out) if args.out else None)
        if target_dir.exists():
            log.info(f"Force mode active: clearing previous results in {target_dir}")
            for json_file in target_dir.rglob("*.json"):
                if args.optimize and "optimization" in json_file.name:
                    json_file.unlink()
                elif not args.optimize and "evaluation" in json_file.name:
                    json_file.unlink()
    
    if args.optimize:
        run_organized_optimization_pipeline(args.input_path, args.out)
    else:
        run_organized_evaluation_pipeline(args.input_path, args.out)

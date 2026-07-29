"""
Script para generar gráficas de evaluación de incertidumbre por cada nivel de degradación (resolución)
y en general, organizando toda la salida en subcarpetas por degradación y por métrica (AUC_*, ECE_*),
y copiando el boxplot general a la raíz.

Utiliza las rutinas de evaluate_uncertainty.py.
"""
import argparse
import logging
import shutil
import sys
from pathlib import Path
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.evaluate_uncertainty import generate_all_plots

log = logging.getLogger(__name__)

def generate_organized_plots(input_dir: Path, beta: float = 42.2103, strategy: str = "max_pooling", tau: float = 0.5):
    input_dir = Path(input_dir).resolve()
    if not input_dir.exists():
        log.error(f"El directorio {input_dir} no existe.")
        return

    graficas_dir = input_dir / "graficas"
    graficas_dir.mkdir(parents=True, exist_ok=True)

    # 1. Buscar archivos parquet individuales por degradación
    parquets = sorted(list(input_dir.glob("*_results.parquet")))
    if not parquets:
        parquets = [p for p in sorted(list(input_dir.glob("*.parquet"))) if "metadata" not in p.name and "combined" not in p.name]

    if not parquets:
        log.warning("No se encontraron archivos parquet individuales de degradación.")
    else:
        log.info(f"Se encontraron {len(parquets)} archivos parquet individuales. Generando gráficas en {graficas_dir} ...")

    for p in parquets:
        deg_name = p.stem.replace("_results", "")
        log.info(f"--> Procesando resolución / degradación: {deg_name} ({p.name})")
        
        df = pd.read_parquet(p)
        if df.empty:
            log.warning(f"El archivo {p.name} está vacío. Saltando.")
            continue
            
        subfolder = graficas_dir / deg_name
        subfolder.mkdir(parents=True, exist_ok=True)
        
        # Generar gráficas usando evaluate_uncertainty.py
        generate_all_plots(df, subfolder, beta=beta, strategy=strategy, tau=tau)
        
        # Renombrar archivos dentro de la subcarpeta para incluir el prefijo de la degradación
        for img_path in list(subfolder.glob("*.png")):
            if not img_path.name.startswith(f"{deg_name}_"):
                new_name = f"{deg_name}_{img_path.name}"
                img_path.rename(subfolder / new_name)

    # 2. Procesar el dataset general (combinado)
    combined_parquet = input_dir / "all_degradations_combined.parquet"
    if combined_parquet.exists():
        log.info(f"--> Procesando dataset general: General ({combined_parquet.name})")
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
        log.warning(f"No se encontró {combined_parquet.name} para generar las gráficas de 'General'.")

    # 3. Organizar en subcarpetas por métrica (AUC_*, ECE_*)
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

    log.info("--> Organizando gráficas por métricas en carpetas AUC_* y ECE_* ...")
    for folder_name, pattern in metric_folders.items():
        metric_dir = graficas_dir / folder_name
        metric_dir.mkdir(parents=True, exist_ok=True)
        
        for subfolder in graficas_dir.iterdir():
            if subfolder.is_dir() and subfolder.name not in metric_folders:
                for img_path in subfolder.glob("*.png"):
                    if img_path.name.endswith(pattern):
                        shutil.copy2(img_path, metric_dir / img_path.name)

    # 4. Copiar el boxplot de fallos catastróficos general a la raíz de 'graficas'
    general_folder = graficas_dir / "General"
    if general_folder.exists():
        for boxplot_path in general_folder.glob("*catastrophic_failures_boxplot.png"):
            shutil.copy2(boxplot_path, graficas_dir / boxplot_path.name)

    log.info(f"\n¡Proceso completado! Todas las gráficas han sido generadas y organizadas en:\n{graficas_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar y organizar gráficas de incertidumbre separadas por degradación y métricas.")
    parser.add_argument("input_dir", nargs="?", default=r"C:\Users\Santiago estudio\Desktop\TFG_Informatica\Resultados_incertidumbre\General\coco\degradation_benchmark", help="Directorio con los archivos parquet.")
    parser.add_argument("--beta", type=float, default=42.2103, help="Hiperparámetro beta para incertidumbre adaptativa.")
    parser.add_argument("--strategy", type=str, default="max_pooling", help="Estrategia adaptativa (max_pooling, gating_k, etc.)")
    parser.add_argument("--tau", type=float, default=0.5, help="Umbral tau para estrategia gating_tau.")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_organized_plots(args.input_dir, beta=args.beta, strategy=args.strategy, tau=args.tau)

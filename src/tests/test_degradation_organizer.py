import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from src.experiments.uncertainty.degradation_organizer import (
    get_graficas_dir,
    get_deg_info,
    run_organized_evaluation_pipeline,
    run_organized_optimization_pipeline
)
from src.experiments.uncertainty.evaluate_uncertainty import COCO_KEYPOINT_NAMES


@pytest.fixture
def sample_image_df():
    """Crea un DataFrame sintético a nivel de imagen con keypoints visibles y ausentes (vis == 0)."""
    np.random.seed(42)
    rows = []
    for i in range(10):
        row = {
            "experiment_name": "test_noise_0.0",
            "is_swapped": 0,
            "uniform_weight_mean": 0.15,
        }
        for k_idx, kp in enumerate(COCO_KEYPOINT_NAMES):
            vis = 0 if k_idx >= 14 else 2
            row[f"vis_{kp}"] = vis
            row[f"n_components_{kp}"] = 2 if k_idx % 2 == 0 else 1
            row[f"cov_det_{kp}"] = abs(np.random.normal(10.0, 2.0)) if vis == 0 else abs(np.random.normal(0.5, 0.1))
            row[f"uniform_weight_{kp}"] = np.clip(np.random.normal(0.8, 0.05), 0, 1) if vis == 0 else np.clip(np.random.normal(0.1, 0.02), 0, 1)
            row[f"base_score_{kp}"] = np.clip(np.random.normal(0.2, 0.05), 0, 1) if vis == 0 else np.clip(np.random.normal(0.9, 0.02), 0, 1)
            row[f"heatmap_entropy_{kp}"] = abs(np.random.normal(3.5, 0.5)) if vis == 0 else abs(np.random.normal(1.2, 0.2))
            row[f"oks_ours_{kp}"] = np.nan if vis == 0 else np.clip(np.random.normal(0.85, 0.05), 0, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def test_get_deg_info():
    assert get_deg_info(Path("00_Baseline_Clean_results.parquet")) == ("00_Baseline_Clean", "00_Baseline_Clean")
    assert get_deg_info(Path("all_degradations_combined.parquet")) == ("06_General", "General")


def test_organized_evaluation_pipeline(sample_image_df, tmp_path):
    """Verifica que el pipeline de evaluación organiza carpetas, prefija archivos y crea maestros combinados."""
    deg_dir = tmp_path / "degradation_benchmark"
    deg_dir.mkdir()
    
    p1 = deg_dir / "00_Baseline_Clean_results.parquet"
    p2 = deg_dir / "01_Resize_Low_results.parquet"
    sample_image_df.to_parquet(p1)
    sample_image_df.to_parquet(p2)
    
    res = run_organized_evaluation_pipeline(deg_dir, out_arg=str(deg_dir / "graficas"))
    assert res is True
    
    graficas = deg_dir / "graficas"
    assert graficas.exists()
    
    f1 = graficas / "00_Baseline_Clean"
    f2 = graficas / "01_Resize_Low"
    assert f1.exists() and f2.exists()
    
    assert (f1 / "00_Baseline_Clean_evaluation_metrics.csv").exists()
    assert (f1 / "00_Baseline_Clean_evaluation_metrics.json").exists()
    assert (f1 / "00_Baseline_Clean_ood_absence_roc_all.png").exists()
    
    assert (f2 / "01_Resize_Low_evaluation_metrics.csv").exists()
    assert (f2 / "01_Resize_Low_evaluation_metrics.json").exists()
    
    assert (graficas / "Metrics_CSV" / "00_Baseline_Clean_evaluation_metrics.csv").exists()
    assert (graficas / "Metrics_JSON" / "01_Resize_Low_evaluation_metrics.json").exists()
    
    combined_csv = graficas / "all_evaluation_metrics_combined.csv"
    combined_json = graficas / "all_evaluation_metrics_combined.json"
    assert combined_csv.exists()
    assert combined_json.exists()
    
    df_comb = pd.read_csv(combined_csv)
    assert "degradation_level" in df_comb.columns
    assert set(df_comb["degradation_level"].unique()) == {"00_Baseline_Clean", "01_Resize_Low"}
    
    with open(combined_json, "r", encoding="utf-8") as f:
        data_json = json.load(f)
    assert "00_Baseline_Clean" in data_json
    assert "01_Resize_Low" in data_json


def test_organized_optimization_pipeline(sample_image_df, tmp_path):
    """Verifica que el pipeline de optimización organiza las gráficas ause_vs_beta y resultados en carpetas."""
    deg_dir = tmp_path / "degradation_benchmark"
    deg_dir.mkdir()
    
    p1 = deg_dir / "00_Baseline_Clean_results.parquet"
    sample_image_df.to_parquet(p1)
    
    res = run_organized_optimization_pipeline(deg_dir, out_arg=str(deg_dir / "graficas"))
    assert res is True
    
    graficas = deg_dir / "graficas"
    f1 = graficas / "00_Baseline_Clean"
    
    assert (f1 / "00_Baseline_Clean_ause_vs_beta_all.png").exists()
    assert (f1 / "00_Baseline_Clean_optimization_results.csv").exists()
    assert (f1 / "00_Baseline_Clean_optimization_results.json").exists()
    
    assert (graficas / "AUSE_vs_Beta_all" / "00_Baseline_Clean_ause_vs_beta_all.png").exists()
    assert (graficas / "all_optimization_results_combined.csv").exists()

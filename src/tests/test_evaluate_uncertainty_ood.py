import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from experiments.evaluate_uncertainty import (
    _unroll_keypoints,
    plot_sparsification,
    plot_ece,
    plot_ood_absence_roc,
    plot_ood_absence_kde,
    generate_all_plots,
    COCO_KEYPOINT_NAMES
)


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
            # Simulamos que los 3 últimos keypoints están ausentes (vis == 0)
            vis = 0 if k_idx >= 14 else 2
            row[f"vis_{kp}"] = vis
            row[f"n_components_{kp}"] = 2 if k_idx % 2 == 0 else 1
            row[f"cov_det_{kp}"] = abs(np.random.normal(10.0, 2.0)) if vis == 0 else abs(np.random.normal(0.5, 0.1))
            row[f"uniform_weight_{kp}"] = np.clip(np.random.normal(0.8, 0.05), 0, 1) if vis == 0 else np.clip(np.random.normal(0.1, 0.02), 0, 1)
            row[f"base_score_{kp}"] = np.clip(np.random.normal(0.2, 0.05), 0, 1) if vis == 0 else np.clip(np.random.normal(0.9, 0.02), 0, 1)
            row[f"heatmap_entropy_{kp}"] = abs(np.random.normal(3.5, 0.5)) if vis == 0 else abs(np.random.normal(1.2, 0.2))
            
            # Si vis == 0, oks_ours no tiene sentido y es NaN
            row[f"oks_ours_{kp}"] = np.nan if vis == 0 else np.clip(np.random.normal(0.85, 0.05), 0, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def test_unroll_keypoints_retains_absent_and_uniform_weight(sample_image_df):
    """Verifica que _unroll_keypoints retiene vis == 0 y recoge uniform_weight."""
    df_kp = _unroll_keypoints(sample_image_df)
    
    assert not df_kp.empty
    assert "uniform_weight" in df_kp.columns
    assert "cov_det" in df_kp.columns
    assert "oks_ours" in df_kp.columns
    
    # Comprobar que existen filas con vis == 0
    absent_rows = df_kp[df_kp["vis"] == 0]
    assert len(absent_rows) > 0
    assert absent_rows["oks_ours"].isna().all()
    assert not absent_rows["cov_det"].isna().any()
    assert not absent_rows["uniform_weight"].isna().any()
    
    # Comprobar que existen filas con vis > 0 y tienen oks_ours válido
    visible_rows = df_kp[df_kp["vis"] > 0]
    assert len(visible_rows) > 0
    assert not visible_rows["oks_ours"].isna().any()


def test_geometric_plots_ignore_absent_keypoints(sample_image_df, tmp_path):
    """Verifica que plot_sparsification y plot_ece filtran vis == 0 sin fallar."""
    df_kp = _unroll_keypoints(sample_image_df)
    
    # Ejecutar sin errores
    plot_sparsification(df_kp, tmp_path, suffix="_test")
    plot_ece(df_kp, tmp_path, suffix="_test")
    
    assert (tmp_path / "sparsification_curve_test.png").exists()
    assert (tmp_path / "ece_calibration_test.png").exists()


def test_ood_absence_roc_and_kde(sample_image_df, tmp_path):
    """Verifica que plot_ood_absence_roc y plot_ood_absence_kde generan las gráficas OoD."""
    df_kp = _unroll_keypoints(sample_image_df)
    
    plot_ood_absence_roc(df_kp, tmp_path, suffix="_test")
    plot_ood_absence_kde(df_kp, tmp_path, suffix="_test")
    
    assert (tmp_path / "ood_absence_roc_test.png").exists()
    assert (tmp_path / "ood_absence_kde_grid_test.png").exists()
    assert (tmp_path / "ood_absence_kde_log_cov_det_test.png").exists()
    assert (tmp_path / "ood_absence_kde_uniform_weight_test.png").exists()
    assert (tmp_path / "ood_absence_kde_heatmap_entropy_test.png").exists()
    assert (tmp_path / "ood_absence_kde_inv_base_score_test.png").exists()


def test_generate_all_plots_ood_integration(sample_image_df, tmp_path):
    """Verifica que generate_all_plots ejecuta la pipeline completa incluyendo OoD."""
    generate_all_plots(sample_image_df, tmp_path)
    
    assert (tmp_path / "ood_absence_roc_all.png").exists()
    assert (tmp_path / "ood_absence_kde_grid_all.png").exists()
    assert (tmp_path / "evaluation_metrics.json").exists()
    assert (tmp_path / "evaluation_metrics.csv").exists()

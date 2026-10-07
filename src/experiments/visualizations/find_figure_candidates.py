"""
Find Figure Candidates: Systematic Pandas queries across ALL .parquet benchmark tables
(Resultados_Incertidumbre, Resultados_Precision/TTA, and Resultados_Precision/Sin_DARK/MRF)
covering HRNet-W32, ResNet50, and ViTPose-Small across COCO, CrowdPose, and OCHuman.

Pipeline Stages Mapped:
1. GMM (No TTA): Paper/Resultados_Incertidumbre/[Modelos_Secundarios/]
2. GMM + TTA: Paper/Resultados_Precision/TTA/[Modelos_Secundarios/TTA_]
3. GMM + TTA + MRF: Paper/Resultados_Precision/Sin_DARK/MRF/[Modelos_Secundarios/]

Catalog of Target Figures:
- Fig 1: Symmetric Swap Disambiguation (GMM rescuing crossed limbs from DARK)
- Fig 2: The MRF Dilemma (Comparing GMM+TTA vs. GMM+TTA+MRF directly)
- Fig 3: Heuristic Overconfidence vs. Volumetric OoD Alert (CrowdPose / OCHuman)
- Fig 4: Mitigating Heatmap Poisoning in Continuous TTA
- Fig 5: Uniform Background Component (pi_uniform) Absorbing Severe Noise
- Fig 6: Cross-Architecture Consistency (HRNet vs. ResNet50 vs. ViTPose on the same Image ID)

Usage:
    # Scan all 3 pipeline stages across all models and datasets
    poetry run python src/experiments/visualizations/find_figure_candidates.py --scan-all

    # Filter by specific pipeline stage: gmm_no_tta, gmm_tta, gmm_tta_mrf, or all
    poetry run python src/experiments/visualizations/find_figure_candidates.py --stage gmm_tta

    # Filter by model: hrnet_w32, resnet50, vitpose_small, or all
    poetry run python src/experiments/visualizations/find_figure_candidates.py --model hrnet_w32

    # Save curated catalog to JSON
    poetry run python src/experiments/visualizations/find_figure_candidates.py --scan-all --out-json "figure_candidates_catalog.json"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from src.pose_uncertainty.core.skeleton import COCO_KEYPOINT_NAMES

MODELS = ["hrnet_w32", "resnet50", "vitpose_small"]
DATASETS = ["coco", "crowdpose", "ochuman"]

# Base roots (checks outputs/data first, then fallback)
data_root = (
    project_root / "outputs" / "data"
    if (project_root / "outputs" / "data" / "Resultados_Incertidumbre").exists()
    else (
        project_root.parent / "Paper"
        if (project_root.parent / "Paper").exists()
        else project_root / "Paper"
    )
)
ROOT_UNCERTAINTY = data_root / "Resultados_Incertidumbre"
ROOT_TTA = data_root / "Resultados_Precision" / "TTA"
ROOT_MRF = data_root / "Resultados_Precision" / "Sin_DARK" / "MRF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find ideal qualitative figure candidates from parquet tables.")
    parser.add_argument(
        "--scan-all",
        action="store_true",
        default=True,
        help="Scan and merge all benchmark parquet files across all 3 pipeline stages (default: True)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "gmm_no_tta", "gmm_tta", "gmm_tta_mrf"],
        help="Filter by pipeline stage (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "hrnet_w32", "resnet50", "vitpose_small"],
        help="Filter by model architecture (default: all)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top candidates to display per figure (default: 5)",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="figure_candidates_catalog.json",
        help="Output JSON file path (default: figure_candidates_catalog.json)",
    )
    return parser.parse_args()


# ==============================================================================
# Multi-Stage Parquet Loader
# ==============================================================================

def get_parquet_path(stage: str, model: str, dataset: str) -> Optional[Path]:
    """Resolve the exact parquet path for a given (stage, model, dataset) triplet."""
    if stage == "gmm_no_tta":
        if model == "hrnet_w32":
            p = ROOT_UNCERTAINTY / f"{model}_{dataset}" / "degradation_benchmark" / "all_degradations_combined.parquet"
        else:
            p = ROOT_UNCERTAINTY / "Modelos_Secundarios" / f"{model}_{dataset}" / "degradation_benchmark" / "all_degradations_combined.parquet"
    elif stage == "gmm_tta":
        if model == "hrnet_w32":
            p = ROOT_TTA / f"{model}_{dataset}" / "degradation_benchmark" / "all_degradations_combined.parquet"
        else:
            p = ROOT_TTA / "Modelos_Secundarios" / f"TTA_{model}_{dataset}" / "degradation_benchmark" / "all_degradations_combined.parquet"
    elif stage == "gmm_tta_mrf":
        if model == "hrnet_w32":
            p = ROOT_MRF / f"{model}_{dataset}" / "degradation_benchmark" / "all_degradations_combined.parquet"
        else:
            p = ROOT_MRF / "Modelos_Secundarios" / f"{model}_{dataset}" / "degradation_benchmark" / "all_degradations_combined.parquet"
    else:
        return None

    if p.exists():
        return p
    return None


def load_all_pipeline_parquets(stage_filter: str = "all", model_filter: str = "all") -> Dict[str, pd.DataFrame]:
    """Load and tag all available parquet tables across all stages, models, and datasets."""
    stages = ["gmm_no_tta", "gmm_tta", "gmm_tta_mrf"] if stage_filter == "all" else [stage_filter]
    models = MODELS if model_filter == "all" else [model_filter]

    dfs_by_stage: Dict[str, List[pd.DataFrame]] = {"gmm_no_tta": [], "gmm_tta": [], "gmm_tta_mrf": []}

    for st in stages:
        for m in models:
            for ds in DATASETS:
                p_path = get_parquet_path(st, m, ds)
                if p_path and p_path.exists():
                    try:
                        df_sub = pd.read_parquet(p_path)
                        df_sub["pipeline_stage"] = st
                        df_sub["model_arch"] = m
                        df_sub["dataset_name"] = ds
                        df_sub["source_parquet"] = str(p_path)
                        dfs_by_stage[st].append(df_sub)
                    except Exception as e:
                        logger.warning("Error reading %s: %s", p_path, e)

    consolidated: Dict[str, pd.DataFrame] = {}
    for st, df_list in dfs_by_stage.items():
        if df_list:
            consolidated[st] = pd.concat(df_list, ignore_index=True)
            logger.info("Stage '%s' loaded: %d rows (%d benchmarks)", st, len(consolidated[st]), len(df_list))

    return consolidated


def print_section_header(title: str):
    print("\n" + "=" * 95)
    print(f" {title.upper()}")
    print("=" * 95)


def display_candidates_table(df_subset: pd.DataFrame, columns: List[Tuple[str, str, int]], top_k: int = 5):
    if df_subset.empty:
        print("  [!] No matches found for this specific query.")
        return

    header = " | ".join(f"{name:<{w}}" for name, _, w in columns)
    print(header)
    print("-" * len(header))

    for _, row in df_subset.head(top_k).iterrows():
        row_str = []
        for _, col_name, w in columns:
            val = row.get(col_name, "")
            if isinstance(val, float):
                val_str = f"{val:.4f}" if not np.isnan(val) else "N/A"
            elif isinstance(val, (int, np.integer)):
                val_str = str(int(val))
            else:
                val_str = str(val)
            row_str.append(f"{val_str:<{w}}")
        print(" | ".join(row_str))


COCO_SYMMETRIC_PAIRS = [
    ("left_shoulder", "right_shoulder", 5, 6, "L_Shoulder", "R_Shoulder"),
    ("left_elbow", "right_elbow", 7, 8, "L_Elbow", "R_Elbow"),
    ("left_wrist", "right_wrist", 9, 10, "L_Wrist", "R_Wrist"),
    ("left_hip", "right_hip", 11, 12, "L_Hip", "R_Hip"),
    ("left_knee", "right_knee", 13, 14, "L_Knee", "R_Knee"),
    ("left_ankle", "right_ankle", 15, 16, "L_Ankle", "R_Ankle"),
]


def extract_best_swapped_joint(row: pd.Series) -> Dict[str, Any]:
    """
    Find the symmetric joint that had the highest delta OKS and best recovery.
    Returns a dict with joint name, index, delta, base OKS, and ours OKS.
    """
    best_name, best_idx = "N/A", 16
    best_delta = -999.0
    best_o_val, best_b_val = 0.0, 0.0

    for l_col, r_col, l_idx, r_idx, l_label, r_label in COCO_SYMMETRIC_PAIRS:
        d_l = float(row.get(f"delta_oks_{l_col}", -999))
        o_l = float(row.get(f"oks_ours_{l_col}", -999))
        b_l = float(row.get(f"oks_base_{l_col}", -999))
        if d_l > best_delta:
            best_delta = d_l
            best_name = l_label
            best_idx = l_idx
            best_o_val = o_l
            best_b_val = b_l

        d_r = float(row.get(f"delta_oks_{r_col}", -999))
        o_r = float(row.get(f"oks_ours_{r_col}", -999))
        b_r = float(row.get(f"oks_base_{r_col}", -999))
        if d_r > best_delta:
            best_delta = d_r
            best_name = r_label
            best_idx = r_idx
            best_o_val = o_r
            best_b_val = b_r

    return {
        "swapped_joint_label": f"{best_name} (kp={best_idx})",
        "focus_kp_idx": best_idx,
        "kp_delta": best_delta,
        "kp_oks_base": best_b_val,
        "kp_oks_ours": best_o_val,
    }


# ==============================================================================
# Main Catalog Mining Queries
# ==============================================================================

def main() -> None:
    args = parse_args()
    dfs_by_stage = load_all_pipeline_parquets(args.stage, args.model)
    if not dfs_by_stage:
        logger.error("No parquet benchmark data could be loaded.")
        return

    # Master combined DF
    all_dfs = list(dfs_by_stage.values())
    master_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    results_catalog: Dict[str, Any] = {}

    # --------------------------------------------------------------------------
    # FIGURA 1: Disociación de Swaps Simétricos a Nivel Articular Específico
    # Target Stage: GMM + TTA / GMM + TTA + MRF
    # --------------------------------------------------------------------------
    print_section_header("Fig 1: Corrección de Swaps Simétricos (Foco en Keypoint Específico)")
    print("Concepto: Para el Zoom Articular de Fig 1, buscamos imágenes donde un keypoint específico colapsó en DARK (OKS~0.0)")
    print("          y nuestro GMM sobre el heatmap recuperó la articulación con precisión casi perfecta (OKS_Ours >= 0.85).")

    target_df_fig1 = dfs_by_stage.get("gmm_no_tta", dfs_by_stage.get("gmm_tta", master_df)).copy()
    swap_corr_col = "swaps_corrected_strict" if "swaps_corrected_strict" in target_df_fig1.columns else "swaps_corrected"
    swap_base_col = "swaps_base_strict" if "swaps_base_strict" in target_df_fig1.columns else "swaps_base"
    swap_ours_col = "swaps_ours_strict" if "swaps_ours_strict" in target_df_fig1.columns else "swaps_ours"

    # Extract per-joint best swap metrics for each image
    joint_metrics = target_df_fig1.apply(extract_best_swapped_joint, axis=1)
    target_df_fig1["swapped_joint"] = [m["swapped_joint_label"] for m in joint_metrics]
    target_df_fig1["focus_kp"] = [m["focus_kp_idx"] for m in joint_metrics]
    target_df_fig1["kp_delta"] = [m["kp_delta"] for m in joint_metrics]
    target_df_fig1["kp_oks_base"] = [m["kp_oks_base"] for m in joint_metrics]
    target_df_fig1["kp_oks_ours"] = [m["kp_oks_ours"] for m in joint_metrics]

    # Query: High per-joint Delta + High Joint Accuracy + Clean Global Pose
    query_fig1 = target_df_fig1[
        (target_df_fig1["kp_delta"] >= 0.40) &
        (target_df_fig1["kp_oks_ours"] >= 0.80) &
        (target_df_fig1.get("oks_ours", 0) >= 0.65)
    ].sort_values(by=["kp_delta", "oks_ours"], ascending=[False, False])

    cols_fig1 = [
        ("Image ID", "image_id", 10),
        ("Model", "model_arch", 13),
        ("Dataset", "dataset_name", 10),
        ("Experiment", "experiment_name", 18),
        ("Global OKS", "oks_ours", 11),
        ("Swapped Keypoint", "swapped_joint", 18),
        ("KP Base OKS", "kp_oks_base", 12),
        ("KP Ours OKS", "kp_oks_ours", 12),
        ("KP Delta OKS", "kp_delta", 13),
        ("Swaps (B->O)", swap_base_col, 13),
    ]
    display_candidates_table(query_fig1, cols_fig1, args.top_k)
    results_catalog["fig1_swaps_corrected"] = query_fig1.head(args.top_k).to_dict(orient="records")

    # Rescate específico en tobillos y muñecas (extremidades distales de alto impacto visual)
    print("\n  >> Top Casos en Extremidades Distales (Tobillos / Muñecas / Codos con Salto OKS >= +0.80):")
    query_distal = query_fig1[
        query_fig1["swapped_joint"].str.contains("Ankle|Wrist|Elbow", regex=True) &
        (query_fig1["kp_delta"] >= 0.80)
    ].sort_values(by="kp_delta", ascending=False)
    display_candidates_table(query_distal, cols_fig1, min(5, args.top_k))

    # --------------------------------------------------------------------------
    # FIGURA 2: El Dilema del MRF (Comparación directa GMM+TTA vs. GMM+TTA+MRF)
    # --------------------------------------------------------------------------
    print_section_header("Fig 2: El Dilema del MRF (Éxito del Muelle Cinemático vs. Limitación 2D)")
    print("Concepto 2A: MRF reajusta articulaciones ruidosas hacia la longitud anatómica ideal.")
    print("Concepto 2B: En oclusión severa, la rigidez 2D arrastra la extremidad hacia otra persona.")

    # Cross-Merge between GMM+TTA and GMM+TTA+MRF
    if "gmm_tta" in dfs_by_stage and "gmm_tta_mrf" in dfs_by_stage:
        df_tta = dfs_by_stage["gmm_tta"].copy()
        df_mrf = dfs_by_stage["gmm_tta_mrf"].copy()

        # Merge on identity
        merged_mrf = pd.merge(
            df_tta,
            df_mrf,
            on=["image_id", "dataset_name", "model_arch", "experiment_name"],
            suffixes=("_tta", "_mrf"),
            how="inner"
        )
        merged_mrf["oks_gmm_tta"] = merged_mrf["oks_ours_tta"]
        merged_mrf["oks_gmm_mrf"] = merged_mrf["oks_ours_mrf"]
        merged_mrf["delta_mrf_over_tta"] = merged_mrf["oks_gmm_mrf"] - merged_mrf["oks_gmm_tta"]

        PARENT_CHILD_BONES = [
            ("left_shoulder", "left_elbow", 5, 7, "L_Elbow (from L_Shoulder)"),
            ("right_shoulder", "right_elbow", 6, 8, "R_Elbow (from R_Shoulder)"),
            ("left_elbow", "left_wrist", 7, 9, "L_Wrist (from L_Elbow)"),
            ("right_elbow", "right_wrist", 8, 10, "R_Wrist (from R_Elbow)"),
            ("left_hip", "left_knee", 11, 13, "L_Knee (from L_Hip)"),
            ("right_hip", "right_knee", 12, 14, "R_Knee (from R_Hip)"),
            ("left_knee", "left_ankle", 13, 15, "L_Ankle (from L_Knee)"),
            ("right_knee", "right_ankle", 14, 16, "R_Ankle (from R_Knee)"),
        ]

        # 2A: Éxito Cinemático (Muelle restaura longitud ósea: TTA colapsa/jitter -> MRF perfecto)
        records_2a = []
        for idx, r in merged_mrf.iterrows():
            for p_name, c_name, p_idx, c_idx, label in PARENT_CHILD_BONES:
                p_ok = float(r.get(f"oks_ours_{p_name}_mrf", 0))
                c_tta = float(r.get(f"oks_ours_{c_name}_tta", 0))
                c_mrf = float(r.get(f"oks_ours_{c_name}_mrf", 0))
                if p_ok >= 0.80 and c_tta <= 0.60 and c_mrf >= 0.88 and (c_mrf - c_tta) >= 0.40:
                    records_2a.append({
                        "image_id": r["image_id"],
                        "model_arch": r["model_arch"],
                        "dataset_name": r["dataset_name"],
                        "experiment_name": r["experiment_name"],
                        "bone_label": f"{label} (kp={c_idx})",
                        "focus_kp": c_idx,
                        "parent_oks": p_ok,
                        "tta_joint_oks": c_tta,
                        "mrf_joint_oks": c_mrf,
                        "delta_joint": c_mrf - c_tta,
                        "global_mrf": r["oks_gmm_mrf"],
                    })

        query_fig2a = pd.DataFrame(records_2a).sort_values(by=["delta_joint", "global_mrf"], ascending=[False, False]) if records_2a else pd.DataFrame()

        cols_fig2a = [
            ("Image ID", "image_id", 10),
            ("Model", "model_arch", 13),
            ("Dataset", "dataset_name", 10),
            ("Experiment", "experiment_name", 18),
            ("Evaluated Bone", "bone_label", 30),
            ("Parent OKS", "parent_oks", 11),
            ("TTA Joint", "tta_joint_oks", 10),
            ("MRF Joint", "mrf_joint_oks", 10),
            ("Delta Bone", "delta_joint", 11),
            ("Global MRF", "global_mrf", 11),
        ]
        print("\n[Fig 2A - Éxito del MRF (Muelle Anatómico Restaura Longitud Ósea)]:")
        display_candidates_table(query_fig2a, cols_fig2a, args.top_k)
        results_catalog["fig2a_mrf_success"] = query_fig2a.head(args.top_k).to_dict(orient="records")

        # 2B: Limitación Cinemática por Deformación 2D / Escorzo
        records_2b = []
        for idx, r in merged_mrf.iterrows():
            for p_name, c_name, p_idx, c_idx, label in PARENT_CHILD_BONES:
                p_ok = float(r.get(f"oks_ours_{p_name}_tta", 0))
                c_tta = float(r.get(f"oks_ours_{c_name}_tta", 0))
                c_mrf = float(r.get(f"oks_ours_{c_name}_mrf", 0))
                if p_ok >= 0.70 and c_tta >= 0.80 and c_mrf <= 0.25 and (c_mrf - c_tta) <= -0.60:
                    records_2b.append({
                        "image_id": r["image_id"],
                        "model_arch": r["model_arch"],
                        "dataset_name": r["dataset_name"],
                        "experiment_name": r["experiment_name"],
                        "bone_label": f"{label} (kp={c_idx})",
                        "focus_kp": c_idx,
                        "parent_oks": p_ok,
                        "tta_joint_oks": c_tta,
                        "mrf_joint_oks": c_mrf,
                        "delta_joint": c_mrf - c_tta,
                        "global_tta": r["oks_gmm_tta"],
                    })

        query_fig2b = pd.DataFrame(records_2b).sort_values(by=["delta_joint", "global_tta"], ascending=[True, False]) if records_2b else pd.DataFrame()

        cols_fig2b = [
            ("Image ID", "image_id", 10),
            ("Model", "model_arch", 13),
            ("Dataset", "dataset_name", 10),
            ("Experiment", "experiment_name", 18),
            ("Evaluated Bone", "bone_label", 30),
            ("Parent OKS", "parent_oks", 11),
            ("TTA Joint", "tta_joint_oks", 10),
            ("MRF Joint", "mrf_joint_oks", 10),
            ("Delta Bone", "delta_joint", 11),
            ("Global TTA", "global_tta", 11),
        ]
        print("\n[Fig 2B - Limitación del MRF (Rigidez 2D arrastra la extremidad en Escorzo)]:")
        display_candidates_table(query_fig2b, cols_fig2b, args.top_k)
        results_catalog["fig2b_mrf_limitation"] = query_fig2b.head(args.top_k).to_dict(orient="records")

    # --------------------------------------------------------------------------
    # FIGURA 3: Alucinación Fantasma en Articulaciones Ausentes (vis = 0)
    # Target Stage: GMM (No TTA) o GMM+TTA
    # --------------------------------------------------------------------------
    print_section_header("Fig 3: Alucinación Fantasma en Articulaciones Ausentes (vis=0) vs. Alerta GMM")
    print("Concepto: Articulaciones completamente ausentes/fuera de cuadro (vis=0 y vis_opp=0) donde DARK alucina")
    print("          un punto en el fondo con alta confianza (P_dark >= 0.50), mientras que el GMM expande det(Sigma).")

    SYMMETRIC_PAIRS_MAP = {
        5: 6, 6: 5, 7: 8, 8: 7, 9: 10, 10: 9,
        11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15
    }

    target_df_fig3 = dfs_by_stage.get("gmm_no_tta", master_df)
    records_fig3 = []
    for idx, r in target_df_fig3.iterrows():
        if r.get("swaps_base", 0) > 0:
            continue
        global_oks = float(r.get("oks_ours", 0) or 0)
        if global_oks < 0.35:
            continue

        for k_idx in range(5, 17):
            k_name = COCO_KEYPOINT_NAMES[k_idx]
            opp_k_idx = SYMMETRIC_PAIRS_MAP.get(k_idx)
            opp_k_name = COCO_KEYPOINT_NAMES[opp_k_idx] if opp_k_idx is not None else None

            try:
                v_val = r.get(f"vis_{k_name}")
                vis = int(v_val) if v_val is not None and not pd.isna(v_val) else 1
                if vis != 0:
                    continue

                b_val = r.get(f"base_score_{k_name}")
                b_score = float(b_val) if b_val is not None and not pd.isna(b_val) else 0.0

                c_val = r.get(f"cov_det_{k_name}")
                cov_det = float(c_val) if c_val is not None and not pd.isna(c_val) else 0.0

                opp_vis = int(r.get(f"vis_{opp_k_name}", 1)) if opp_k_name else 1

                # Prioritize cases where BOTH sides are invisible (pure phantom hallucination)
                if b_score >= 0.45 and cov_det >= 1.0:
                    status = "Both Sides Absent" if opp_vis == 0 else "Absent (vis=0)"
                    records_fig3.append({
                        "image_id": r["image_id"],
                        "model_arch": r["model_arch"],
                        "dataset_name": r["dataset_name"],
                        "experiment_name": r["experiment_name"],
                        "joint_label": f"{k_name.replace('_', ' ').title()} (kp={k_idx})",
                        "vis_status": status,
                        "both_inv": (opp_vis == 0),
                        "focus_kp": k_idx,
                        "dark_conf": b_score,
                        "cov_det": cov_det,
                        "entropy": float(r.get("entropy", 0) or 0),
                        "global_oks": global_oks,
                    })
            except Exception:
                continue

    query_fig3 = pd.DataFrame(records_fig3).sort_values(by=["both_inv", "cov_det", "dark_conf"], ascending=[False, False, False]) if records_fig3 else pd.DataFrame()

    cols_fig3 = [
        ("Image ID", "image_id", 10),
        ("Model", "model_arch", 13),
        ("Dataset", "dataset_name", 10),
        ("Experiment", "experiment_name", 18),
        ("Absent Joint", "joint_label", 24),
        ("Visibility Status", "vis_status", 20),
        ("DARK Conf", "dark_conf", 10),
        ("Cov Det", "cov_det", 9),
        ("Global OKS", "global_oks", 10),
    ]
    display_candidates_table(query_fig3, cols_fig3, args.top_k)
    results_catalog["fig3_ood_volumetric_alert"] = query_fig3.head(args.top_k).to_dict(orient="records")

    # --------------------------------------------------------------------------
    # FIGURA 4: Mitigación del Heatmap Poisoning en TTA Continuo (Out-of-FoV Exits)
    # Target Stage: GMM + TTA
    # --------------------------------------------------------------------------
    print_section_header("Fig 4: Mitigación del Heatmap Poisoning en TTA Continuo (Out-of-FoV Exits)")
    print("Concepto: Escalas agresivas (s=0.85x) recortan keypoints fuera del bounding box por un margen moderado,")
    print("          envenenando el promedio ingenuo de DARK, mientras que la agregación adaptativa continua descarta el veneno.")

    # Load annotations for bbox and GT coordinates
    ann_root = project_root / "data"
    ann_files = {
        "coco": ann_root / "coco" / "annotations" / "person_keypoints_val2017.json",
        "crowdpose": ann_root / "crowdpose" / "json" / "crowdpose_val.json",
        "ochuman": ann_root / "ochuman" / "annotations" / "ochuman_coco_format_val_range_0.00_1.00.json",
    }
    annotations_by_ds = {}
    images_dims_by_ds = {}
    for ds_name, ann_path in ann_files.items():
        if ann_path.exists():
            with open(ann_path, "r", encoding="utf-8") as f:
                d_ann = json.load(f)
                img_anns = {}
                for ann in d_ann.get("annotations", []):
                    if ann.get("num_keypoints", 0) > 0 or "keypoints" in ann:
                        img_anns.setdefault(ann["image_id"], []).append(ann)
                annotations_by_ds[ds_name] = img_anns
                images_dims_by_ds[ds_name] = {im["id"]: (im.get("height", 1000), im.get("width", 1000)) for im in d_ann.get("images", [])}

    target_df_fig4 = dfs_by_stage.get("gmm_tta", master_df)
    records_fig4 = []

    for idx, r in target_df_fig4.iterrows():
        ds = str(r.get("dataset_name", "")).lower()
        iid = int(r.get("image_id", -1))
        if ds not in annotations_by_ds or iid not in annotations_by_ds[ds]:
            continue

        anns = annotations_by_ds[ds][iid]
        img_h, img_w = images_dims_by_ds[ds].get(iid, (1000, 1000))
        delta_ours = float(r.get("delta_oks_ours_over_tta", 0) or 0)
        delta_tta = float(r.get("delta_oks_tta", 0) or 0)
        oks_ours = float(r.get("oks_ours", 0) or 0)

        if delta_ours < 0.04 or oks_ours < 0.65:
            continue

        for ann in anns:
            bbox = ann.get("bbox")
            kps = np.array(ann.get("keypoints", [])).reshape(-1, 3)
            if bbox is None or len(bbox) < 4 or len(kps) < 17:
                continue

            bx, by, bw, bh = bbox
            if bw <= 15 or bh <= 15:
                continue

            # Check 0.85x scale bbox
            from src.pose_uncertainty.pipeline.scale_tta import scale_bbox
            sbx, sby, sbw, sbh = scale_bbox((bx, by, bw, bh), 0.85, img_h, img_w)

            for k_idx in range(min(17, len(kps))):
                kx, ky, kv = kps[k_idx]
                if kv == 0:
                    continue

                in_orig = (bx <= kx <= bx + bw) and (by <= ky <= by + bh)
                if not in_orig:
                    continue

                out_left = max(0.0, sbx - kx)
                out_right = max(0.0, kx - (sbx + sbw))
                out_top = max(0.0, sby - ky)
                out_bottom = max(0.0, ky - (sby + sbh))
                dist_out = np.sqrt(max(out_left, out_right)**2 + max(out_top, out_bottom)**2)

                # Moderate exit ("un cachito": between 4px and 35px)
                if 4.0 <= dist_out <= 35.0:
                    rel_dist = dist_out / max(bw, bh)
                    k_name = COCO_KEYPOINT_NAMES[k_idx]
                    hm_ent = r.get(f"heatmap_entropy_{k_name}")
                    if hm_ent is None or pd.isna(hm_ent):
                        hm_ent = r.get("heatmap_entropy_mean", r.get("entropy", 0.0))
                    hm_ent = float(hm_ent)

                    records_fig4.append({
                        "image_id": iid,
                        "model_arch": r["model_arch"],
                        "dataset_name": ds,
                        "experiment_name": r["experiment_name"],
                        "kp_idx": k_idx,
                        "kp_name": k_name.replace("_", " ").title(),
                        "dist_out_str": f"{dist_out:.1f}px ({rel_dist*100:.1f}%)",
                        "kp_entropy": hm_ent,
                        "oks_base": r["oks_base"],
                        "oks_tta": r["oks_tta"],
                        "oks_ours": r["oks_ours"],
                        "delta_oks_ours_over_tta": delta_ours,
                        "delta_oks_tta": delta_tta,
                    })

    query_fig4 = pd.DataFrame(records_fig4).drop_duplicates(subset=["image_id", "dataset_name", "model_arch", "experiment_name", "kp_idx"]).sort_values(
        by=["kp_entropy", "delta_oks_ours_over_tta"], ascending=[False, False]
    ) if records_fig4 else pd.DataFrame()

    cols_fig4 = [
        ("Image ID", "image_id", 10),
        ("Model", "model_arch", 13),
        ("Dataset", "dataset_name", 10),
        ("Experiment", "experiment_name", 18),
        ("Exiting Joint", "kp_name", 14),
        ("Exit Dist (0.85x)", "dist_out_str", 17),
        ("HM Entropy", "kp_entropy", 11),
        ("OKS Base", "oks_base", 9),
        ("OKS TTA", "oks_tta", 9),
        ("OKS Ours", "oks_ours", 9),
        ("Ours vs TTA", "delta_oks_ours_over_tta", 12),
    ]
    display_candidates_table(query_fig4, cols_fig4, args.top_k)
    results_catalog["fig4_heatmap_poisoning_resolved"] = query_fig4.head(args.top_k).to_dict(orient="records")

    # --------------------------------------------------------------------------
    # FIGURE 5: Visualization of Uniform Background Component (pi_uniform) under Resolution Degradation
    # Target: Gradual, monotonic growth of pi_uniform across 5 degradation tiers
    # --------------------------------------------------------------------------
    print_section_header("Fig 5: Uniform Background Component (pi_uniform) under Resolution Degradation")
    print("Concept: Monotonic progression of pi_uniform [Clean -> Low -> Med -> High -> Extreme]")
    print("         where intermediate tiers absorb noise and extreme tier exhibits substantial background mass.")

    target_df_fig5 = dfs_by_stage.get("gmm_no_tta", master_df)
    exp_order = ["00_Baseline_Clean", "01_Resize_Low", "02_Resize_Medium", "03_Resize_High", "04_Resize_Extreme"]
    dfs_by_exp = {exp: target_df_fig5[target_df_fig5["experiment_name"] == exp].set_index(["image_id", "dataset_name", "model_arch"]).to_dict(orient="index") for exp in exp_order}

    clean_dict = dfs_by_exp["00_Baseline_Clean"]
    records_fig5 = []

    coco_img_dir = Path(r"data/coco/val2017")
    ochuman_img_dir = Path(r"data/ochuman/images")
    crowdpose_img_dir = Path(r"data/crowdpose/images")

    for key, r_clean in clean_dict.items():
        iid, ds, m_arch = key
        ds = str(ds).lower()

        # Must exist across all 5 resolution tiers
        if not all(key in dfs_by_exp[exp] for exp in exp_order):
            continue

        oks_clean = float(r_clean.get("oks_ours", 0) or 0)
        if oks_clean < 0.6:
            continue

        if ds not in annotations_by_ds or iid not in annotations_by_ds[ds]:
            continue

        # Check local image file existence
        if ds == "coco":
            img_p = coco_img_dir / f"{iid:012d}.jpg"
        elif ds == "ochuman":
            img_p = ochuman_img_dir / f"{iid:06d}.jpg"
        elif ds == "crowdpose":
            img_p = crowdpose_img_dir / f"{iid}.jpg"
        else:
            img_p = Path("")

        if not img_p.exists():
            continue

        anns = annotations_by_ds[ds][iid]
        img_h, img_w = images_dims_by_ds[ds].get(iid, (1000, 1000))

        for ann in anns:
            bbox = ann.get("bbox")
            kps = np.array(ann.get("keypoints", [])).reshape(-1, 3)
            if bbox is None or len(bbox) < 4 or len(kps) < 17:
                continue
            bx, by, bw, bh = bbox
            area_ratio = (bw * bh) / max(1, img_w * img_h)

            # Prominent person criteria
            if bh < 220 or bw < 120 or area_ratio < 0.20 or area_ratio > 0.95:
                continue

            for k_idx in [10, 9, 16, 15, 14, 13, 6, 5]: # Wrists, Ankles, Knees, Shoulders
                kx, ky, kv = kps[k_idx]
                if kv < 2:
                    continue
                k_name = COCO_KEYPOINT_NAMES[k_idx]
                k_oks = float(r_clean.get(f"oks_ours_{k_name}", 0) or 0)
                if k_oks < 0.85:
                    continue

                pi_vec = []
                for exp in exp_order:
                    r_exp = dfs_by_exp[exp][key]
                    u_val = float(r_exp.get(f"uniform_weight_{k_name}", r_exp.get("uniform_weight_mean", 0.0)) or 0.0)
                    pi_vec.append(u_val)

                p0, p1, p2, p3, p4 = pi_vec

                # Criterio ingenioso de crecimiento progresivo y gradual:
                # 1. p4 >= 5% (masa visual clara en extremo)
                # 2. p4 > p3 y p3 >= p1 (tendencia estrictamente ascendente)
                # 3. Etapas intermedias activas (evitar que sea 0 en todas las etapas previas)
                if p4 >= 0.05 and p4 > p3 and p3 >= p1 and (p2 >= 0.002 or p3 >= 0.008):
                    diffs = np.diff(pi_vec)
                    neg_drops = sum(1 for d in diffs if d < -0.008)
                    if neg_drops == 0:
                        x = np.array([0, 1, 2, 3, 4])
                        y = np.array(pi_vec)
                        corr = np.corrcoef(x, y)[0, 1] if np.std(y) > 1e-6 else 0

                        # Gradualness score = correlacion * p4 * factor_distribucion_intermedia
                        inter_mass = (p1 + 2.0 * p2 + 3.0 * p3) / (6.0 * p4 + 1e-6)
                        gradual_score = corr * p4 * (0.3 + 0.7 * inter_mass)

                        records_fig5.append({
                            "image_id": iid,
                            "model_arch": m_arch,
                            "dataset_name": ds,
                            "joint_label": f"{k_name.replace('_', ' ').title()} (kp={k_idx})",
                            "focus_kp": k_idx,
                            "bbox_dim": f"{int(bw)}x{int(bh)} ({area_ratio*100:.1f}%)",
                            "area_ratio": area_ratio,
                            "progression": f"[{p0*100:.1f}% -> {p1*100:.1f}% -> {p2*100:.1f}% -> {p3*100:.1f}% -> {p4*100:.1f}%]",
                            "p0": p0, "p1": p1, "p2": p2, "p3": p3, "p4": p4,
                            "corr": corr,
                            "gradual_score": gradual_score,
                            "global_oks": oks_clean,
                        })

    query_fig5 = pd.DataFrame(records_fig5).drop_duplicates(subset=["image_id", "dataset_name", "focus_kp"]).sort_values(
        by=["gradual_score", "p4", "area_ratio"], ascending=[False, False, False]
    ) if records_fig5 else pd.DataFrame()

    cols_fig5 = [
        ("Image ID", "image_id", 10),
        ("Model", "model_arch", 13),
        ("Dataset", "dataset_name", 10),
        ("Evaluated Joint", "joint_label", 22),
        ("Uniform Progression (1.0x -> 0.06x)", "progression", 38),
        ("Gradual Score", "gradual_score", 14),
        ("BBox Size (% Area)", "bbox_dim", 20),
        ("Global OKS", "global_oks", 10),
    ]
    display_candidates_table(query_fig5, cols_fig5, args.top_k)
    results_catalog["fig5_uniform_background_absorption"] = query_fig5.head(args.top_k).to_dict(orient="records")

    # --------------------------------------------------------------------------
    # FIGURA 6: Consistencia Cross-Architecture (HRNet vs. ResNet50 vs. ViTPose)
    # --------------------------------------------------------------------------
    print_section_header("Fig 6: Consistencia Cross-Architecture (HRNet vs. ResNet50 vs. ViTPose)")
    print("Concepto: El mismo Image ID mejorado de forma robusta a través de los 3 modelos.")

    if "gmm_tta" in dfs_by_stage:
        df_models = dfs_by_stage["gmm_tta"]
        piv = df_models.pivot_table(
            index=["image_id", "dataset_name", "experiment_name"],
            columns="model_arch",
            values="delta_oks"
        ).dropna()

        if not piv.empty and "hrnet_w32" in piv.columns and "resnet50" in piv.columns and "vitpose_small" in piv.columns:
            piv["avg_delta"] = (piv["hrnet_w32"] + piv["resnet50"] + piv["vitpose_small"]) / 3.0
            piv = piv.sort_values(by="avg_delta", ascending=False).reset_index()

            print(f"\n[Fig 6 - Top Imágenes con Mejoras Simultáneas en los 3 Modelos]:")
            for _, r in piv.head(args.top_k).iterrows():
                iid = int(r["image_id"])
                ds = str(r["dataset_name"])
                exp = str(r["experiment_name"])
                d_hr = float(r["hrnet_w32"])
                d_res = float(r["resnet50"])
                d_vit = float(r["vitpose_small"])
                d_avg = float(r["avg_delta"])
                print(f"  ID: {iid:<8} | {ds:<10} | {exp:<18} | Delta HRNet: {d_hr:+.4f} | Delta ResNet: {d_res:+.4f} | Delta ViTPose: {d_vit:+.4f} | Media: {d_avg:+.4f}")
            results_catalog["fig6_cross_architecture"] = piv.head(args.top_k).to_dict(orient="records")

    # --------------------------------------------------------------------------
    # Save to JSON
    # --------------------------------------------------------------------------
    out_path = Path(args.out_json)
    with open(out_path, "w", encoding="utf-8") as f:
        clean_catalog = {}
        for fig_k, records in results_catalog.items():
            clean_records = []
            for r in records:
                clean_r = {}
                for k, v in r.items():
                    if isinstance(v, (np.integer, int)):
                        clean_r[k] = int(v)
                    elif isinstance(v, (np.floating, float)):
                        clean_r[k] = float(v) if not np.isnan(v) else None
                    elif isinstance(v, np.ndarray):
                        clean_r[k] = v.tolist()
                    else:
                        clean_r[k] = str(v)
                clean_records.append(clean_r)
            clean_catalog[fig_k] = clean_records
        json.dump(clean_catalog, f, indent=2)

    logger.info("\n[OK] Successfully saved Multi-Stage Figure Candidates Catalog to: %s", out_path.resolve())


if __name__ == "__main__":
    main()

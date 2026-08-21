"""
Generate Candidate Packets & Figures On-Demand from Parquet Selections.

This script uses the exact Hydra and EvaluationRunner pipeline to perform targeted
Deep Profiling on specific image_ids discovered from .parquet benchmark queries,
saving their full .pkl.gz packets and rendering publication-ready figures.

Supports selecting pipeline execution modes for both DARK and GMM decoding:
    - 'basic'   : Single-forward pass without TTA or MRF (DARK and GMM evaluated on basic single heatmap).
    - 'tta'     : Multi-scale TTA with continuous aggregation (GMM argmax mixture selection on aggregated heatmap).
    - 'tta_mrf' : Multi-scale TTA + MRF Belief Propagation tree decoding on anatomical bone length priors.

Usage:
    # Run in default TTA mode:
    poetry run python src/experiments/visualizations/generate_candidate_packets.py

    # Run in Basic (No TTA / Single-Image) mode:
    poetry run python src/experiments/visualizations/generate_candidate_packets.py +mode=basic

    # Run in TTA + MRF mode:
    poetry run python src/experiments/visualizations/generate_candidate_packets.py +mode=tta_mrf
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pose_uncertainty.models.adapters import create_model_adapter
from src.pose_uncertainty.evaluation.runner import EvaluationRunner, _build_dataloader
from src.pose_uncertainty.evaluation import storage
from src.experiments.visualizations.generate_all_paper_figures import (
    render_figure_1,
    render_figure_2,
    render_figure_2_single_candidate,
    render_figure_3,
    render_figure_4,
    render_figure_5,
)

logger = logging.getLogger(__name__)

DEGRADATION_CONFIGS = {
    "00_Baseline_Clean": {"dataset.resize_scale": 1.0, "dataset.noise_sigma": 0.0, "dataset.blur_kernel_size": 0},
    "01_Resize_Low": {"dataset.resize_scale": 0.5, "dataset.noise_sigma": 0.0, "dataset.blur_kernel_size": 0},
    "02_Resize_Medium": {"dataset.resize_scale": 0.25, "dataset.noise_sigma": 0.0, "dataset.blur_kernel_size": 0},
    "03_Resize_High": {"dataset.resize_scale": 0.125, "dataset.noise_sigma": 0.0, "dataset.blur_kernel_size": 0},
    "04_Resize_Extreme": {"dataset.resize_scale": 0.0625, "dataset.noise_sigma": 0.0, "dataset.blur_kernel_size": 0},
}

DATASET_CONFIGS = {
    "coco": {
        "dataset.name": "coco",
        "dataset.root_dir": "${paths.project_root}/data/coco",
        "dataset.annotations": "annotations/person_keypoints_val2017.json",
        "dataset.images": "val2017",
    },
    "crowdpose": {
        "dataset.name": "crowdpose",
        "dataset.root_dir": "${paths.project_root}/data/crowdpose",
        "dataset.annotations": "json/crowdpose_val.json",
        "dataset.images": "images",
    },
    "ochuman": {
        "dataset.name": "ochuman",
        "dataset.root_dir": "${paths.project_root}/data/ochuman",
        "dataset.annotations": "annotations/ochuman_coco_format_val_range_0.00_1.00.json",
        "dataset.images": "images",
    },
}

# Curated Parquet Candidates for all Figures
PARQUET_FIGURE_CANDIDATES = [
    # Fig 1: Swaps Simétricos con Salto Articular Extremo (Modo Básico / Sin TTA)
    #{"fig": "figura_1_swaps", "cand": "candidato_image_460", "image_id": 460, "dataset": "ochuman", "model": "resnet50", "exp": "02_Resize_Medium", "kp": 15, "mode": "basic"},
    #{"fig": "figura_1_swaps", "cand": "candidato_image_270908", "image_id": 270908, "dataset": "coco", "model": "hrnet_w32", "exp": "01_Resize_Low", "kp": 16, "mode": "basic"},
    #{"fig": "figura_1_swaps", "cand": "candidato_image_355", "image_id": 355, "dataset": "ochuman", "model": "resnet50", "exp": "03_Resize_High", "kp": 15, "mode": "basic"},
    #{"fig": "figura_1_swaps", "cand": "candidato_image_32", "image_id": 32, "dataset": "ochuman", "model": "hrnet_w32", "exp": "03_Resize_High", "kp": 5, "mode": "basic"},
    #{"fig": "figura_1_swaps", "cand": "candidato_image_230", "image_id": 230, "dataset": "ochuman", "model": "vitpose_small", "exp": "00_Baseline_Clean", "kp": 15, "mode": "basic"},
    #{"fig": "figura_1_swaps", "cand": "candidato_4_image_116209", "image_id": 116209, "dataset": "crowdpose", "model": "vitpose_small", "exp": "00_Baseline_Clean", "kp": 10, "mode": "basic"},

    # ==========================================================================
    # FIG 3: ARTICULACIÓN AUSENTE (VIS=0) CON K=1 GAUSSIANA Y ALTO ARGMAX (GENERATED)
    # ==========================================================================
    #{"fig": "figura_3_ood_alert", "cand": "candidato_1_image_467511", "image_id": 467511, "dataset": "coco", "model": "vitpose_small", "exp": "02_Resize_Medium", "kp": 2, "mode": "basic"},
    #{"fig": "figura_3_ood_alert", "cand": "candidato_2_image_108525", "image_id": 108525, "dataset": "crowdpose", "model": "hrnet_w32", "exp": "00_Baseline_Clean", "kp": 1, "mode": "basic"},
    #{"fig": "figura_3_ood_alert", "cand": "candidato_3_image_243", "image_id": 243, "dataset": "ochuman", "model": "vitpose_small", "exp": "01_Resize_Low", "kp": 1, "mode": "basic"},
    #{"fig": "figura_3_ood_alert", "cand": "candidato_4_image_17207", "image_id": 17207, "dataset": "coco", "model": "vitpose_small", "exp": "00_Baseline_Clean", "kp": 1, "mode": "basic"},
    #{"fig": "figura_3_ood_alert", "cand": "candidato_5_image_508312", "image_id": 508312, "dataset": "coco", "model": "vitpose_small", "exp": "00_Baseline_Clean", "kp": 10, "mode": "basic"},


    # Fig 3: OoD Alert
    #{"fig": "figura_3_ood_alert", "cand": "candidato_1_image_108463", "image_id": 108463, "dataset": "crowdpose", "model": "resnet50", "exp": "01_Resize_Low", "kp": 9},
    #{"fig": "figura_3_ood_alert", "cand": "candidato_2_image_34", "image_id": 34, "dataset": "ochuman", "model": "vitpose_small", "exp": "02_Resize_Medium", "kp": 9},

    # ==========================================================================
    # FIG 4: MITIGACIÓN DEL HEATMAP POISONING (MAXIMUM ENTROPY SPIKES & FOV EXITS)
    # ==========================================================================
    #{
    #    "fig": "figura_4_heatmap_poisoning",
    #    "cand": "candidato_1_image_10",
    #    "image_id": 10,
    #    "dataset": "ochuman",
    #    "model": "vitpose_small",
    #    "exp": "00_Baseline_Clean",
    #    "kp": 10,  # Right Wrist (Exits 0.85x bbox by 14.7px, HM Entropy=6.91 nats, OKS Base=0.641, TTA=0.729, Ours=0.869, Delta=+0.140)
    #    "mode": "tta",
    #},
    #{
    #    "fig": "figura_4_heatmap_poisoning",
    #    "cand": "candidato_2_image_251",
    #    "image_id": 251,
    #    "dataset": "ochuman",
    #    "model": "vitpose_small",
    #    "exp": "03_Resize_High",
    #    "kp": 13,  # Left Knee (Exits 0.85x bbox by 5.1px, HM Entropy=7.05 nats, OKS Base=0.799, TTA=0.750, Ours=0.821, Delta=+0.071)
    #    "mode": "tta",
    #},
    #{
    #    "fig": "figura_4_heatmap_poisoning",
    #    "cand": "candidato_3_image_361238",
    #    "image_id": 361238,
    #    "dataset": "coco",
    #    "model": "vitpose_small",
    #    "exp": "02_Resize_Medium",
    #    "kp": 9,   # Left Wrist (Exits 0.85x bbox by 16.4px, HM Entropy=6.92 nats, OKS Base=0.724, TTA=0.679, Ours=0.742, Delta=+0.063)
    #    "mode": "tta",
    #},
    #{
    #    "fig": "figura_4_heatmap_poisoning",
    #    "cand": "candidato_4_image_471",
    #    "image_id": 471,
    #    "dataset": "ochuman",
    #    "model": "hrnet_w32",
    #    "exp": "01_Resize_Low",
    #    "kp": 16,  # Right Ankle (Exits 0.85x bbox by 30.0px, HM Entropy=7.04 nats, OKS Base=0.831, TTA=0.835, Ours=0.891, Delta=+0.056)
    #    "mode": "tta",
    #},
    #{
    #    "fig": "figura_4_heatmap_poisoning",
    #    "cand": "candidato_5_image_345361",
    #    "image_id": 345361,
    #    "dataset": "coco",
    #    "model": "hrnet_w32",
    #    "exp": "03_Resize_High",
    #    "kp": 13,  # Left Knee (Exits 0.85x bbox by 8.9px, HM Entropy=6.90 nats, OKS Base=0.673, TTA=0.698, Ours=0.756, Delta=+0.059)
    #    "mode": "tta",
    #},

    # Fig 5: Uniform Trash & Resolution Progression (Top Gradual Monotonic Growth Across Scales)
    #{"fig": "figura_5_uniform_trash", "cand": "candidato_1_image_238", "image_id": 238, "dataset": "ochuman", "model": "hrnet_w32", "exp": "00_Baseline_Clean", "kp": 16, "mode": "basic"},
    #{"fig": "figura_5_uniform_trash", "cand": "candidato_2_image_402118", "image_id": 402118, "dataset": "coco", "model": "hrnet_w32", "exp": "00_Baseline_Clean", "kp": 13, "mode": "basic"},
    #{"fig": "figura_5_uniform_trash", "cand": "candidato_3_image_482", "image_id": 482, "dataset": "ochuman", "model": "hrnet_w32", "exp": "00_Baseline_Clean", "kp": 15, "mode": "basic"},
    {
        "fig": "figura_1_swaps",
        "cand": "01_MEJORADA_candidato_image_460",
        "image_id": 460,
        "dataset": "ochuman",
        "model": "resnet50",
        "exp": "02_Resize_Medium",
        "kp": 15,          # Articulación enfocada (tobillo/left_ankle)
        "mode": "basic",   # Modo básico sin TTA para mostrar el swap simétrico puro
    },
]


def configure_pipeline_mode(exp_cfg: DictConfig, mode: str) -> None:
    """
    Configure the Hydra configuration tree to execute in one of three pipeline modes:
    - 'basic'   : Single-forward pass without TTA or MRF. DARK and GMM are evaluated on the basic heatmap.
    - 'tta'     : Multi-scale test-time augmentation enabled. GMM is fitted on the aggregated heatmap (argmax).
    - 'tta_mrf' : Multi-scale TTA enabled + MRF Belief Propagation tree decoding for kinematic bone length priors.
    """
    m = str(mode).lower().strip()
    with open_dict(exp_cfg):
        if m in ["basic", "no_tta", "gmm_no_tta", "baseline"]:
            # Disable TTA completely
            OmegaConf.update(exp_cfg, "tta.enabled", False, force_add=True)
            OmegaConf.update(exp_cfg, "tta.scale.enabled", False, force_add=True)
            OmegaConf.update(exp_cfg, "tta.scale.scales", [1.0], force_add=True)
            OmegaConf.update(exp_cfg, "tta.flip.enabled", False, force_add=True)
            OmegaConf.update(exp_cfg, "scale.enabled", False, force_add=True)
            OmegaConf.update(exp_cfg, "scale.scales", [1.0], force_add=True)
            OmegaConf.update(exp_cfg, "flip.enabled", False, force_add=True)
            OmegaConf.update(exp_cfg, "enabled", False, force_add=True)

            # Disable MRF
            OmegaConf.update(exp_cfg, "decode_strategy", "argmax", force_add=True)
            OmegaConf.update(exp_cfg, "mixture_model.decode_strategy", "argmax", force_add=True)
            OmegaConf.update(exp_cfg, "math_core.mixture_model.decode_strategy", "argmax", force_add=True)

        elif m in ["tta", "gmm_tta"]:
            # Enable TTA (Multi-scale + Horizontal Flip)
            OmegaConf.update(exp_cfg, "tta.enabled", True, force_add=True)
            OmegaConf.update(exp_cfg, "tta.scale.enabled", True, force_add=True)
            OmegaConf.update(exp_cfg, "tta.scale.scales", [0.85, 0.925, 1.0, 1.075, 1.15], force_add=True)
            OmegaConf.update(exp_cfg, "tta.flip.enabled", True, force_add=True)
            OmegaConf.update(exp_cfg, "scale.enabled", True, force_add=True)
            OmegaConf.update(exp_cfg, "scale.scales", [0.85, 0.925, 1.0, 1.075, 1.15], force_add=True)
            OmegaConf.update(exp_cfg, "flip.enabled", True, force_add=True)
            OmegaConf.update(exp_cfg, "enabled", True, force_add=True)

            # Disable MRF (argmax mixture selection)
            OmegaConf.update(exp_cfg, "decode_strategy", "argmax", force_add=True)
            OmegaConf.update(exp_cfg, "mixture_model.decode_strategy", "argmax", force_add=True)
            OmegaConf.update(exp_cfg, "math_core.mixture_model.decode_strategy", "argmax", force_add=True)

        elif m in ["tta_mrf", "gmm_tta_mrf", "mrf"]:
            # Enable TTA
            OmegaConf.update(exp_cfg, "tta.enabled", True, force_add=True)
            OmegaConf.update(exp_cfg, "tta.scale.enabled", True, force_add=True)
            OmegaConf.update(exp_cfg, "tta.scale.scales", [0.85, 0.925, 1.0, 1.075, 1.15], force_add=True)
            OmegaConf.update(exp_cfg, "tta.flip.enabled", True, force_add=True)
            OmegaConf.update(exp_cfg, "scale.enabled", True, force_add=True)
            OmegaConf.update(exp_cfg, "scale.scales", [0.85, 0.925, 1.0, 1.075, 1.15], force_add=True)
            OmegaConf.update(exp_cfg, "flip.enabled", True, force_add=True)
            OmegaConf.update(exp_cfg, "enabled", True, force_add=True)

            # Enable MRF graph decoding
            OmegaConf.update(exp_cfg, "decode_strategy", "mrf_graph", force_add=True)
            OmegaConf.update(exp_cfg, "mixture_model.decode_strategy", "mrf_graph", force_add=True)
            OmegaConf.update(exp_cfg, "math_core.mixture_model.decode_strategy", "mrf_graph", force_add=True)
        else:
            raise ValueError(f"Unknown pipeline mode: '{mode}'. Must be one of: 'basic', 'tta', 'tta_mrf'")


def profile_single_candidate(
    cfg: DictConfig,
    cand_info: Dict[str, Any],
    out_dir: Path,
    mode: str = "tta"
) -> Optional[Path]:
    """Execute deep profiling for one specific candidate and save packet."""
    img_id = cand_info["image_id"]
    ds_name = cand_info["dataset"]
    m_name = cand_info["model"]
    exp_name = cand_info.get("exp", "00_Baseline_Clean")
    cand_mode = cand_info.get("mode", mode)

    logger.info("=== Deep Profiling: Image ID %d | Dataset: %s | Model: %s | Exp: %s | Mode: %s ===", img_id, ds_name, m_name, exp_name, cand_mode)

    exp_cfg = copy.deepcopy(cfg)
    with open_dict(exp_cfg):
        # Apply dataset overrides
        for k, v in DATASET_CONFIGS[ds_name].items():
            OmegaConf.update(exp_cfg, k, v)
        # Apply degradation overrides
        if exp_name in DEGRADATION_CONFIGS:
            for k, v in DEGRADATION_CONFIGS[exp_name].items():
                OmegaConf.update(exp_cfg, k, v)
        # Apply model override
        exp_cfg.model.name = m_name
        exp_cfg.logging.output_dir = str(out_dir)

    # Configure pipeline mode (basic, tta, or tta_mrf)
    configure_pipeline_mode(exp_cfg, cand_mode)

    # Initialize model
    device = OmegaConf.select(exp_cfg, "model.device", default="cuda")
    model = create_model_adapter(m_name, device=device)

    # Build dataloader
    dataloader = _build_dataloader(exp_cfg)

    # Initialize runner
    runner = EvaluationRunner(exp_cfg, dataloader=dataloader)
    runner.model = model
    runner.output_dir = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run targeted profiling
    target_dict = {f"candidate_{img_id}": [img_id]}
    saved = runner.run_deep_profiling(target_dict)

    if saved:
        logger.info("[SUCCESS] Saved packet for ID %d -> %s", img_id, saved[0])
        return saved[0]
    else:
        logger.warning("[!] Could not find Image ID %d in dataset %s", img_id, ds_name)
        return None


@hydra.main(config_path="../../../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    target_output_base = (
        project_root / "outputs" / "figures" / "visualizations"
    )
    target_output_base.mkdir(parents=True, exist_ok=True)

    # Global pipeline mode option: "basic", "tta", or "tta_mrf" (can be overridden via `mode=basic` or `+mode=basic`)
    global_mode = str(OmegaConf.select(cfg, "mode", default="tta"))

    logger.info("=" * 80)
    logger.info("ON-DEMAND PACKET GENERATOR & RENDERER FOR PARQUET CANDIDATES")
    logger.info("Pipeline Execution Mode: %s", global_mode.upper())
    logger.info("=" * 80)

    for item in PARQUET_FIGURE_CANDIDATES:
        fig_sub = item["fig"]
        cand_sub = item["cand"]
        cand_mode = item.get("mode", global_mode)
        dest_folder = target_output_base / fig_sub / cand_sub

        # 1. Profile sample and save .pkl.gz packet
        pkt_path = profile_single_candidate(cfg, item, dest_folder, mode=cand_mode)
        if pkt_path and pkt_path.exists():
            pkt = storage.load_deep_analysis(str(pkt_path))
            focus_kp = item.get("kp", 16)

            # 2. Render Figure
            if fig_sub == "figura_1_swaps":
                render_figure_1(pkt, focus_kp, dest_folder, cand_sub, mode=cand_mode)
            elif fig_sub == "figura_2_mrf_dilemma":
                render_figure_2_single_candidate(pkt, focus_kp, dest_folder, cand_sub, mode=cand_mode)
            elif fig_sub == "figura_3_ood_alert":
                render_figure_3(pkt, focus_kp, dest_folder, cand_sub, mode=cand_mode)
            elif fig_sub == "figura_4_heatmap_poisoning":
                render_figure_4(pkt, focus_kp, dest_folder, cand_sub, mode=cand_mode)
            elif fig_sub == "figura_5_uniform_trash":
                render_figure_5(pkt, dest_folder, cand_sub, mode=cand_mode, focus_kp_idx=focus_kp)

    logger.info("=" * 80)
    logger.info("[COMPLETE] All targeted parquet candidate packets & figures generated!")
    logger.info("Check directory: %s", target_output_base)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

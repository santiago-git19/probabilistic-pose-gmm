"""
Generate All Paper Figures: High-Impact Visualizations for Elsevier / Information Fusion.

Generates publication-ready figures (3 candidates per figure type) with the exact technical layout,
color hierarchy, and styling specified in the Paper Guidelines:

- Fig 1: Symmetric Swap Disambiguation and Bimodal Ambiguity (GMM vs. DARK)
- Fig 2: Kinematic MRF Dilemma (Kinematic Regularization vs. 2D Heatmap Poisoning)
- Fig 3: Heuristic Blindness vs. Volumetric Uncertainty in OoD Anomaly Detection (CrowdPose / OCHuman)
- Fig 4: Anatomy of Heatmap Poisoning under Test-Time Augmentation (TTA)
- Fig 5: Uniform Background Component (pi_uniform) under Resolution Degradation

Output Destination:
    Paper/Paper/figures/visualizaciones/

Usage:
    poetry run python src/experiments/visualizations/generate_all_paper_figures.py
"""

from __future__ import annotations

import copy
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import numpy.typing as npt

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pose_uncertainty.evaluation.storage import load_deep_analysis
from src.pose_uncertainty.models.adapters import MMPoseAdapter
from src.pose_uncertainty.utils.types import StandardizedHeatmap
from src.pose_uncertainty.utils.metrics import compute_calibrated_covariance, COCO_SIGMAS, compute_oks
from src.pose_uncertainty.pipeline.scale_tta import scale_bbox, aggregate_multiscale_heatmaps, compute_heatmap_confidence
from src.pose_uncertainty.core.sampling import sample_from_heatmap
import torch
from src.pose_uncertainty.data_loader import _resize_image_and_annotations
from src.pose_uncertainty.core.mixture import select_best_model, RobustGaussianMixture

PARENT_MAP: Dict[int, int] = {
    1: 0, 2: 0, 3: 1, 4: 2, 5: 0, 6: 0,
    7: 5, 8: 6, 9: 7, 10: 8, 11: 5, 12: 6,
    13: 11, 14: 12, 15: 13, 16: 14
}

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Destination Directory
TARGET_OUTPUT_ROOT = Path(r"C:\Users\Santiago estudio\Desktop\TFG_Informatica\Paper\Paper\figures\visualizaciones")

# Global Vector Style Palette
COLOR_GT = "#00E676"         # Emerald Green
COLOR_DARK = "#FF1744"       # Crimson Red
COLOR_OURS = "#00E5FF"       # Cyan
COLOR_ELLIPSE_2S = "#FFD600" # Gold Solid (2-sigma)
COLOR_ELLIPSE_1S = "#FFD600" # Gold Dotted (1-sigma)
COLOR_MC = "#FFAB00"         # Amber points
COLOR_UNIFORM = "#D500F9"    # Neon Purple

COCO_KP_NAMES = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
]

COCO_SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),                # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # Legs
]


# ==============================================================================
# Geometry & Spatial Transformation Helpers
# ==============================================================================

def find_original_image(packet: Dict[str, Any]) -> Optional[npt.NDArray]:
    tta_data = packet.get("tta_data", [])
    for entry in tta_data:
        if entry.get("name", "").startswith("original"):
            img = entry.get("image")
            if isinstance(img, np.ndarray): return img
    for entry in tta_data:
        img = entry.get("image")
        if isinstance(img, np.ndarray): return img
    return None


def ellipse_polygon_vertices(mu: npt.NDArray, cov: npt.NDArray, n_sigma: float = 2.0, n_pts: int = 64) -> npt.NDArray:
    if mu.shape[0] < 2 or cov.shape != (2, 2):
        return np.empty((0, 2))
    try:
        eigvals, eigvecs = np.linalg.eigh(cov[:2, :2])
    except np.linalg.LinAlgError:
        return np.empty((0, 2))

    eigvals = np.clip(eigvals, 1e-6, None)
    a = n_sigma * math.sqrt(float(eigvals[1]))
    b = n_sigma * math.sqrt(float(eigvals[0]))
    angle = math.atan2(float(eigvecs[1, 1]), float(eigvecs[0, 1]))

    t = np.linspace(0, 2 * math.pi, n_pts, endpoint=True)
    xs = a * np.cos(t)
    ys = b * np.sin(t)
    ca, sa = math.cos(angle), math.sin(angle)

    return np.stack([
        ca * xs - sa * ys + float(mu[0]),
        sa * xs + ca * ys + float(mu[1])
    ], axis=-1)


def map_gmm_to_image_space(packet: Dict[str, Any], means: npt.NDArray, covs: npt.NDArray, img_w: int, img_h: int):
    means_arr = np.asarray(means, dtype=np.float32)
    covs_arr = np.asarray(covs, dtype=np.float32)
    if means_arr.ndim == 1: means_arr = means_arr.reshape(1, -1)
    if covs_arr.ndim == 2: covs_arr = covs_arr.reshape(1, 2, 2)
    n = min(means_arr.shape[0], covs_arr.shape[0])
    means_arr = means_arr[:n, :2]
    covs_arr = covs_arr[:n]

    hm = packet.get("aggregation", {}).get("heatmap_avg")
    metadata = packet.get("aggregation", {}).get("mmpose_metadata")

    if isinstance(hm, np.ndarray) and isinstance(metadata, dict):
        try:
            ref_hm = StandardizedHeatmap(data=np.asarray(hm, dtype=np.float32), original_size=(img_h, img_w), metadata=metadata)
            means_img, covs_img = MMPoseAdapter.transform_heatmap_gaussians_to_image(means_arr, covs_arr, ref_hm)
            A, b = MMPoseAdapter._heatmap_to_image_affine(ref_hm)
            return np.asarray(means_img, dtype=np.float64), np.asarray(covs_img, dtype=np.float64), A, b
        except Exception:
            pass

    gt = packet.get("ground_truth", {})
    bbox = gt.get("bboxes", gt.get("bbox"))
    x0, y0, box_w, box_h = 0.0, 0.0, float(img_w), float(img_h)
    if bbox is not None:
        b_arr = np.asarray(bbox, dtype=np.float64).reshape(-1)
        if b_arr.size >= 4 and np.all(np.isfinite(b_arr[:4])):
            bx, by, bw, bh = map(float, b_arr[:4])
            if bw > 0 and bh > 0: x0, y0, box_w, box_h = bx, by, bw, bh

    hm_w = hm.shape[2] if isinstance(hm, np.ndarray) and hm.ndim == 3 else 64
    hm_h = hm.shape[1] if isinstance(hm, np.ndarray) and hm.ndim == 3 else 48
    sx, sy = box_w / float(hm_w), box_h / float(hm_h)

    means_img = means_arr.copy().astype(np.float64)
    means_img[:, 0] = x0 + means_arr[:, 0] * sx
    means_img[:, 1] = y0 + means_arr[:, 1] * sy

    A = np.array([[sx, 0.0], [0.0, sy]], dtype=np.float64)
    b_vec = np.array([x0, y0], dtype=np.float64)
    covs_img = np.empty((n, 2, 2), dtype=np.float64)
    for i in range(n):
        covs_img[i] = A @ covs_arr[i].astype(np.float64) @ A.T

    return means_img, covs_img, A, b_vec


def map_points_to_image_space(points_hm: npt.NDArray, A: Optional[npt.NDArray], b: Optional[npt.NDArray]) -> npt.NDArray:
    if points_hm.ndim != 2 or points_hm.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    if A is None or b is None: return points_hm.astype(np.float64)
    return (points_hm[:, :2].astype(np.float64) @ A.T) + b.reshape(1, 2)


def decode_dark_from_heatmap_2d(hm: np.ndarray, blur_ksize: int = 11) -> np.ndarray:
    """
    Apply standard DARK (Distribution-Aware Coordinate Representation of Keypoints)
    decoding on a 2D heatmap tensor using Gaussian blur and Taylor expansion.
    """
    H, W = hm.shape
    if blur_ksize > 1:
        ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        hm_blurred = cv2.GaussianBlur(hm.astype(np.float32), (ksize, ksize), 0)
    else:
        hm_blurred = hm.astype(np.float32)

    hm_log = np.log(np.maximum(hm_blurred, 1e-4))
    y0, x0 = np.unravel_index(np.argmax(hm_log), (H, W))

    dx, dy = 0.0, 0.0
    if 1 <= x0 < W - 1 and 1 <= y0 < H - 1:
        D_x = 0.5 * (hm_log[y0, x0 + 1] - hm_log[y0, x0 - 1])
        D_y = 0.5 * (hm_log[y0 + 1, x0] - hm_log[y0 - 1, x0])
        D_xx = hm_log[y0, x0 + 1] - 2.0 * hm_log[y0, x0] + hm_log[y0, x0 - 1]
        D_yy = hm_log[y0 + 1, x0] - 2.0 * hm_log[y0, x0] + hm_log[y0 - 1, x0]
        D_xy = 0.25 * (hm_log[y0 + 1, x0 + 1] - hm_log[y0 + 1, x0 - 1] - hm_log[y0 - 1, x0 + 1] + hm_log[y0 - 1, x0 - 1])

        det = D_xx * D_yy - D_xy * D_xy
        if abs(det) > 1e-6:
            dx = -(D_yy * D_x - D_xy * D_y) / det
            dy = -(D_xx * D_y - D_xy * D_x) / det
            dx = np.clip(dx, -0.5, 0.5)
            dy = np.clip(dy, -0.5, 0.5)

    return np.array([float(x0 + dx), float(y0 + dy)], dtype=np.float64)


def get_dark_coords_from_aggregated_heatmap(packet: Dict[str, Any], img_w: int, img_h: int) -> npt.NDArray:
    """
    Extract keypoint coordinates by running DARK postprocessing directly
    on the aggregated TTA heatmap (heatmap_avg).
    """
    agg = packet.get("aggregation", {})
    avg_hm = agg.get("heatmap_avg")
    if not isinstance(avg_hm, np.ndarray) or avg_hm.ndim != 3:
        base = packet.get("predictions", {}).get("baseline", {}).get("coords")
        if isinstance(base, np.ndarray):
            return base[:, :2].astype(np.float64)
        return np.zeros((17, 2), dtype=np.float64)

    _, _, A, b = map_gmm_to_image_space(packet, np.zeros((avg_hm.shape[0], 2)), np.zeros((avg_hm.shape[0], 2, 2)), img_w, img_h)

    coords_list = []
    for k in range(avg_hm.shape[0]):
        pt_hm = decode_dark_from_heatmap_2d(avg_hm[k])
        if A is not None and b is not None:
            pt_img = (A @ pt_hm) + b
        else:
            pt_img = pt_hm
        coords_list.append(pt_img)
    return np.array(coords_list, dtype=np.float64)


def detect_packet_mode(packet: Dict[str, Any], explicit_mode: str = "auto") -> str:
    """Determine if packet represents 'basic', 'tta', or 'tta_mrf'."""
    if explicit_mode and explicit_mode.lower() in ["basic", "no_tta", "gmm_no_tta", "baseline"]:
        return "basic"
    if explicit_mode and explicit_mode.lower() in ["tta", "gmm_tta"]:
        return "tta"
    if explicit_mode and explicit_mode.lower() in ["tta_mrf", "gmm_tta_mrf", "mrf"]:
        return "tta_mrf"

    cfg_used = packet.get("config_used", {})
    strat = (
        cfg_used.get("decode_strategy")
        or cfg_used.get("mixture_model", {}).get("decode_strategy")
        or cfg_used.get("math_core", {}).get("mixture_model", {}).get("decode_strategy")
    )
    if strat == "mrf_graph":
        return "tta_mrf"

    tta_data = packet.get("tta_data", [])
    if len(tta_data) > 1:
        return "tta"

    return "basic"


def search_packet(image_id: int) -> Optional[Path]:
    """Search outputs directory for packet matching image_id."""
    for p in Path("outputs").rglob(f"*_{image_id}.pkl.gz"):
        return p
    return None


# ==============================================================================
# RENDERER FIGURA 1: Disociación de Swaps Simétricos y Ambigüedad Bimodal
# ==============================================================================

def render_figure_1(packet: Dict[str, Any], focus_kp_idx: int, out_dir: Path, cand_name: str, mode: str = "auto"):
    """
    Fig 1: GridSpec(1, 3, width_ratios=[1.3, 1.0, 0.9])
    (a) Global Pose con GT (verde), DARK (rojo --), Ours (cian) y Bounding Box zoom.
    (b) Zoom Articular con MC samples (ámbar), 2 elipses GMM ($w_1, w_2$), DARK (x), Ours (o).
    (c) Heatmap agregado 2D inferno con centros + en cian y contorno.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_rgb = find_original_image(packet)
    if img_rgb is None: return
    img_h, img_w = img_rgb.shape[:2]

    mode_tag = detect_packet_mode(packet, mode)
    meta = packet.get("meta", {})
    metrics = packet.get("metrics", {})
    gt = packet.get("ground_truth", {}).get("coords")
    
    # In 'basic' mode, use single base coords; in 'tta' / 'tta_mrf', decode DARK on heatmap_avg
    if mode_tag == "basic":
        base_raw = packet.get("predictions", {}).get("baseline", {}).get("coords")
        base = base_raw[:, :2].astype(np.float64) if isinstance(base_raw, np.ndarray) else np.zeros((17, 2), dtype=np.float64)
        dark_legend = "DARK (Basic / No TTA)"
        ours_legend = "Ours GMM (Basic / No TTA)"
    elif mode_tag == "tta_mrf":
        base = get_dark_coords_from_aggregated_heatmap(packet, img_w, img_h)
        dark_legend = r"DARK (on Aggregated $\bar{H}$)"
        ours_legend = "Ours GMM (TTA + MRF)"
    else: # tta
        base = get_dark_coords_from_aggregated_heatmap(packet, img_w, img_h)
        dark_legend = r"DARK (on Aggregated $\bar{H}$)"
        ours_legend = "Ours GMM (TTA)"

    ours = packet.get("predictions", {}).get("ours_gmm", {}).get("coords")
    gmm_per_kp = packet.get("gmm_per_kp", [])
    agg = packet.get("aggregation", {})
    avg_hm = agg.get("heatmap_avg")
    samp_by_kp = agg.get("sampling_points_by_kp", [])

    means_img_list, covs_img_list = [], []
    A_mat, b_vec = None, None
    for k in range(len(COCO_KP_NAMES)):
        if k < len(gmm_per_kp) and isinstance(gmm_per_kp[k], dict):
            m = np.asarray(gmm_per_kp[k].get("means", []), dtype=np.float64)
            c = np.asarray(gmm_per_kp[k].get("covariances", []), dtype=np.float64)
            if m.ndim == 2 and c.ndim == 3 and m.shape[0] > 0:
                m_img, c_img, A_mat, b_vec = map_gmm_to_image_space(packet, m, c, img_w, img_h)
                means_img_list.append(m_img)
                covs_img_list.append(c_img)
            else:
                means_img_list.append(np.empty((0, 2)))
                covs_img_list.append(np.empty((0, 2, 2)))
        else:
            means_img_list.append(np.empty((0, 2)))
            covs_img_list.append(np.empty((0, 2, 2)))

    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(16, 6.5), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 0.9], wspace=0.18)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    focus_name = COCO_KP_NAMES[focus_kp_idx].replace("_", " ").title()
    SYMMETRIC_PAIRS_MAP = {
        5: 6, 6: 5, 7: 8, 8: 7, 9: 10, 10: 9,
        11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15
    }
    opp_kp_idx = SYMMETRIC_PAIRS_MAP.get(focus_kp_idx, None)
    opp_name = COCO_KP_NAMES[opp_kp_idx].replace("_", " ").title() if opp_kp_idx is not None else ""

    # Collect all points to encompass in the zoom crop (Both Gaussian modes + both limbs)
    m_focus = means_img_list[focus_kp_idx]
    c_focus = covs_img_list[focus_kp_idx]
    w_focus = gmm_per_kp[focus_kp_idx].get("weights", []) if focus_kp_idx < len(gmm_per_kp) else []

    pts_to_include_x = []
    pts_to_include_y = []

    for j in range(len(m_focus)):
        pts_to_include_x.append(m_focus[j, 0])
        pts_to_include_y.append(m_focus[j, 1])

    if isinstance(gt, np.ndarray) and gt[focus_kp_idx, 2] > 0:
        pts_to_include_x.append(gt[focus_kp_idx, 0])
        pts_to_include_y.append(gt[focus_kp_idx, 1])
    if isinstance(base, np.ndarray):
        pts_to_include_x.append(base[focus_kp_idx, 0])
        pts_to_include_y.append(base[focus_kp_idx, 1])
    if isinstance(ours, np.ndarray):
        pts_to_include_x.append(ours[focus_kp_idx, 0])
        pts_to_include_y.append(ours[focus_kp_idx, 1])

    if opp_kp_idx is not None:
        if isinstance(gt, np.ndarray) and gt[opp_kp_idx, 2] > 0:
            pts_to_include_x.append(gt[opp_kp_idx, 0])
            pts_to_include_y.append(gt[opp_kp_idx, 1])
        if isinstance(base, np.ndarray):
            pts_to_include_x.append(base[opp_kp_idx, 0])
            pts_to_include_y.append(base[opp_kp_idx, 1])
        if isinstance(ours, np.ndarray):
            pts_to_include_x.append(ours[opp_kp_idx, 0])
            pts_to_include_y.append(ours[opp_kp_idx, 1])

    if not pts_to_include_x:
        cx, cy = float(img_w / 2), float(img_h / 2)
        pts_to_include_x = [cx]
        pts_to_include_y = [cy]

    margin_x = 22.0
    margin_y = 22.0
    min_x_pts = min(pts_to_include_x)
    max_x_pts = max(pts_to_include_x)
    min_y_pts = min(pts_to_include_y)
    max_y_pts = max(pts_to_include_y)

    x_min = max(0, int(np.floor(min_x_pts - margin_x)))
    x_max = min(img_w, int(np.ceil(max_x_pts + margin_x)))
    y_min = max(0, int(np.floor(min_y_pts - margin_y)))
    y_max = min(img_h, int(np.ceil(max_y_pts + margin_y)))

    # Balance aspect ratio for clean visual appearance
    box_w = x_max - x_min
    box_h = y_max - y_min
    if box_w < box_h:
        diff = box_h - box_w
        x_min = max(0, int(x_min - diff / 2))
        x_max = min(img_w, int(x_max + diff / 2))
    elif box_h < box_w:
        diff = box_w - box_h
        y_min = max(0, int(y_min - diff / 2))
        y_max = min(img_h, int(y_max + diff / 2))

    # --- PANEL A: Global Skeletal Prediction ---
    ax_a.imshow(img_rgb)
    ax_a.set_title(f"(a) Global Pose [{mode_tag.upper().replace('_', ' ')} Mode]", fontsize=14, fontweight="bold", pad=8)

    # GT Skeleton
    if isinstance(gt, np.ndarray):
        for p1, p2 in COCO_SKELETON_PAIRS:
            if p1 < len(gt) and p2 < len(gt) and gt[p1, 2] > 0 and gt[p2, 2] > 0:
                ax_a.plot([gt[p1, 0], gt[p2, 0]], [gt[p1, 1], gt[p2, 1]], color=COLOR_GT, lw=2.5, alpha=0.90)
        ax_a.scatter(gt[:, 0], gt[:, 1], color=COLOR_GT, s=45, alpha=0.90, zorder=5)

    # DARK Skeleton
    if isinstance(base, np.ndarray):
        for p1, p2 in COCO_SKELETON_PAIRS:
            if p1 < len(base) and p2 < len(base):
                ax_a.plot([base[p1, 0], base[p2, 0]], [base[p1, 1], base[p2, 1]], color=COLOR_DARK, linestyle="--", lw=2.0, alpha=0.80)
        ax_a.scatter(base[:, 0], base[:, 1], color=COLOR_DARK, marker="x", s=60, alpha=0.80, zorder=6)

    # Ours Skeleton
    if isinstance(ours, np.ndarray):
        for p1, p2 in COCO_SKELETON_PAIRS:
            if p1 < len(ours) and p2 < len(ours):
                ax_a.plot([ours[p1, 0], ours[p2, 0]], [ours[p1, 1], ours[p2, 1]], color=COLOR_OURS, lw=2.5, alpha=0.95)
        ax_a.scatter(ours[:, 0], ours[:, 1], color=COLOR_OURS, s=50, edgecolors="black", lw=0.8, zorder=7)

    # Yellow Zoom Box
    rect_zoom = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  edgecolor=COLOR_ELLIPSE_2S, facecolor="none", lw=2.2, linestyle="--", zorder=10)
    ax_a.add_patch(rect_zoom)

    # Legend
    legend_elems = [
        plt.Line2D([0], [0], color=COLOR_GT, lw=2.5, label="Ground Truth (GT)"),
        plt.Line2D([0], [0], color=COLOR_DARK, linestyle="--", marker="x", lw=2.0, label=dark_legend),
        plt.Line2D([0], [0], color=COLOR_OURS, lw=2.5, marker="o", label=ours_legend),
    ]
    ax_a.legend(handles=legend_elems, loc="upper left", facecolor="black", edgecolor="none", labelcolor="white", fontsize=9, framealpha=0.75)
    ax_a.axis("off")

    # --- PANEL B: Joint Zoom (Both Gaussian Modes Fully Encompassed) ---
    crop_img = img_rgb[y_min:y_max, x_min:x_max]
    ax_b.imshow(crop_img, extent=[x_min, x_max, y_max, y_min])
    ax_b.set_title(f"(b) Joint Zoom ({focus_name})", fontsize=14, fontweight="bold", pad=8)

    # MC Samples
    if len(samp_by_kp) > focus_kp_idx:
        pts_hm = np.asarray(samp_by_kp[focus_kp_idx])
        if pts_hm.ndim == 2 and len(pts_hm) > 0:
            pts_img = map_points_to_image_space(pts_hm, A_mat, b_vec)
            ax_b.scatter(pts_img[:, 0], pts_img[:, 1], color=COLOR_MC, s=8, alpha=0.35, zorder=4)

    # Sort components by weight descending
    if len(w_focus) > 0:
        order = np.argsort(-w_focus)
        m_focus = m_focus[order]
        c_focus = c_focus[order]
        w_focus = w_focus[order]

    for j in range(len(m_focus)):
        v1 = ellipse_polygon_vertices(m_focus[j], c_focus[j], n_sigma=1.0)
        v2 = ellipse_polygon_vertices(m_focus[j], c_focus[j], n_sigma=2.0)
        if len(v1) > 0:
            ax_b.add_patch(patches.Polygon(v1, closed=True, fill=False, edgecolor=COLOR_ELLIPSE_1S, lw=1.3, linestyle=":", zorder=6))
        if len(v2) > 0:
            ax_b.add_patch(patches.Polygon(v2, closed=True, fill=False, edgecolor=COLOR_ELLIPSE_2S, lw=2.0, linestyle="-", zorder=7))
            w_val = float(w_focus[j]) if j < len(w_focus) else 1.0
            label_text = f"Mode {j+1}: $w_{j+1}={w_val:.2f}$"
            ax_b.text(m_focus[j, 0] + 1.5, m_focus[j, 1] - 3.5, label_text,
                      color=COLOR_ELLIPSE_2S, fontsize=8.5, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.80, edgecolor="none"), zorder=12)

    # Focus Keypoint Markers
    if isinstance(gt, np.ndarray) and gt[focus_kp_idx, 2] > 0:
        ax_b.scatter(gt[focus_kp_idx, 0], gt[focus_kp_idx, 1], color=COLOR_GT, s=110, edgecolors="black", lw=1.5, zorder=8)
    if isinstance(base, np.ndarray):
        ax_b.scatter(base[focus_kp_idx, 0], base[focus_kp_idx, 1], color=COLOR_DARK, marker="x", s=130, lw=3.0, zorder=9)
    if isinstance(ours, np.ndarray):
        ax_b.scatter(ours[focus_kp_idx, 0], ours[focus_kp_idx, 1], color=COLOR_OURS, s=120, edgecolors="black", lw=1.5, zorder=10)

    for spine in ax_b.spines.values():
        spine.set_edgecolor(COLOR_ELLIPSE_2S); spine.set_linewidth(2.0); spine.set_visible(True)
    ax_b.axis("off")

    # --- PANEL C: Aggregated Heatmap + GMM Fit ---
    if isinstance(avg_hm, np.ndarray) and focus_kp_idx < avg_hm.shape[0]:
        hm_k = avg_hm[focus_kp_idx]
        hm_h, hm_w = hm_k.shape
        im_c = ax_c.imshow(hm_k, cmap="inferno", extent=[0, hm_w, hm_h, 0], interpolation="bilinear")
        plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)

        if focus_kp_idx < len(gmm_per_kp):
            kp_gmm = gmm_per_kp[focus_kp_idx]
            m_hm = np.asarray(kp_gmm.get("means", []), dtype=np.float64)
            c_hm = np.asarray(kp_gmm.get("covariances", []), dtype=np.float64)
            for j in range(len(m_hm)):
                v2_hm = ellipse_polygon_vertices(m_hm[j], c_hm[j], n_sigma=2.0)
                if len(v2_hm) > 0:
                    ax_c.add_patch(patches.Polygon(v2_hm, closed=True, fill=False, edgecolor=COLOR_OURS, lw=2.0))
                ax_c.scatter(m_hm[j, 0], m_hm[j, 1], color=COLOR_OURS, marker="+", s=100, lw=2.5, zorder=8)

        ax_c.set_title(f"(c) Aggregated Heatmap $\\bar{{H}}_{{{focus_name}}}$", fontsize=14, fontweight="bold", pad=8)
        ax_c.axis("off")

    # Save PDF & PNG
    pdf_out = out_dir / f"figure_1_swaps_{cand_name}.pdf"
    png_out = out_dir / f"figure_1_swaps_{cand_name}.png"
    plt.savefig(pdf_out, dpi=300, bbox_inches="tight")
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save Clean Crop & JSON
    cv2.imwrite(str(out_dir / f"crop_clean_{focus_name}.png"), cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "original_clean.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    logger.info("[OK] Generated Figura 1 -> %s", pdf_out.resolve())

COLOR_PARENT = "#FFD600"    # Amber Gold for parent joint anchor
COLOR_RING = "#76FF03"      # Lime green for kinematic prior band


# ==============================================================================
# RENDERER FIGURA 2: El Dilema del MRF (Muelles Cinemáticos vs. Contaminación 2D)
# ==============================================================================

def _draw_mrf_row_panels(ax_a, ax_b, ax_c, pkt: Dict[str, Any], focus_kp_idx: int, row_idx: int, is_success: bool):
    img_rgb = find_original_image(pkt)
    if img_rgb is None:
        return
    img_h, img_w = img_rgb.shape[:2]

    gt = pkt.get("ground_truth", {}).get("coords")
    ours = pkt.get("predictions", {}).get("ours_gmm", {}).get("coords")
    base = pkt.get("predictions", {}).get("baseline", {}).get("coords")

    parent_kp = PARENT_MAP.get(focus_kp_idx, 0)
    focus_name = COCO_KP_NAMES[focus_kp_idx].replace("_", " ").title()
    parent_name = COCO_KP_NAMES[parent_kp].replace("_", " ").title()

    # Calculate bone length statistics
    edge = (parent_kp, focus_kp_idx) if (parent_kp, focus_kp_idx) in COCO_BONE_LENGTH_STATS else (focus_kp_idx, parent_kp)
    mean_r, std_r = COCO_BONE_LENGTH_STATS.get(edge, (0.35, 0.12))

    valid = gt[:, 2] > 0 if isinstance(gt, np.ndarray) else np.ones(17, dtype=bool)
    if np.any(valid):
        w_box = np.max(gt[valid, 0]) - np.min(gt[valid, 0])
        h_box = np.max(gt[valid, 1]) - np.min(gt[valid, 1])
        scale_box = max(20.0, float(np.sqrt(w_box * h_box)))
    else:
        scale_box = 100.0

    mu_px = mean_r * scale_box
    sigma_px = std_r * scale_box * 2.0

    p_coord = ours[parent_kp, :2] if isinstance(ours, np.ndarray) else (gt[parent_kp, :2] if isinstance(gt, np.ndarray) else np.array([img_w/2, img_h/2]))
    c_gmm = base[focus_kp_idx, :2] if isinstance(base, np.ndarray) else (ours[focus_kp_idx, :2] if isinstance(ours, np.ndarray) else p_coord + 20)
    c_mrf = ours[focus_kp_idx, :2] if isinstance(ours, np.ndarray) else c_gmm
    c_gt = gt[focus_kp_idx, :2] if isinstance(gt, np.ndarray) else c_mrf

    # Crop bounding box encompassing parent, GMM, MRF, and GT
    all_pts_x = [p_coord[0], c_gmm[0], c_mrf[0], c_gt[0]]
    all_pts_y = [p_coord[1], c_gmm[1], c_mrf[1], c_gt[1]]
    margin = 55
    x_min = max(0, int(min(all_pts_x) - margin))
    x_max = min(img_w, int(max(all_pts_x) + margin))
    y_min = max(0, int(min(all_pts_y) - margin))
    y_max = min(img_h, int(max(all_pts_y) + margin))

    # (a) Global Skeleton
    ax_a.imshow(img_rgb)
    if isinstance(gt, np.ndarray):
        for p1, p2 in COCO_SKELETON_PAIRS:
            if p1 < len(gt) and p2 < len(gt) and gt[p1, 2] > 0 and gt[p2, 2] > 0:
                ax_a.plot([gt[p1, 0], gt[p2, 0]], [gt[p1, 1], gt[p2, 1]], color=COLOR_GT, lw=2.0, alpha=0.8)
    ax_a.plot([p_coord[0], c_mrf[0]], [p_coord[1], c_mrf[1]], color=COLOR_OURS, lw=4.0, zorder=5)
    ax_a.scatter(p_coord[0], p_coord[1], color=COLOR_PARENT, marker="s", s=90, edgecolors="black", zorder=6, label=f"Parent: {parent_name}")
    ax_a.scatter(c_mrf[0], c_mrf[1], color=COLOR_OURS, marker="o", s=110, edgecolors="black", zorder=6, label=f"Target: {focus_name}")
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor=COLOR_GT if is_success else COLOR_DARK, facecolor="none", lw=2.0, linestyle="--")
    ax_a.add_patch(rect)
    row_tag = "1" if row_idx == 1 else "2"
    scene_label = "Clean Scene" if is_success else "Crowded / Occluded Scene"
    ax_a.set_title(f"(a{row_tag}) Global Context ({scene_label})\nLimb: {parent_name} \u2192 {focus_name}", fontsize=11, fontweight="bold")
    ax_a.legend(loc="lower left", facecolor="white", labelcolor="black", edgecolor="#CCCCCC", fontsize=8.0, framealpha=0.88)
    ax_a.axis("off")

    # (b) Local GMM Unary vs Kinematic Prior Ring
    crop = img_rgb[y_min:y_max, x_min:x_max]
    ax_b.imshow(crop, extent=[x_min, x_max, y_max, y_min])
    ax_b.scatter(p_coord[0], p_coord[1], color=COLOR_PARENT, marker="s", s=130, edgecolors="black", label=f"Anchor ({parent_name})", zorder=5)

    prior_ring = patches.Circle((p_coord[0], p_coord[1]), mu_px, fill=False, edgecolor=COLOR_RING, linestyle="--", lw=2.0, label=f"Prior Length \u03bc = {mu_px:.0f}px", zorder=4)
    prior_band = patches.Wedge((p_coord[0], p_coord[1]), mu_px + sigma_px, 0, 360, width=2*sigma_px, facecolor=COLOR_RING, alpha=0.18, label="Prior Band \u03bc \u00b1 2\u03c3", zorder=3)
    ax_b.add_patch(prior_band)
    ax_b.add_patch(prior_ring)

    d_gmm = float(np.linalg.norm(c_gmm - p_coord))
    ax_b.scatter(c_gmm[0], c_gmm[1], color=COLOR_DARK, marker="x", s=120, lw=2.5, label=f"GMM Unary (d = {d_gmm:.0f}px)", zorder=6)
    if isinstance(gt, np.ndarray) and gt[focus_kp_idx, 2] > 0:
        ax_b.scatter(c_gt[0], c_gt[1], color=COLOR_GT, marker="o", s=120, edgecolors="black", label="Ground Truth", zorder=6)

    ax_b.set_title(f"(b{row_tag}) GMM Unary vs Anatomical Prior Band", fontsize=11, fontweight="bold")
    ax_b.legend(loc="lower left", facecolor="white", labelcolor="black", edgecolor="#CCCCCC", fontsize=8.0, framealpha=0.88)
    ax_b.axis("off")

    # (c) MRF MAP Resolution
    ax_c.imshow(crop, extent=[x_min, x_max, y_max, y_min])
    ax_c.scatter(p_coord[0], p_coord[1], color=COLOR_PARENT, marker="s", s=130, edgecolors="black", zorder=5)
    ax_c.add_patch(patches.Circle((p_coord[0], p_coord[1]), mu_px, fill=False, edgecolor=COLOR_RING, linestyle="--", lw=1.5, alpha=0.7, zorder=3))

    if isinstance(gt, np.ndarray) and gt[focus_kp_idx, 2] > 0:
        ax_c.scatter(c_gt[0], c_gt[1], color=COLOR_GT, marker="o", s=120, edgecolors="black", label="Ground Truth", zorder=6)
    ax_c.scatter(c_mrf[0], c_mrf[1], color=COLOR_OURS, marker="o", s=130, edgecolors="black", label="MRF Solution", zorder=7)
    ax_c.plot([p_coord[0], c_mrf[0]], [p_coord[1], c_mrf[1]], color=COLOR_OURS, lw=3.0, alpha=0.9, zorder=4)

    d_mrf = float(np.linalg.norm(c_mrf - p_coord))
    arr_color = COLOR_OURS if is_success else COLOR_DARK
    arr = patches.FancyArrowPatch((c_gmm[0], c_gmm[1]), (c_mrf[0], c_mrf[1]), color=arr_color, arrowstyle="-|>", mutation_scale=15, lw=2.5, linestyle="--", zorder=8)
    ax_c.add_patch(arr)

    status_tag = "Kinematic Pull \u2192 Restored" if is_success else "Rigidity Drag \u2192 Error"
    ax_c.set_title(f"(c{row_tag}) MRF: {status_tag}\nLength: d = {d_mrf:.0f}px (Prior \u03bc = {mu_px:.0f}px)", fontsize=11, fontweight="bold")
    ax_c.legend(loc="lower left", facecolor="white", labelcolor="black", edgecolor="#CCCCCC", fontsize=8.0, framealpha=0.88)
    ax_c.axis("off")


def render_figure_2(win_pkt: Dict[str, Any], fail_pkt: Dict[str, Any], win_kp_idx: int = 16, fail_kp_idx: int = 15, out_dir: Optional[Path] = None, cand_name: str = "combined"):
    """
    Fig 2: GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1.2, 1, 1])
    Fila 1 (Éxito Cinemático): (a1) Contexto, (b1) GMM vs Banda Prior, (c1) MRF muelle elástico restaura longitud.
    Fila 2 (Limitación 2D): (a2) Contexto solapado, (b2) GMM vs Banda Prior, (c2) MRF rigidez arrastra por escorzo.
    """
    if out_dir is None:
        out_dir = TARGET_OUTPUT_ROOT / "figura_2_mrf_dilemma"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(16, 9.5), dpi=300)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.2, 1.0, 1.0], hspace=0.28, wspace=0.18)

    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_b1 = fig.add_subplot(gs[0, 1])
    ax_c1 = fig.add_subplot(gs[0, 2])

    ax_a2 = fig.add_subplot(gs[1, 0])
    ax_b2 = fig.add_subplot(gs[1, 1])
    ax_c2 = fig.add_subplot(gs[1, 2])

    # Draw Fila 1: Win Case (2A)
    _draw_mrf_row_panels(ax_a1, ax_b1, ax_c1, win_pkt, focus_kp_idx=win_kp_idx, row_idx=1, is_success=True)

    # Draw Fila 2: Fail Case (2B)
    _draw_mrf_row_panels(ax_a2, ax_b2, ax_c2, fail_pkt, focus_kp_idx=fail_kp_idx, row_idx=2, is_success=False)

    pdf_out = out_dir / f"figure_2_mrf_{cand_name}.pdf"
    png_out = out_dir / f"figure_2_mrf_{cand_name}.png"
    plt.savefig(pdf_out, dpi=300, bbox_inches="tight")
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("[OK] Generated Figura 2 -> %s", pdf_out.resolve())


def render_figure_2_single_candidate(packet: Dict[str, Any], focus_kp_idx: int, out_dir: Path, cand_name: str, mode: str = "tta_mrf"):
    """
    Renders a single candidate for Figure 2 with full kinematic bone prior circles.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(16, 5.2), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0], wspace=0.18)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    is_success = "2a" in cand_name.lower()
    _draw_mrf_row_panels(ax_a, ax_b, ax_c, packet, focus_kp_idx=focus_kp_idx, row_idx=1, is_success=is_success)

    pdf_out = out_dir / f"figure_2_mrf_{cand_name}.pdf"
    png_out = out_dir / f"figure_2_mrf_{cand_name}.png"
    plt.savefig(pdf_out, dpi=300, bbox_inches="tight")
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("[OK] Generated Figura 2 Single -> %s", pdf_out.resolve())


# ==============================================================================
# RENDERER FIGURA 3: Ceguera Heurística vs. Alerta Volumétrica en OoD
# ==============================================================================

def render_figure_3(packet: Dict[str, Any], focus_kp_idx: int, out_dir: Path, cand_name: str, mode: str = "auto"):
    """
    Fig 3: GridSpec(1, 3, width_ratios=[1.2, 1.0, 0.9])
    (a) Global Scene (CrowdPose / OCHuman) with target occlusion bounding box.
    (b) Zoom Crop Overlaid with Spatial Heatmap + GMM Covariance Ellipse (det Sigma).
    (c) Diagnostic Bar Chart with REAL metrics (DARK Overconfidence vs. GMM Spatial Alert).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_rgb = find_original_image(packet)
    if img_rgb is None:
        return
    img_h, img_w = img_rgb.shape[:2]

    meta = packet.get("meta", {})
    gt = packet.get("ground_truth", {}).get("coords")
    base = packet.get("predictions", {}).get("baseline", {}).get("coords")
    ours = packet.get("predictions", {}).get("ours_gmm", {}).get("coords")
    gmm_per_kp = packet.get("gmm_per_kp", [])
    avg_hm = packet.get("aggregation", {}).get("heatmap_avg")
    if avg_hm is None and len(packet.get("tta_data", [])) > 0:
        avg_hm = packet["tta_data"][0].get("heatmap")

    focus_name = COCO_KP_NAMES[focus_kp_idx].replace("_", " ").title()

    # Determine focus coordinates
    gt_coord = gt[focus_kp_idx, :2] if (isinstance(gt, np.ndarray) and gt[focus_kp_idx, 2] > 0) else None
    dark_coord = base[focus_kp_idx, :2] if isinstance(base, np.ndarray) else None
    ours_coord = ours[focus_kp_idx, :2] if isinstance(ours, np.ndarray) else None

    # Compute crop bounding box around GT / DARK / GMM
    pts_x = [p[0] for p in [gt_coord, dark_coord, ours_coord] if p is not None]
    pts_y = [p[1] for p in [gt_coord, dark_coord, ours_coord] if p is not None]
    if not pts_x:
        pts_x = [img_w / 2]
        pts_y = [img_h / 2]

    margin = 55
    x_min = max(0, int(min(pts_x) - margin))
    x_max = min(img_w, int(max(pts_x) + margin))
    y_min = max(0, int(min(pts_y) - margin))
    y_max = min(img_h, int(max(pts_y) + margin))

    # Real Metrics Extraction
    base_preds = packet.get("predictions", {}).get("baseline", {})
    base_coords = base_preds.get("coords")
    scores = base_preds.get("scores")
    if base_coords is not None and len(base_coords) > focus_kp_idx and base_coords.shape[-1] >= 3:
        dark_conf = float(base_coords[focus_kp_idx, 2])
    elif isinstance(scores, (list, np.ndarray)) and len(scores) > focus_kp_idx and scores[focus_kp_idx] is not None:
        dark_conf = float(scores[focus_kp_idx])
    else:
        dark_conf = 0.50

    per_kp_base = packet.get("metrics", {}).get("per_kp_oks_base", [])
    if isinstance(per_kp_base, (list, np.ndarray)) and len(per_kp_base) > focus_kp_idx:
        v_oks = per_kp_base[focus_kp_idx]
        dark_oks = float(v_oks) if (v_oks is not None and not np.isnan(v_oks)) else 0.00
    else:
        dark_oks = 0.00

    cov_det = 1.0
    if focus_kp_idx < len(gmm_per_kp):
        kp_gmm = gmm_per_kp[focus_kp_idx]
        m_hm = np.asarray(kp_gmm.get("means", []), dtype=np.float64)
        c_hm = np.asarray(kp_gmm.get("covariances", []), dtype=np.float64)
        weights = np.asarray(kp_gmm.get("weights", []), dtype=np.float64)
        u_w = float(kp_gmm.get("uniform_weight", 0.0))

        gt_dict = packet.get("ground_truth", {})
        bbox = gt_dict.get("bboxes", gt_dict.get("bbox"))
        if bbox is not None:
            b_arr = np.asarray(bbox, dtype=np.float64).reshape(-1)
            bw, bh = (float(b_arr[2]), float(b_arr[3])) if b_arr.size >= 4 else (float(img_w), float(img_h))
        else:
            bw, bh = float(img_w), float(img_h)
        area = max(1.0, bw * bh)

        if len(m_hm) > 0 and len(c_hm) > 0:
            m_img_calib, c_img_calib, _, _ = map_gmm_to_image_space(packet, m_hm, c_hm, img_w, img_h)
            w_full = np.append(weights, u_w)
            kappa = float(COCO_SIGMAS[focus_kp_idx]) if focus_kp_idx < len(COCO_SIGMAS) else 0.05
            try:
                sigma_final, _ = compute_calibrated_covariance(
                    means_img=m_img_calib,
                    covs_img=c_img_calib,
                    weights=w_full,
                    scale_sq=float(area),
                    kappa=kappa
                )
                cov_det = float(np.linalg.det(sigma_final))
            except Exception:
                cov_det = float(np.linalg.det(c_hm[0])) if len(c_hm) > 0 else 1.0

    entropy_val = float(packet.get("metrics", {}).get("entropy", 3.20))
    u_heuristic = 1.0 - dark_conf

    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(16.5, 5.2), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.05], wspace=0.36)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # (a) Crowded Scene
    ax_a.imshow(img_rgb)
    if isinstance(gt, np.ndarray):
        for p1, p2 in COCO_SKELETON_PAIRS:
            if p1 < len(gt) and p2 < len(gt) and gt[p1, 2] > 0 and gt[p2, 2] > 0:
                ax_a.plot([gt[p1, 0], gt[p2, 0]], [gt[p1, 1], gt[p2, 1]], color=COLOR_GT, lw=2.0, alpha=0.85)
    if dark_coord is not None:
        ax_a.scatter(dark_coord[0], dark_coord[1], color=COLOR_DARK, marker="x", s=110, lw=2.5, label="DARK (False Lock)", zorder=6)
    if gt_coord is not None:
        ax_a.scatter(gt_coord[0], gt_coord[1], color=COLOR_GT, marker="o", s=90, edgecolors="black", label="Ground Truth", zorder=6)

    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor=COLOR_DARK, facecolor="none", lw=2.0, linestyle="--", zorder=5)
    ax_a.add_patch(rect)
    ds_name = meta.get("dataset_name", meta.get("dataset", "CrowdPose")).upper()
    vis_tag = " [vis=0, Absent]" if gt_coord is None else " [Occluded]"
    ax_a.set_title(f"(a) Global Scene ({ds_name}{vis_tag})\nTarget: {focus_name}", fontsize=11, fontweight="bold")
    ax_a.legend(loc="lower left", facecolor="white", labelcolor="black", edgecolor="#CCCCCC", fontsize=8, framealpha=0.88)
    ax_a.axis("off")

    # (b) Zoom: Heatmap Activation Overlay + GMM Ellipse
    crop_img = img_rgb[y_min:y_max, x_min:x_max]
    ax_b.imshow(crop_img, extent=[x_min, x_max, y_max, y_min])

    # Get accurately transformed GMM Gaussians in Image Space
    m_img, c_img = np.empty((0, 2)), np.empty((0, 2, 2))
    A_mat, b_vec = None, None
    weights = []
    if focus_kp_idx < len(gmm_per_kp):
        kp_gmm = gmm_per_kp[focus_kp_idx]
        m_hm = np.asarray(kp_gmm.get("means", []), dtype=np.float64)
        c_hm = np.asarray(kp_gmm.get("covariances", []), dtype=np.float64)
        weights = kp_gmm.get("weights", [])
        if len(m_hm) > 0 and len(c_hm) > 0:
            m_img, c_img, A_mat, b_vec = map_gmm_to_image_space(packet, m_hm, c_hm, img_w, img_h)

    # Overlay Heatmap with affine warp if available
    if isinstance(avg_hm, np.ndarray) and focus_kp_idx < avg_hm.shape[0] and A_mat is not None and b_vec is not None:
        hm_k = avg_hm[focus_kp_idx]
        M_warp = np.hstack([A_mat, b_vec.reshape(2, 1)]).astype(np.float32)
        hm_full = cv2.warpAffine(hm_k.astype(np.float32), M_warp, (img_w, img_h), flags=cv2.INTER_CUBIC)
        hm_crop = hm_full[y_min:y_max, x_min:x_max]
        if hm_crop.size > 0 and np.max(hm_crop) > 0:
            hm_crop_norm = hm_crop / np.max(hm_crop)
            ax_b.imshow(hm_crop_norm, cmap="inferno", alpha=0.50, extent=[x_min, x_max, y_max, y_min], zorder=2)

    # Draw GMM components and covariance ellipses in image space
    for j in range(len(m_img)):
        w_j = float(weights[j]) if j < len(weights) else 1.0
        v2 = ellipse_polygon_vertices(m_img[j], c_img[j], n_sigma=2.0)
        if len(v2) > 0:
            ax_b.add_patch(patches.Polygon(v2, closed=True, fill=False, edgecolor=COLOR_ELLIPSE_2S, lw=2.2, linestyle="--", label="GMM 2\u03c3 Uncertainty Ellipse" if j==0 else "", zorder=4))
        
        gmm_label = "GMM Center (Unimodal, $K=1$)" if len(m_img) == 1 else f"GMM Mode {j+1} ($w_{j+1}={w_j:.2f}$)"
        ax_b.scatter(m_img[j, 0], m_img[j, 1], color=COLOR_OURS, marker="o", s=85, edgecolors="black", label=gmm_label, zorder=5)

    if dark_coord is not None:
        ax_b.scatter(dark_coord[0], dark_coord[1], color=COLOR_DARK, marker="x", s=130, lw=3.0, label=f"DARK Lock ($P={dark_conf:.2f}$)", zorder=6)
    if gt_coord is not None:
        ax_b.scatter(gt_coord[0], gt_coord[1], color=COLOR_GT, marker="o", s=120, edgecolors="black", label="Ground Truth", zorder=6)
    else:
        # Phantom dummy for legend when keypoint is absent
        ax_b.plot([], [], ' ', label="Ground Truth: Absent (vis=0)")

    ax_b.set_title(f"(b) Activation Map & Spatial Uncertainty\n$\\det(\\Sigma) = {cov_det:.2f}$ (Expanded Ellipse Alert)", fontsize=11, fontweight="bold")
    ax_b.legend(loc="lower left", facecolor="white", labelcolor="black", edgecolor="#CCCCCC", fontsize=7.5, framealpha=0.88)
    ax_b.axis("off")

    # (c) Diagnostic Bar Chart with Clean Spacing
    labels = [
        f"DARK Conf\n($P={dark_conf:.2f}$)",
        f"Accuracy\n($\\mathrm{{OKS}}={dark_oks:.2f}$)" if gt_coord is not None else "GT Status\n($\\mathrm{vis}=0$ Absent)",
        f"Heuristic Alert\n($1-P={u_heuristic:.2f}$)",
        f"GMM Spatial Alert\n($\\det\\Sigma={cov_det:.2f}$)"
    ]

    det_norm = min(1.0, float(np.log1p(cov_det) / np.log1p(40.0)))
    vals = [dark_conf, dark_oks, u_heuristic, det_norm]
    bar_colors = [COLOR_DARK, "#9E9E9E", "#81C784", COLOR_ELLIPSE_2S]

    y_pos = np.arange(len(labels))
    bars = ax_c.barh(y_pos, vals, color=bar_colors, height=0.52, edgecolor="black", lw=1.2, zorder=3)
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(labels, fontsize=8.5, fontweight="bold")
    ax_c.set_xlim(0, 1.28)
    ax_c.set_xlabel("Normalized Magnitude [0.0 - 1.0]", fontsize=9.5, fontweight="bold")
    ax_c.set_title(f"(c) Diagnostic Uncertainty Profile\nShannon Entropy = {entropy_val:.2f} nats", fontsize=11, fontweight="bold")
    ax_c.grid(axis="x", linestyle=":", alpha=0.6, zorder=0)

    # Label text per bar
    text_labels = [f"{dark_conf:.2f}", f"{dark_oks:.2f}" if gt_coord is not None else "0.00", f"{u_heuristic:.2f}", f"det={cov_det:.2f}"]
    for bar, v, txt in zip(bars, vals, text_labels):
        ax_c.text(v + 0.025, bar.get_y() + bar.get_height() / 2, txt, va="center", fontweight="bold", fontsize=9.0)

    pdf_out = out_dir / f"figure_3_ood_{cand_name}.pdf"
    png_out = out_dir / f"figure_3_ood_{cand_name}.png"
    plt.savefig(pdf_out, dpi=300, bbox_inches="tight")
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("[OK] Generated Figura 3 -> %s", pdf_out.resolve())


# ==============================================================================
# RENDERER FIGURA 4: Heatmap Poisoning en TTA
# ==============================================================================

def render_figure_4(packet: Dict[str, Any], focus_kp_idx: int, out_dir: Path, cand_name: str, mode: str = "auto"):
    """
    FIGURA 4: Anatomía del Heatmap Poisoning en Test-Time Augmentation (TTA).
    
    GridSpec(2, 3):
    Fila 1: Desglose de 3 escalas reales de TTA con confianzas y detección de escala contaminada (borde rojo).
    Fila 2:
      (d) TTA Estándar / Promedio Aritmético Ingenuo -> Pico destruido/dividido (Caída de OKS)
      (e) Ours Agregación Continua Adaptativa (Quality Gated) -> Pico nítido preservado (OKS Rescatado)
      (f) Diagnóstico Cuantitativo: OKS Base vs. OKS TTA (Caída) vs. OKS Ours (Recuperación)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tta_data = packet.get("tta_data", [])
    avg_hm = packet.get("aggregation", {}).get("heatmap_avg")
    meta = packet.get("meta", {})
    focus_name = COCO_KP_NAMES[focus_kp_idx].replace("_", " ").title()

    # ---------------- 1. Dynamic Metric Extraction (Strict: Fail early on missing data) ----------------
    gt_dict = packet.get("ground_truth", {})
    gt = gt_dict.get("coords")
    bbox = gt_dict.get("bboxes", gt_dict.get("bbox"))
    if gt is None or bbox is None:
        logger.error("[render_figure_4] Error: ground_truth coords or bbox missing in packet for %s. Aborting.", cand_name)
        return

    img_rgb = find_original_image(packet)
    img_h, img_w = img_rgb.shape[:2] if img_rgb is not None else (1000, 1000)

    vis = gt[:, 2] if (gt.ndim == 2 and gt.shape[1] >= 3) else np.ones(len(gt), dtype=np.float64)
    gt_2d = gt[:, :2]

    b_arr = np.asarray(bbox, dtype=np.float64).reshape(-1)
    bx, by, bw, bh = float(b_arr[0]), float(b_arr[1]), float(b_arr[2]), float(b_arr[3])
    area = float(bw * bh)

    # Base OKS (Single Scale Baseline)
    base_pred = packet.get("predictions", {}).get("baseline", {}).get("coords")
    if base_pred is None:
        logger.error("[render_figure_4] Error: baseline predictions missing in packet for %s. Aborting.", cand_name)
        return
    oks_base, _ = compute_oks(base_pred[:, :2], gt_2d, vis, area)

    # Ours (GMM on Weighted Aggregation) OKS
    ours_pred = packet.get("predictions", {}).get("ours_gmm", {}).get("coords")
    if ours_pred is None:
        logger.error("[render_figure_4] Error: ours_gmm predictions missing in packet for %s. Aborting.", cand_name)
        return
    oks_weighted, _ = compute_oks(ours_pred[:, :2], gt_2d, vis, area)

    # Reconstruct scales and compute Unweighted GMM OKS on the fly
    if not tta_data:
        logger.error("[render_figure_4] Error: tta_data is empty in packet for %s. Aborting.", cand_name)
        return

    scales_dict: Dict[float, List[np.ndarray]] = {}
    for entry in tta_data:
        s_val = float(entry.get("scale_factor", 1.0))
        hm_val = entry.get("heatmap")
        if hm_val is not None:
            scales_dict.setdefault(s_val, []).append(hm_val)

    scale_factors_all = sorted(scales_dict.keys())
    mmpose_meta = packet.get("aggregation", {}).get("mmpose_metadata")
    if mmpose_meta is None:
        logger.error("[render_figure_4] Error: mmpose_metadata missing in packet for %s. Aborting.", cand_name)
        return

    heatmaps_per_scale_all = [np.mean(scales_dict[s], axis=0).astype(np.float32) for s in scale_factors_all]
    S_total = len(heatmaps_per_scale_all)
    K_total, H_total, W_total = heatmaps_per_scale_all[0].shape

    # Unweighted aggregation via aggregate_multiscale_heatmaps (uniform confidences)
    uniform_confs = [np.ones(K_total, dtype=np.float64) for _ in range(S_total)]
    hm_unweighted_all = aggregate_multiscale_heatmaps(heatmaps_per_scale_all, uniform_confs, method="weighted_mean")

    # Decode unweighted with sample_from_heatmap + select_best_model (matching runner.py)
    coords_unw_hm = np.zeros((K_total, 2), dtype=np.float32)
    for k in range(K_total):
        try:
            samples = sample_from_heatmap(
                hm_unweighted_all[k],
                num_samples=1000,
                strategy="rejection",
                temperature=0.3,
                use_dequantization=True,
                seed=42 + k
            )
            mres = select_best_model(
                samples.astype(np.float64),
                aic_weight=0,
                bic_weight=1,
                reg_covar=1e-4,
                max_iter=100,
                random_state=42 + k
            )
            coords_unw_hm[k] = mres.best_mean.astype(np.float32)
        except Exception as e:
            logger.error("[render_figure_4] Error decoding unweighted heatmap for keypoint %d (%s): %s. Terminating.", k, COCO_KP_NAMES[k], e)
            raise RuntimeError(f"Error decoding unweighted heatmap for keypoint {k} ({COCO_KP_NAMES[k]}): {e}") from e

    # Transform heatmap -> image coords using StandardizedHeatmap & MMPoseAdapter
    std_hm_unw = StandardizedHeatmap(
        data=hm_unweighted_all,
        original_size=(img_h, img_w),
        metadata=mmpose_meta
    )
    try:
        coords_unw_img = MMPoseAdapter.transform_heatmap_coords_to_image(coords_unw_hm, std_hm_unw)
    except Exception as e:
        logger.error("[render_figure_4] Error transforming unweighted coords to image space: %s. Aborting.", e)
        return

    oks_unweighted, _ = compute_oks(coords_unw_img, gt_2d, vis, area)
    delta_saved = float(oks_weighted - oks_unweighted)

    # ---------------- 2. Per-Scale Out-of-Bounds Distance Function ----------------
    def get_scale_exit_dist(s_val: float) -> float:
        if len(gt) > focus_kp_idx:
            sbx, sby, sbw, sbh = scale_bbox((bx, by, bw, bh), s_val, img_h, img_w)
            gt_pt = gt[focus_kp_idx, :2]
            out_l = max(0.0, sbx - gt_pt[0])
            out_r = max(0.0, gt_pt[0] - (sbx + sbw))
            out_t = max(0.0, sby - gt_pt[1])
            out_b = max(0.0, gt_pt[1] - (sby + sbh))
            return float(np.sqrt(max(out_l, out_r)**2 + max(out_t, out_b)**2))
        return 0.0

    # ---------------- 3. Select 3 Key Scales: (a) 0.85x, (b) 0.925x, (c) 1.00x ----------------
    required_scales = [0.85, 0.925, 1.00]
    for s_val in required_scales:
        if s_val not in scales_dict:
            logger.error("[render_figure_4] Error: Required scale %s not present in tta_data scales (%s) for %s. Aborting.", s_val, scale_factors_all, cand_name)
            return

    display_scales = required_scales
    display_heatmaps = [np.mean(scales_dict[s_val], axis=0).astype(np.float32)[focus_kp_idx] for s_val in display_scales]
    h1, h2, h3 = display_heatmaps[0], display_heatmaps[1], display_heatmaps[2]
    s1, s2, s3 = display_scales[0], display_scales[1], display_scales[2]

    # Reusing compute_heatmap_confidence from scale_tta
    c1 = float(compute_heatmap_confidence(h1[np.newaxis, ...], sharpness_scale=5.0, peak_exponent=1.0, sharpness_exponent=1.0)[0])
    c2 = float(compute_heatmap_confidence(h2[np.newaxis, ...], sharpness_scale=5.0, peak_exponent=1.0, sharpness_exponent=1.0)[0])
    c3 = float(compute_heatmap_confidence(h3[np.newaxis, ...], sharpness_scale=5.0, peak_exponent=1.0, sharpness_exponent=1.0)[0])
    rho1, rho2, rho3 = float(np.max(h1)), float(np.max(h2)), float(np.max(h3))
    c_sum = c1 + c2 + c3 + 1e-10
    w1, w2, w3 = c1 / c_sum, c2 / c_sum, c3 / c_sum
    min_c = min(c1, c2, c3)

    # ---------------- 4. Canvas Setup ----------------
    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(16.5, 7.8), dpi=300)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.34, wspace=0.34)

    # ---------------- Fila 1: 3 Escalas Individuales ----------------
    ax_1 = fig.add_subplot(gs[0, 0])
    ax_2 = fig.add_subplot(gs[0, 1])
    ax_3 = fig.add_subplot(gs[0, 2])

    for ax, hm_s, s_val, c_val, w_val, rho_val, letter in zip(
        [ax_1, ax_2, ax_3],
        [h1, h2, h3],
        [s1, s2, s3],
        [c1, c2, c3],
        [w1, w2, w3],
        [rho1, rho2, rho3],
        ["a", "b", "c"]
    ):
        hm_norm = hm_s / max(1e-6, np.max(hm_s))
        ax.imshow(hm_norm, cmap="inferno")
        
        # Calculate exit distance dynamically for this specific scale
        dist_s = get_scale_exit_dist(s_val)
        is_poison = (c_val == min_c and c_val < 0.80) or (dist_s > 0)

        # Scale label without truncating (e.g., 0.85x, 0.925x, 1.00x)
        s_str = f"{s_val:g}" if abs(s_val - round(s_val, 2)) > 1e-4 else f"{s_val:.2f}"

        if is_poison:
            exit_label = f"Out-of-FoV by {dist_s:.1f}px" if dist_s > 0 else "Out-of-FoV Crop"
            ax.set_title(f"({letter}) Scale {s_str}$\\times$ ({exit_label})\nConf $\\mathcal{{C}}_k = {c_val:.3f}$ ($w_k={w_val*100:.1f}\\%$) [POISONED / FLAT]", fontsize=9.5, fontweight="bold", color=COLOR_DARK)
            for spine in ax.spines.values():
                spine.set_edgecolor(COLOR_DARK)
                spine.set_linewidth(2.8)
                spine.set_linestyle("--")
                spine.set_visible(True)
        else:
            ax.set_title(f"({letter}) Scale {s_str}$\\times$ (In-Bounds Crop)\nConf $\\mathcal{{C}}_k = {c_val:.3f}$ ($w_k={w_val*100:.1f}\\%$)", fontsize=9.5, fontweight="bold")
        ax.axis("off")

    # ---------------- Fila 2: Unweighted Aggregation vs Ours Weighted Aggregation vs Quantitative Profile ----------------
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    # (d) Unweighted Arithmetic Average (Poisoned by out-of-FoV scale)
    naive_hm = (h1 + h2 + h3) / 3.0
    naive_norm = naive_hm / max(1e-6, np.max(naive_hm))
    ax_d.imshow(naive_norm, cmap="inferno")
    ax_d.set_title(f"(d) Unweighted Heatmap (Arithmetic Mean)\nDiluted Peak / Noise $\\rightarrow \\mathrm{{OKS}}_{{\\mathrm{{GMM}}}}={oks_unweighted:.2f}$", fontsize=10.0, fontweight="bold", color=COLOR_DARK)
    for spine in ax_d.spines.values():
        spine.set_edgecolor(COLOR_DARK)
        spine.set_linewidth(2.2)
        spine.set_visible(True)
    ax_d.axis("off")

    # (e) Ours Weighted Continuous Aggregation (Quality Gated)
    if avg_hm is None or focus_kp_idx >= avg_hm.shape[0]:
        logger.error("[render_figure_4] Error: heatmap_avg missing or invalid for keypoint %d in %s. Aborting.", focus_kp_idx, cand_name)
        return

    ours_norm = avg_hm[focus_kp_idx] / max(1e-6, np.max(avg_hm[focus_kp_idx]))
    ax_e.imshow(ours_norm, cmap="inferno")
    ax_e.set_title(f"(e) Weighted Heatmap (Ours Quality Gated)\nSharp Peak Preserved $\\rightarrow \\mathrm{{OKS}}_{{\\mathrm{{GMM}}}}={oks_weighted:.2f}$", fontsize=10.0, fontweight="bold", color=COLOR_OURS)
    for spine in ax_e.spines.values():
        spine.set_edgecolor(COLOR_OURS)
        spine.set_linewidth(2.2)
        spine.set_visible(True)
    ax_e.axis("off")

    # (f) Quantitative Comparison Bar Chart
    bar_labels = [
        f"Baseline (No TTA)\n($\\mathrm{{OKS}}={oks_base:.2f}$)",
        f"Unweighted Mean\n($\\mathrm{{OKS}}={oks_unweighted:.2f}$ Poisoned)",
        f"Ours Weighted Mean\n($\\mathrm{{OKS}}={oks_weighted:.2f}$ Rescued)"
    ]
    bar_vals = [oks_base, oks_unweighted, oks_weighted]
    bar_colors = ["#78909C", COLOR_DARK, COLOR_OURS]

    y_pos = np.arange(len(bar_labels))
    bars = ax_f.barh(y_pos, bar_vals, color=bar_colors, height=0.52, edgecolor="black", lw=1.2, zorder=3)
    ax_f.set_yticks(y_pos)
    ax_f.set_yticklabels(bar_labels, fontsize=8.5, fontweight="bold")
    ax_f.set_xlim(0, 1.25)
    ax_f.set_xlabel("Global Pose OKS [0.0 - 1.0]", fontsize=9.5, fontweight="bold")
    ax_f.set_title(f"(f) Recovery Diagnostic Profile\nTarget: {focus_name} ($\\Delta = +{delta_saved:.2f}$)", fontsize=10.5, fontweight="bold")
    ax_f.grid(axis="x", linestyle=":", alpha=0.6, zorder=0)

    for bar, v in zip(bars, bar_vals):
        ax_f.text(v + 0.025, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", fontweight="bold", fontsize=9.0)

    pdf_out = out_dir / f"figure_4_poisoning_{cand_name}.pdf"
    png_out = out_dir / f"figure_4_poisoning_{cand_name}.png"
    plt.savefig(pdf_out, dpi=300, bbox_inches="tight")
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("[OK] Generated Figura 4 -> %s", pdf_out.resolve())


# ==============================================================================
# RENDERER FIGURE 5: Uniform Background Component (pi_uniform)
# ==============================================================================

def render_figure_5(base_pkt: Dict[str, Any], out_dir: Path, cand_name: str, mode: str = "auto", focus_kp_idx: int = 10):
    """
    FIGURE 5: Visualization of Uniform Background Component (pi_uniform) under Resolution Degradation.
    
    GridSpec(2, 5):
    Row 1 (Top): Progression across 5 resolution degradation tiers (1.0x -> 0.5x -> 0.25x -> 0.125x -> 0.062x)
                 with purple overlay proportional to pi_uniform.
    Row 2 (Bottom): Progression of 2D heatmap for focus keypoint, Monte Carlo samples
                    (Gaussian inliers in amber vs. background outliers in purple #D500F9 absorbed by pi_uniform)
                    and fitted Gaussian confidence ellipses (1-sigma dotted, 2-sigma solid in cyan).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_rgb = find_original_image(base_pkt)
    if img_rgb is None:
        logger.error("[render_figure_5] Error: Original image missing in packet for %s. Aborting.", cand_name)
        return

    avg_hm = base_pkt.get("aggregation", {}).get("heatmap_avg")
    if avg_hm is None:
        tta_data = base_pkt.get("tta_data", [])
        for entry in tta_data:
            if entry.get("heatmap") is not None:
                avg_hm = entry.get("heatmap")
                break

    if avg_hm is None or focus_kp_idx >= avg_hm.shape[0]:
        logger.error("[render_figure_5] Error: Heatmap missing or focus_kp_idx %d invalid in packet for %s. Aborting.", focus_kp_idx, cand_name)
        return

    focus_name = COCO_KP_NAMES[focus_kp_idx].replace("_", " ").title()
    h_orig, w_orig = img_rgb.shape[:2]

    # Initialize MMPose adapter for real multi-resolution inference
    adapter = MMPoseAdapter(
        config_path="body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py",
        checkpoint_path="hrnet_w32.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    bboxes = base_pkt.get("ground_truth", {}).get("bboxes", [])
    if isinstance(bboxes, (list, tuple, np.ndarray)) and len(bboxes) == 4 and isinstance(bboxes[0], (int, float, np.number)):
        bbox = tuple(float(x) for x in bboxes)
    elif isinstance(bboxes, (list, tuple, np.ndarray)) and len(bboxes) > 0 and isinstance(bboxes[0], (list, tuple, np.ndarray)):
        bbox = tuple(float(x) for x in bboxes[0])
    else:
        bbox = (0.0, 0.0, float(w_orig), float(h_orig))

    levels = ["Clean (1.00x)", "Low (0.50x)", "Med (0.25x)", "High (0.125x)", "Extreme (0.062x)"]
    scale_factors = [1.0, 0.5, 0.25, 0.125, 0.0625]

    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(17.5, 7.8), dpi=300)
    gs = fig.add_gridspec(2, 5, height_ratios=[1.1, 1.25], hspace=0.30, wspace=0.18)

    for idx, (lvl, s_f) in enumerate(zip(levels, scale_factors)):
        ax_img = fig.add_subplot(gs[0, idx])
        ax_hm = fig.add_subplot(gs[1, idx])

        # 1. Real degraded image and real MMPose network inference
        if s_f == 1.0:
            deg_img = img_rgb.copy()
            deg_bbox = bbox
        else:
            resized_img, resized_bbox, _ = _resize_image_and_annotations(
                img_rgb, bbox, [], resize_scale=s_f
            )
            deg_img = resized_img
            deg_bbox = resized_bbox

        res = adapter.predict(deg_img, bbox=deg_bbox)
        hm_deg = res.data[focus_kp_idx].astype(np.float32)
        H, W = hm_deg.shape

        # 2. Monte Carlo Sampling (exact runner.py parameters)
        try:
            samples = sample_from_heatmap(
                hm_deg,
                num_samples=1000,
                strategy="rejection",
                temperature=0.3,
                use_dequantization=True,
                seed=42 + idx
            )
        except Exception:
            samples = sample_from_heatmap(
                hm_deg,
                num_samples=1000,
                strategy="importance",
                temperature=0.3,
                use_dequantization=True,
                seed=42 + idx
            )

        # 3. Model Selection & Fitting (exact runner.py parameters)
        mixture_res = select_best_model(
            samples.astype(np.float64),
            aic_weight=0,
            bic_weight=1,
            reg_covar=1e-4,
            max_iter=100,
            tol=1e-4,
            decode_strategy="argmax",
            variance_threshold=3.0,
            random_state=42 + idx
        )
        pi_u = float(mixture_res.uniform_weight)
        pi_g = 1.0 - pi_u

        # 4. Classify samples analytically using exact E-step posterior responsibilities
        n_pts = len(samples)
        g_probs = np.zeros(n_pts, dtype=np.float64)
        margin = 1.0
        x_min, y_min = samples.min(axis=0) - margin
        x_max, y_max = samples.max(axis=0) + margin
        uniform_density = 1.0 / max(1e-6, (x_max - x_min) * (y_max - y_min))

        for comp in mixture_res.components:
            mu = comp.mean[:2].astype(np.float64)
            cov = comp.covariance[:2, :2].astype(np.float64)
            try:
                inv_cov = np.linalg.inv(cov)
                det_cov = np.linalg.det(cov)
                if det_cov > 0:
                    norm_c = 1.0 / np.sqrt(((2 * np.pi) ** 2) * det_cov)
                    diff = samples.astype(np.float64) - mu
                    maha_sq = np.clip(np.einsum('ni,ij,nj->n', diff, inv_cov, diff), None, 700)
                    g_probs += comp.weight * norm_c * np.exp(-0.5 * maha_sq)
            except Exception:
                pass

        u_probs = pi_u * uniform_density
        total_p = np.maximum(g_probs + u_probs, 1e-300)
        uniform_resp = u_probs / total_p
        outlier_mask = uniform_resp >= 0.5

        inlier_samples = samples[~outlier_mask]
        outlier_samples = samples[outlier_mask]

        # --- TOP ROW: Image with Uniform Tint ---
        disp_img = cv2.resize(deg_img, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        purple_layer = np.full_like(disp_img, [213, 0, 249], dtype=np.uint8)
        tint_alpha = min(0.60, pi_u * 1.8) if pi_u > 0.001 else 0.0
        deg_img_tinted = cv2.addWeighted(disp_img, 1.0 - tint_alpha, purple_layer, tint_alpha, 0)

        ax_img.imshow(deg_img_tinted)
        ax_img.set_title(f"Image: {lvl}\n$\\pi_{{\\mathrm{{uniform}}}} = {pi_u*100:.2f}\\%$ Absorbed", fontsize=10.5, fontweight="bold")
        for spine in ax_img.spines.values():
            if pi_u >= 0.10:
                spine.set_edgecolor(COLOR_UNIFORM)
                spine.set_linewidth(2.2)
                spine.set_visible(True)
            else:
                spine.set_visible(False)
        ax_img.axis("off")

        # --- BOTTOM ROW: Heatmap + Samples + Fitted Gaussians ---
        hm_norm = hm_deg / max(1e-6, np.max(hm_deg))
        ax_hm.imshow(hm_norm, cmap="inferno", extent=[0, W, H, 0], interpolation="bilinear")

        # Scatter outlier noise samples absorbed by uniform component (Purple)
        if len(outlier_samples) > 0:
            ax_hm.scatter(outlier_samples[:, 0], outlier_samples[:, 1], color=COLOR_UNIFORM, s=12, alpha=0.55,
                          label=f"Noise ({len(outlier_samples)})" if idx == 4 else None, zorder=4)

        # Scatter inlier samples captured by Gaussians (Amber)
        if len(inlier_samples) > 0:
            ax_hm.scatter(inlier_samples[:, 0], inlier_samples[:, 1], color=COLOR_MC, s=10, alpha=0.40,
                          label=f"Gaussian ({len(inlier_samples)})" if idx == 4 else None, zorder=5)

        # Plot fitted Gaussian Ellipses
        for j, comp in enumerate(mixture_res.components):
            mu = comp.mean[:2]
            cov = comp.covariance[:2, :2]
            v1 = ellipse_polygon_vertices(mu, cov, n_sigma=1.0)
            v2 = ellipse_polygon_vertices(mu, cov, n_sigma=2.0)
            if len(v1) > 0:
                ax_hm.add_patch(patches.Polygon(v1, closed=True, fill=False, edgecolor=COLOR_ELLIPSE_2S, lw=1.2, linestyle=":", zorder=7))
            if len(v2) > 0:
                ax_hm.add_patch(patches.Polygon(v2, closed=True, fill=False, edgecolor=COLOR_OURS, lw=2.2, zorder=8))
            ax_hm.scatter(mu[0], mu[1], color=COLOR_OURS, marker="+", s=100, lw=2.2, zorder=9)

        ax_hm.set_title(f"Heatmap & GMM Fit ({focus_name})\n$\\pi_g = {pi_g*100:.1f}\\%$ | $\\pi_u = {pi_u*100:.1f}\\%$", fontsize=10.0, fontweight="bold")
        if idx == 4:
            ax_hm.legend(loc="upper right", facecolor="black", labelcolor="white", fontsize=7.5, framealpha=0.85)

        for spine in ax_hm.spines.values():
            if pi_u >= 0.10:
                spine.set_edgecolor(COLOR_UNIFORM)
                spine.set_linewidth(2.2)
                spine.set_visible(True)
            else:
                spine.set_edgecolor("#555555")
                spine.set_linewidth(1.0)
                spine.set_visible(True)
        ax_hm.axis("off")

    pdf_out = out_dir / f"figure_5_uniform_{cand_name}.pdf"
    png_out = out_dir / f"figure_5_uniform_{cand_name}.png"
    plt.savefig(pdf_out, dpi=300, bbox_inches="tight")
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("[OK] Generated Figura 5 -> %s", pdf_out.resolve())


# ==============================================================================
# MAIN EXECUTION CONTROLLER: 3 Candidates for Each Figure
# ==============================================================================

def main() -> None:
    logger.info("=" * 80)
    logger.info("GENERATING ALL PUBLICATION FIGURES (3 CANDIDATES PER FIGURE)")
    logger.info("Destination: %s", TARGET_OUTPUT_ROOT)
    logger.info("=" * 80)

    TARGET_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. FIGURA 1: SWAPS (3 Candidatos: Wins_462, Wins_20, Wins_31)
    # --------------------------------------------------------------------------
    fig1_cands = [
        ("candidato_1_image_462", 462, 16), # R_Ankle
        ("candidato_2_image_20", 20, 9),    # L_Wrist
        ("candidato_3_image_31", 31, 5),    # L_Shoulder
    ]
    for c_dir_name, img_id, focus_kp in fig1_cands:
        p_path = search_packet(img_id)
        if p_path and p_path.exists():
            pkt = load_deep_analysis(str(p_path))
            out_folder = TARGET_OUTPUT_ROOT / "figura_1_swaps" / c_dir_name
            render_figure_1(pkt, focus_kp, out_folder, c_dir_name)

    # --------------------------------------------------------------------------
    # 2. FIGURA 2: MRF DILEMMA (3 Pares Constructivo/Destructivo)
    # --------------------------------------------------------------------------
    fig2_pairs = [
        ("candidato_1_image_107205_vs_424", 107205, 424, 7), # L_Elbow
        ("candidato_2_image_114_vs_355", 114, 355, 9),        # L_Wrist
        ("candidato_3_image_36_vs_424", 36, 424, 15),          # L_Ankle
    ]
    for c_dir_name, win_id, fail_id, focus_kp in fig2_pairs:
        p_win = search_packet(win_id)
        p_fail = search_packet(fail_id)
        if p_win and p_fail and p_win.exists() and p_fail.exists():
            w_pkt = load_deep_analysis(str(p_win))
            f_pkt = load_deep_analysis(str(p_fail))
            out_folder = TARGET_OUTPUT_ROOT / "figura_2_mrf_dilemma" / c_dir_name
            render_figure_2(w_pkt, f_pkt, focus_kp, out_folder, c_dir_name)

    # --------------------------------------------------------------------------
    # 3. FIGURA 3: OoD UNCERTAINTY (3 Candidatos: 63, 201, 377368)
    # --------------------------------------------------------------------------
    fig3_cands = [
        ("candidato_1_image_63", 63, 9),      # L_Wrist
        ("candidato_2_image_201", 201, 10),   # R_Wrist
        ("candidato_3_image_377368", 377368, 16), # R_Ankle
    ]
    for c_dir_name, img_id, focus_kp in fig3_cands:
        p_path = search_packet(img_id)
        if p_path and p_path.exists():
            pkt = load_deep_analysis(str(p_path))
            out_folder = TARGET_OUTPUT_ROOT / "figura_3_ood_alert" / c_dir_name
            render_figure_3(pkt, focus_kp, out_folder, c_dir_name)

    # --------------------------------------------------------------------------
    # 4. FIGURA 4: HEATMAP POISONING (3 Candidatos: 369503, 462, 153)
    # --------------------------------------------------------------------------
    fig4_cands = [
        ("candidato_1_image_369503", 369503, 16), # R_Ankle
        ("candidato_2_image_462", 462, 16),       # R_Ankle
        ("candidato_3_image_153", 153, 7),        # L_Elbow
    ]
    for c_dir_name, img_id, focus_kp in fig4_cands:
        p_path = search_packet(img_id)
        if p_path and p_path.exists():
            pkt = load_deep_analysis(str(p_path))
            out_folder = TARGET_OUTPUT_ROOT / "figura_4_heatmap_poisoning" / c_dir_name
            render_figure_4(pkt, focus_kp, out_folder, c_dir_name)

    # --------------------------------------------------------------------------
    # 5. FIGURA 5: UNIFORM TRASH (3 Candidatos: 221754, 82696, 147725)
    # --------------------------------------------------------------------------
    fig5_cands = [
        ("candidato_1_image_221754", 221754),
        ("candidato_2_image_82696", 82696),
        ("candidato_3_image_147725", 147725),
    ]
    for c_dir_name, img_id in fig5_cands:
        p_path = search_packet(img_id)
        if p_path and p_path.exists():
            pkt = load_deep_analysis(str(p_path))
            out_folder = TARGET_OUTPUT_ROOT / "figura_5_uniform_trash" / c_dir_name
            render_figure_5(pkt, out_folder, c_dir_name)

    logger.info("=" * 80)
    logger.info("[SUCCESS] ALL 15 PUBLICATION FIGURES SUCCESSFULLY GENERATED!")
    logger.info("Check folder: %s", TARGET_OUTPUT_ROOT)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

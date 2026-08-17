"""
Paper Packet Figure Exporter: Extract and render publication-ready figures from Deep Profiling packets.

Features:
1. Loads any .pkl.gz packet (or searches by --image-id).
2. Maps all GMM components, MC sampling points, and predictions from heatmap space to image space
   using exact MMPose affine transforms.
3. Generates high-resolution multi-panel publication figures (PNG / PDF at 300 DPI).
4. Exports modular assets: cropped joint zooms, transparent heatmap overlays, clean image,
   and complete numerical JSON data (means, covariances, weights, OKS).

Usage:
    # Generate multi-panel publication figure for a specific packet
    poetry run python src/experiments/visualizations/export_paper_packet_figure.py --packet "outputs/2026-07-06/12-38-24/Wins_31.pkl.gz"

    # Search by image ID in a directory and zoom in on specific keypoint (e.g. L_Shoulder or auto)
    poetry run python src/experiments/visualizations/export_paper_packet_figure.py --image-id 31 --data-dir "outputs/2026-07-06/12-38-24" --focus-kp auto

    # Export all modular assets (PDFs, raw PNGs, cropped zooms, JSON data)
    poetry run python src/experiments/visualizations/export_paper_packet_figure.py --packet "outputs/2026-07-06/12-38-24/Wins_462.pkl.gz" --export-modular
"""

from __future__ import annotations

import argparse
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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# COCO 17 Keypoints Definition
COCO_KP_NAMES = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
]

# Standard COCO Skeleton Connections (pairs of keypoint indices)
COCO_SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),                # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # Legs
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper-ready figures and assets from analysis packets.")
    parser.add_argument(
        "--packet",
        type=str,
        default=None,
        help="Path to .pkl.gz packet file",
    )
    parser.add_argument(
        "--image-id",
        type=int,
        default=None,
        help="Search for packet by image ID",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory to search packets in if --image-id is given (defaults to latest outputs/)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="paper_figures",
        help="Output directory to save figures and assets (default: paper_figures)",
    )
    parser.add_argument(
        "--focus-kp",
        type=str,
        default="auto",
        help="Keypoint to zoom into: name (e.g., L_Wrist), index (0-16), or 'auto' for max delta OKS / bimodal",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI resolution (default: 300)",
    )
    parser.add_argument(
        "--export-modular",
        action="store_true",
        help="If set, also exports individual transparent layers, crops, heatmaps, and JSON numerical data",
    )
    return parser.parse_args()


# ==============================================================================
# Helper Functions: Image Recovery & Spatial Transformations
# ==============================================================================

def find_original_image(packet: Dict[str, Any]) -> Optional[npt.NDArray]:
    """Extract full-resolution RGB image from packet."""
    tta_data = packet.get("tta_data", [])
    for entry in tta_data:
        if entry.get("name", "").startswith("original"):
            img = entry.get("image")
            if isinstance(img, np.ndarray):
                return img
    for entry in tta_data:
        img = entry.get("image")
        if isinstance(img, np.ndarray):
            return img
    return None


def ellipse_polygon_vertices(mu: npt.NDArray, cov: npt.NDArray, n_sigma: float = 2.0, n_pts: int = 64) -> npt.NDArray:
    """Parametric 2D confidence ellipse from (mean, covariance) -> (n_pts, 2) array."""
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


def map_gmm_to_image_space_exact(
    packet: Dict[str, Any],
    means: npt.NDArray,
    covs: npt.NDArray,
    img_w: int,
    img_h: int,
) -> Tuple[npt.NDArray, npt.NDArray, Optional[npt.NDArray], Optional[npt.NDArray]]:
    """Map heatmap-space GMM means and covariances to image space using MMPose affine mapping."""
    means_arr = np.asarray(means, dtype=np.float32)
    covs_arr = np.asarray(covs, dtype=np.float32)

    if means_arr.ndim == 1:
        means_arr = means_arr.reshape(1, -1)
    if covs_arr.ndim == 2:
        covs_arr = covs_arr.reshape(1, 2, 2)

    n = min(means_arr.shape[0], covs_arr.shape[0])
    means_arr = means_arr[:n, :2]
    covs_arr = covs_arr[:n]

    hm = packet.get("aggregation", {}).get("heatmap_avg")
    metadata = packet.get("aggregation", {}).get("mmpose_metadata")

    if isinstance(hm, np.ndarray) and isinstance(metadata, dict):
        try:
            ref_hm = StandardizedHeatmap(
                data=np.asarray(hm, dtype=np.float32),
                original_size=(img_h, img_w),
                metadata=metadata,
            )
            means_img, covs_img = MMPoseAdapter.transform_heatmap_gaussians_to_image(
                means_arr, covs_arr, ref_hm
            )
            A, b = MMPoseAdapter._heatmap_to_image_affine(ref_hm)
            return np.asarray(means_img, dtype=np.float64), np.asarray(covs_img, dtype=np.float64), A, b
        except Exception as e:
            logger.debug("MMPose affine failed, using bbox fallback: %s", e)

    # Fallback to Bounding Box scaling
    gt = packet.get("ground_truth", {})
    bbox = gt.get("bboxes", gt.get("bbox"))
    x0, y0 = 0.0, 0.0
    box_w, box_h = float(img_w), float(img_h)

    if bbox is not None:
        b_arr = np.asarray(bbox, dtype=np.float64).reshape(-1)
        if b_arr.size >= 4 and np.all(np.isfinite(b_arr[:4])):
            bx, by, bw, bh = map(float, b_arr[:4])
            if 0.0 <= bx <= 1.0 and 0.0 <= by <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0:
                bx *= float(img_w)
                by *= float(img_h)
                bw *= float(img_w)
                bh *= float(img_h)
            if bw > 0 and bh > 0:
                x0, y0, box_w, box_h = bx, by, bw, bh

    hm_w = hm.shape[2] if isinstance(hm, np.ndarray) and hm.ndim == 3 else 64
    hm_h = hm.shape[1] if isinstance(hm, np.ndarray) and hm.ndim == 3 else 48

    sx = box_w / float(hm_w)
    sy = box_h / float(hm_h)

    means_img = means_arr.copy().astype(np.float64)
    means_img[:, 0] = x0 + means_arr[:, 0] * sx
    means_img[:, 1] = y0 + means_arr[:, 1] * sy

    A = np.array([[sx, 0.0], [0.0, sy]], dtype=np.float64)
    b_vec = np.array([x0, y0], dtype=np.float64)
    covs_img = np.empty((n, 2, 2), dtype=np.float64)
    for i in range(n):
        c = covs_arr[i].astype(np.float64)
        covs_img[i] = A @ c @ A.T

    return means_img, covs_img, A, b_vec


def map_points_to_image_space(
    points_hm: npt.NDArray,
    A: Optional[npt.NDArray],
    b: Optional[npt.NDArray],
) -> npt.NDArray:
    """Apply affine transform A * x + b to an array of points (N, 2)."""
    if points_hm.ndim != 2 or points_hm.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    if A is None or b is None:
        return points_hm.astype(np.float64)

    pts = points_hm[:, :2].astype(np.float64)
    return (pts @ A.T) + b.reshape(1, 2)


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

    _, _, A, b = map_gmm_to_image_space_exact(packet, np.zeros((avg_hm.shape[0], 2)), np.zeros((avg_hm.shape[0], 2, 2)), img_w, img_h)

    coords_list = []
    for k in range(avg_hm.shape[0]):
        pt_hm = decode_dark_from_heatmap_2d(avg_hm[k])
        if A is not None and b is not None:
            pt_img = (A @ pt_hm) + b
        else:
            pt_img = pt_hm
        coords_list.append(pt_img)
    return np.array(coords_list, dtype=np.float64)


# ==============================================================================
# Focus Keypoint Selection
# ==============================================================================

def select_focus_keypoint(packet: Dict[str, Any], focus_arg: str) -> int:
    """Identify the keypoint index to focus/zoom on."""
    if focus_arg.isdigit():
        idx = int(focus_arg)
        if 0 <= idx < len(COCO_KP_NAMES):
            return idx

    for i, name in enumerate(COCO_KP_NAMES):
        if name.lower() == focus_arg.lower():
            return i

    # Auto selection: Prioritize bimodal joint > highest delta OKS
    gmm_per_kp = packet.get("gmm_per_kp", [])
    for k, kp_info in enumerate(gmm_per_kp):
        if isinstance(kp_info, dict):
            w = np.sort(np.asarray(kp_info.get("weights", []), dtype=np.float64))[::-1]
            if len(w) >= 2 and w[1] >= 0.25:
                return k

    # Fallback to keypoint with largest delta OKS
    metrics = packet.get("metrics", {})
    p_ours = metrics.get("per_kp_oks_ours")
    p_base = metrics.get("per_kp_oks_base")
    if p_ours is not None and p_base is not None:
        deltas = np.asarray(p_ours, dtype=np.float32) - np.asarray(p_base, dtype=np.float32)
        valid_idx = np.where(~np.isnan(deltas))[0]
        if len(valid_idx) > 0:
            return int(valid_idx[np.argmax(deltas[valid_idx])])

    return 9  # Default to L_Wrist


# ==============================================================================
# Main Plotting & Export Engines
# ==============================================================================

def render_publication_figure(
    packet: Dict[str, Any],
    focus_kp_idx: int,
    out_file: Path,
    dpi: int = 300,
) -> None:
    """Generate a clean 4-panel publication-grade figure."""
    img_rgb = find_original_image(packet)
    if img_rgb is None:
        logger.error("No RGB image found in packet.")
        return

    img_h, img_w = img_rgb.shape[:2]
    meta = packet.get("meta", {})
    metrics = packet.get("metrics", {})
    gt_coords = packet.get("ground_truth", {}).get("coords")
    # Decode DARK directly on the aggregated TTA heatmap for true apple-to-apple comparison
    base_coords = get_dark_coords_from_aggregated_heatmap(packet, img_w, img_h)
    ours_coords = packet.get("predictions", {}).get("ours_gmm", {}).get("coords")
    gmm_per_kp = packet.get("gmm_per_kp", [])
    agg = packet.get("aggregation", {})
    avg_hm = agg.get("heatmap_avg")
    samp_by_kp = agg.get("sampling_points_by_kp", [])
    tta_data = packet.get("tta_data", [])

    # Map GMM components to image space
    means_img_list, covs_img_list = [], []
    A_mat, b_vec = None, None
    for k in range(len(COCO_KP_NAMES)):
        if k < len(gmm_per_kp) and isinstance(gmm_per_kp[k], dict):
            m = np.asarray(gmm_per_kp[k].get("means", []), dtype=np.float64)
            c = np.asarray(gmm_per_kp[k].get("covariances", []), dtype=np.float64)
            if m.ndim == 2 and c.ndim == 3 and m.shape[0] > 0:
                m_img, c_img, A_mat, b_vec = map_gmm_to_image_space_exact(packet, m, c, img_w, img_h)
                means_img_list.append(m_img)
                covs_img_list.append(c_img)
            else:
                means_img_list.append(np.empty((0, 2)))
                covs_img_list.append(np.empty((0, 2, 2)))
        else:
            means_img_list.append(np.empty((0, 2)))
            covs_img_list.append(np.empty((0, 2, 2)))

    # Set up publication figure style
    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(20, 11), dpi=dpi)

    # Grid layout: 2 rows
    # Top Row: [Panel A: Full Image (Width 2)] [Panel B: Zoomed Crop (Width 1.5)] [Panel D: Heatmap + GMM (Width 1.5)]
    # Bottom Row: [Panel C: TTA Heatmap Decomposition Grid (Span all)]
    gs = fig.add_gridspec(2, 4, height_ratios=[1.3, 0.7], hspace=0.25, wspace=0.20)

    ax_full = fig.add_subplot(gs[0, 0:2])
    ax_zoom = fig.add_subplot(gs[0, 2])
    ax_hm = fig.add_subplot(gs[0, 3])
    ax_tta = fig.add_subplot(gs[1, :])

    # --------------------------------------------------------------------------
    # Panel A: Full Image with Skeletons & Uncertainty Ellipses
    # --------------------------------------------------------------------------
    ax_full.imshow(img_rgb)
    ax_full.set_title(
        f"(a) Global Pose Estimation & Spatial Uncertainty [Image ID: {meta.get('image_id', 'N/A')}]",
        fontsize=14, fontweight="bold", pad=8
    )

    # Draw Ground Truth (Green solid lines & circles)
    if isinstance(gt_coords, np.ndarray) and gt_coords.ndim == 2:
        for p1, p2 in COCO_SKELETON_PAIRS:
            if p1 < len(gt_coords) and p2 < len(gt_coords):
                if gt_coords[p1, 2] > 0 and gt_coords[p2, 2] > 0:
                    ax_full.plot([gt_coords[p1, 0], gt_coords[p2, 0]],
                                 [gt_coords[p1, 1], gt_coords[p2, 1]],
                                 color="#00E676", linewidth=2.5, alpha=0.85)
        for k in range(min(len(gt_coords), len(COCO_KP_NAMES))):
            if gt_coords[k, 2] > 0:
                ax_full.scatter(gt_coords[k, 0], gt_coords[k, 1], color="#00E676", s=40, zorder=5)

    # Draw Baseline (Red dashed lines & markers)
    if isinstance(base_coords, np.ndarray) and base_coords.ndim == 2:
        for p1, p2 in COCO_SKELETON_PAIRS:
            if p1 < len(base_coords) and p2 < len(base_coords):
                ax_full.plot([base_coords[p1, 0], base_coords[p2, 0]],
                             [base_coords[p1, 1], base_coords[p2, 1]],
                             color="#FF1744", linestyle="--", linewidth=2.0, alpha=0.75)
        ax_full.scatter(base_coords[:, 0], base_coords[:, 1], color="#FF1744", marker="x", s=50, zorder=6)

    # Draw Ours GMM (Cyan solid lines & dots)
    if isinstance(ours_coords, np.ndarray) and ours_coords.ndim == 2:
        for p1, p2 in COCO_SKELETON_PAIRS:
            if p1 < len(ours_coords) and p2 < len(ours_coords):
                ax_full.plot([ours_coords[p1, 0], ours_coords[p2, 0]],
                              [ours_coords[p1, 1], ours_coords[p2, 1]],
                              color="#00E5FF", linewidth=2.5, alpha=0.9)
        ax_full.scatter(ours_coords[:, 0], ours_coords[:, 1], color="#00E5FF", s=45, zorder=7)

    # Draw Uncertainty Ellipses (2-sigma, Orange/Gold)
    for k in range(len(means_img_list)):
        m_k = means_img_list[k]
        c_k = covs_img_list[k]
        for j in range(len(m_k)):
            verts = ellipse_polygon_vertices(m_k[j], c_k[j], n_sigma=2.0)
            if len(verts) > 0:
                poly = patches.Polygon(verts, closed=True, fill=False, edgecolor="#FFD600", linewidth=1.8, linestyle="-", zorder=8)
                ax_full.add_patch(poly)

    # Add custom legend
    legend_elements = [
        plt.Line2D([0], [0], color="#00E676", lw=2.5, label=r"$\mathbf{Ground\ Truth}$"),
        plt.Line2D([0], [0], color="#FF1744", lw=2.0, linestyle="--", marker="x", label=f"DARK on Aggregated Heatmap (OKS: {metrics.get('oks_tta', metrics.get('oks_base', 0)):.3f})"),
        plt.Line2D([0], [0], color="#00E5FF", lw=2.5, marker="o", label=f"Ours GMM (OKS: {metrics.get('oks_ours', 0):.3f}, $\Delta_{{\mathrm{{TTA}}}}$: {metrics.get('delta_oks_ours_over_tta', metrics.get('delta_oks', 0)):+.3f})"),
        patches.Patch(edgecolor="#FFD600", facecolor="none", lw=1.8, label=r"Ours Uncertainty ($2\sigma$ Ellipse)"),
    ]
    ax_full.legend(handles=legend_elements, loc="upper left", framealpha=0.85, fontsize=10)
    ax_full.axis("off")

    # --------------------------------------------------------------------------
    # Panel B: High-Res Zoom on the Focused Joint
    # --------------------------------------------------------------------------
    focus_name = COCO_KP_NAMES[focus_kp_idx]
    
    # Calculate crop center (prefer GT > Ours > Base)
    if isinstance(gt_coords, np.ndarray) and gt_coords[focus_kp_idx, 2] > 0:
        cx, cy = float(gt_coords[focus_kp_idx, 0]), float(gt_coords[focus_kp_idx, 1])
    elif isinstance(ours_coords, np.ndarray):
        cx, cy = float(ours_coords[focus_kp_idx, 0]), float(ours_coords[focus_kp_idx, 1])
    else:
        cx, cy = float(base_coords[focus_kp_idx, 0]), float(base_coords[focus_kp_idx, 1])

    # Crop box size: 25% of image width/height or ~120px
    crop_size = max(80, int(min(img_w, img_h) * 0.25))
    x_min = max(0, int(cx - crop_size))
    x_max = min(img_w, int(cx + crop_size))
    y_min = max(0, int(cy - crop_size))
    y_max = min(img_h, int(cy + crop_size))

    crop_img = img_rgb[y_min:y_max, x_min:x_max]
    ax_zoom.imshow(crop_img, extent=[x_min, x_max, y_max, y_min])

    # Draw Zoom Details: MC Samples
    if len(samp_by_kp) > focus_kp_idx:
        pts_hm = np.asarray(samp_by_kp[focus_kp_idx])
        if pts_hm.ndim == 2 and len(pts_hm) > 0:
            pts_img = map_points_to_image_space(pts_hm, A_mat, b_vec)
            ax_zoom.scatter(pts_img[:, 0], pts_img[:, 1], color="#FFAB00", s=8, alpha=0.35, zorder=4, label="MC Samples")

    # Draw Zoom Details: GT, Base, Ours Keypoints
    if isinstance(gt_coords, np.ndarray) and gt_coords[focus_kp_idx, 2] > 0:
        ax_zoom.scatter(gt_coords[focus_kp_idx, 0], gt_coords[focus_kp_idx, 1],
                        color="#00E676", s=120, edgecolors="black", linewidth=1.5, zorder=8, label="GT")
    if isinstance(base_coords, np.ndarray):
        ax_zoom.scatter(base_coords[focus_kp_idx, 0], base_coords[focus_kp_idx, 1],
                        color="#FF1744", marker="x", s=120, linewidth=3, zorder=9, label="DARK (Base)")
    if isinstance(ours_coords, np.ndarray):
        ax_zoom.scatter(ours_coords[focus_kp_idx, 0], ours_coords[focus_kp_idx, 1],
                        color="#00E5FF", s=120, edgecolors="black", linewidth=1.5, zorder=10, label="Ours (Mode)")

    # Draw Zoom GMM Ellipses with weights
    m_focus = means_img_list[focus_kp_idx]
    c_focus = covs_img_list[focus_kp_idx]
    w_focus = gmm_per_kp[focus_kp_idx].get("weights", []) if focus_kp_idx < len(gmm_per_kp) else []

    for j in range(len(m_focus)):
        # 1-sigma & 2-sigma ellipses
        v1 = ellipse_polygon_vertices(m_focus[j], c_focus[j], n_sigma=1.0)
        v2 = ellipse_polygon_vertices(m_focus[j], c_focus[j], n_sigma=2.0)
        if len(v1) > 0:
            ax_zoom.add_patch(patches.Polygon(v1, closed=True, fill=False, edgecolor="#FFD600", linewidth=1.5, linestyle=":", zorder=6))
        if len(v2) > 0:
            ax_zoom.add_patch(patches.Polygon(v2, closed=True, fill=False, edgecolor="#FFD600", linewidth=2.2, linestyle="-", zorder=7))
            # Annotation text with component weight
            w_val = float(w_focus[j]) if j < len(w_focus) else 1.0
            ax_zoom.text(m_focus[j, 0] + 3, m_focus[j, 1] - 3, f"$w_{j+1}={w_val:.2f}$",
                         color="#FFD600", fontsize=10, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.65, edgecolor="none"))

    p_ours_val = metrics.get("per_kp_oks_ours", [np.nan])[focus_kp_idx] if metrics.get("per_kp_oks_ours") is not None else np.nan
    p_base_val = metrics.get("per_kp_oks_base", [np.nan])[focus_kp_idx] if metrics.get("per_kp_oks_base") is not None else np.nan
    delta_val = p_ours_val - p_base_val if not np.isnan(p_ours_val) and not np.isnan(p_base_val) else 0.0

    ax_zoom.set_title(
        f"(b) Joint Zoom: {focus_name}\n[OKS: {p_ours_val:.3f} | $\Delta$: {delta_val:+.3f}]",
        fontsize=13, fontweight="bold", pad=8
    )
    ax_zoom.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax_zoom.axis("off")

    # Draw border on zoom
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor("#FFD600")
        spine.set_linewidth(2.0)
        spine.set_visible(True)

    # --------------------------------------------------------------------------
    # Panel D: Aggregated Heatmap & Continuous Mixture Fit
    # --------------------------------------------------------------------------
    if isinstance(avg_hm, np.ndarray) and focus_kp_idx < avg_hm.shape[0]:
        hm_k = avg_hm[focus_kp_idx]
        hm_h, hm_w = hm_k.shape
        im_hm = ax_hm.imshow(hm_k, cmap="inferno", extent=[0, hm_w, hm_h, 0], interpolation="bilinear")
        plt.colorbar(im_hm, ax=ax_hm, fraction=0.046, pad=0.04)

        # Overlay Heatmap-space GMM components
        if focus_kp_idx < len(gmm_per_kp):
            kp_gmm = gmm_per_kp[focus_kp_idx]
            m_hm = np.asarray(kp_gmm.get("means", []), dtype=np.float64)
            c_hm = np.asarray(kp_gmm.get("covariances", []), dtype=np.float64)
            for j in range(len(m_hm)):
                v2_hm = ellipse_polygon_vertices(m_hm[j], c_hm[j], n_sigma=2.0)
                if len(v2_hm) > 0:
                    ax_hm.add_patch(patches.Polygon(v2_hm, closed=True, fill=False, edgecolor="#00E5FF", linewidth=2.0))
                ax_hm.scatter(m_hm[j, 0], m_hm[j, 1], color="#00E5FF", marker="+", s=80, linewidth=2)

        ax_hm.set_title(
            f"(c) Aggregated Heatmap $\\bar{{H}}_{{{focus_name}}}$ + GMM Fit",
            fontsize=13, fontweight="bold", pad=8
        )
        ax_hm.axis("off")

    # --------------------------------------------------------------------------
    # Panel C: TTA Decomposition Grid (Individual TTA Transforms)
    # --------------------------------------------------------------------------
    n_tta = min(len(tta_data), 10)
    if n_tta > 0:
        # Create sub-gridspec inside ax_tta
        tta_gs = gs[1, :].subgridspec(1, n_tta, wspace=0.15)
        ax_tta.axis("off")  # hide outer container
        ax_tta.set_title("(d) TTA Decomposition: Per-Augmentation Predicted Heatmaps", fontsize=13, fontweight="bold", pad=12)

        for i in range(n_tta):
            ax_sub = fig.add_subplot(tta_gs[0, i])
            entry = tta_data[i]
            name = entry.get("name", f"TTA_{i}")
            scale_f = entry.get("scale_factor", 1.0)
            hm_entry = entry.get("heatmap")
            if isinstance(hm_entry, np.ndarray) and focus_kp_idx < hm_entry.shape[0]:
                ax_sub.imshow(hm_entry[focus_kp_idx], cmap="inferno", interpolation="bilinear")
            
            # Short title
            short_t = f"TTA {i}\n(x{scale_f:.2f})"
            if "flip" in name.lower(): short_t += " [Flip]"
            ax_sub.set_title(short_t, fontsize=9)
            ax_sub.axis("off")

    # Save Composite Figure
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("[OK] Saved composite publication figure: %s", out_file.resolve())


# ==============================================================================
# Modular Asset Exporter
# ==============================================================================

def export_modular_assets(
    packet: Dict[str, Any],
    focus_kp_idx: int,
    out_dir: Path,
    dpi: int = 300,
) -> None:
    """Export clean raw image, transparent layers, individual crops, and JSON numerical data."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img_rgb = find_original_image(packet)
    if img_rgb is None:
        return

    img_h, img_w = img_rgb.shape[:2]
    focus_name = COCO_KP_NAMES[focus_kp_idx]

    # 1. Clean original image
    cv2.imwrite(str(out_dir / "original_clean.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # 2. Keypoint Zoom Crop (Clean)
    gt_coords = packet.get("ground_truth", {}).get("coords")
    ours_coords = packet.get("predictions", {}).get("ours_gmm", {}).get("coords")
    base_coords = packet.get("predictions", {}).get("baseline", {}).get("coords")

    if isinstance(gt_coords, np.ndarray) and gt_coords[focus_kp_idx, 2] > 0:
        cx, cy = float(gt_coords[focus_kp_idx, 0]), float(gt_coords[focus_kp_idx, 1])
    elif isinstance(ours_coords, np.ndarray):
        cx, cy = float(ours_coords[focus_kp_idx, 0]), float(ours_coords[focus_kp_idx, 1])
    else:
        cx, cy = float(base_coords[focus_kp_idx, 0]), float(base_coords[focus_kp_idx, 1])

    crop_size = max(80, int(min(img_w, img_h) * 0.25))
    x_min, x_max = max(0, int(cx - crop_size)), min(img_w, int(cx + crop_size))
    y_min, y_max = max(0, int(cy - crop_size)), min(img_h, int(cy + crop_size))
    crop_img = img_rgb[y_min:y_max, x_min:x_max]
    cv2.imwrite(str(out_dir / f"crop_clean_{focus_name}.png"), cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

    # 3. Aggregated Heatmap for focus keypoint
    agg = packet.get("aggregation", {})
    avg_hm = agg.get("heatmap_avg")
    if isinstance(avg_hm, np.ndarray) and focus_kp_idx < avg_hm.shape[0]:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        ax.imshow(avg_hm[focus_kp_idx], cmap="inferno")
        ax.axis("off")
        plt.savefig(out_dir / f"heatmap_avg_{focus_name}.png", bbox_inches="tight", pad_inches=0)
        plt.savefig(out_dir / f"heatmap_avg_{focus_name}.pdf", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    # 4. Numerical JSON Data Export
    gmm_per_kp = packet.get("gmm_per_kp", [])
    gmm_numerical = []
    for k in range(len(COCO_KP_NAMES)):
        if k < len(gmm_per_kp) and isinstance(gmm_per_kp[k], dict):
            m_hm = np.asarray(gmm_per_kp[k].get("means", []), dtype=np.float64)
            c_hm = np.asarray(gmm_per_kp[k].get("covariances", []), dtype=np.float64)
            w = np.asarray(gmm_per_kp[k].get("weights", []), dtype=np.float64)
            m_img, c_img, A, b = map_gmm_to_image_space_exact(packet, m_hm, c_hm, img_w, img_h)
            gmm_numerical.append({
                "keypoint": COCO_KP_NAMES[k],
                "weights": w.tolist(),
                "means_heatmap": m_hm.tolist(),
                "covariances_heatmap": c_hm.tolist(),
                "means_image": m_img.tolist(),
                "covariances_image": c_img.tolist(),
            })

    json_export = {
        "meta": packet.get("meta", {}),
        "metrics": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in packet.get("metrics", {}).items()
        },
        "ground_truth_coords": gt_coords.tolist() if isinstance(gt_coords, np.ndarray) else None,
        "baseline_coords": base_coords.tolist() if isinstance(base_coords, np.ndarray) else None,
        "ours_gmm_coords": ours_coords.tolist() if isinstance(ours_coords, np.ndarray) else None,
        "gmm_components": gmm_numerical,
        "focus_keypoint": focus_name,
        "crop_box_xyxy": [x_min, y_min, x_max, y_max],
    }

    with open(out_dir / "packet_numerical_data.json", "w", encoding="utf-8") as f:
        json.dump(json_export, f, indent=2)

    logger.info("[OK] Exported modular assets to: %s", out_dir.resolve())


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main() -> None:
    args = parse_args()

    packet_path = None
    if args.packet:
        packet_path = Path(args.packet)
    elif args.image_id is not None:
        data_dir = Path(args.data_dir) if args.data_dir else Path("outputs")
        matches = list(data_dir.rglob(f"*_{args.image_id}.pkl.gz"))
        if matches:
            packet_path = matches[0]
        else:
            logger.error("No packet found with image ID %d in %s", args.image_id, data_dir)
            return
    else:
        logger.error("Please provide either --packet <path> or --image-id <id>")
        return

    if not packet_path.exists():
        logger.error("Packet file not found: %s", packet_path)
        return

    logger.info("Loading analysis packet: %s", packet_path)
    packet = load_deep_analysis(str(packet_path))

    focus_kp_idx = select_focus_keypoint(packet, args.focus_kp)
    logger.info("Selected focus keypoint: [%d] %s", focus_kp_idx, COCO_KP_NAMES[focus_kp_idx])

    meta = packet.get("meta", {})
    img_id = meta.get("image_id", packet_path.stem)
    sample_type = meta.get("sample_type", "Sample")

    out_base = Path(args.out_dir) / f"{sample_type}_{img_id}"

    # Render composite multi-panel figure
    composite_png = out_base / f"figure_{sample_type}_{img_id}_{COCO_KP_NAMES[focus_kp_idx]}.png"
    composite_pdf = out_base / f"figure_{sample_type}_{img_id}_{COCO_KP_NAMES[focus_kp_idx]}.pdf"
    render_publication_figure(packet, focus_kp_idx, composite_png, dpi=args.dpi)
    render_publication_figure(packet, focus_kp_idx, composite_pdf, dpi=args.dpi)

    if args.export_modular:
        export_modular_assets(packet, focus_kp_idx, out_base / "assets", dpi=args.dpi)


if __name__ == "__main__":
    main()

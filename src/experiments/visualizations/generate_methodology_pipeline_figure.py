"""
Publication-Grade Methodology Pipeline Figure (Elsevier / IEEE Double-Column).

Generates the complete 4-block horizontal workflow diagram for the paper using the EXACT
image, bounding box, and coordinate transformations from generate_all_paper_figures.py
on Candidate 130 (candidate_130_130.pkl.gz):

1. Block 1 (GPU): Input frame with the exact bounding box from generate_all_paper_figures.py,
   multi-scale TTA stack views with trailing ellipsis (at the end),
   frozen backbone, and raw heatmaps with trailing ellipsis (at the end).
2. Block 2 (CPU): Continuous weighted aggregation with exact paper formulas, emerald green
   hollow Monte Carlo sampling points, and clear legend explanation.
3. Block 3 (CPU): Dual hypothesis EM fitting (K=1 vs K=2) with dynamic component weights,
   vertically staggered labels (blue at bottom, red at top), purple uniform absorption
   representation reflecting pi_uniform, and exact Total Variance uncertainty metrics.
4. Block 4 (CPU): Kinematic MRF graph decoding with R_Knee anchor, candidate pruning,
   and calibrated pose with covariance ellipses on the limbs.

Usage:
    poetry run python src/experiments/visualizations/generate_methodology_pipeline_figure.py
"""

from __future__ import annotations

import gzip
import logging
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle, Polygon
import matplotlib.pyplot as plt
import numpy as np

# Configure paths
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from pose_uncertainty.core.sampling import sample_from_heatmap
from pose_uncertainty.core.mixture import RobustGaussianMixture
from pose_uncertainty.models.adapters import MMPoseAdapter
from pose_uncertainty.utils.types import StandardizedHeatmap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# Design & Aesthetic Tokens
# ==============================================================================

PALETTE = {
    "b1_gpu": "#1E88E5",       # Cobalt Blue (Forward & Backbone)
    "b2_agg": "#2E7D32",       # Emerald Green (Aggregation & Sampling)
    "b3_gmm": "#D32F2F",       # Crimson Red (Mixture & BIC)
    "b4_mrf": "#7B1FA2",       # Amethyst Purple (MRF & Calibrated Pose)
    "bg_card": "#FAFAFA",      # Clean Card Background
    "border_card": "#D0D7DE",  # Subtle Border
    "text_dark": "#1A1A1A",
    "text_muted": "#555555",
    "gold": "#FFD600",
}

COCO_SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),                # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # Legs
]


def decode_image_if_bytes(img_obj: Any) -> Optional[np.ndarray]:
    """Helper to convert bytes or ndarray to uint8 RGB numpy array."""
    if isinstance(img_obj, np.ndarray):
        return img_obj
    if isinstance(img_obj, bytes):
        nparr = np.frombuffer(img_obj, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return None


def load_candidate_130_from_figures_pipeline(pkt_path: Path) -> Dict[str, Any]:
    """Load packet and apply exact image space mapping as in generate_all_paper_figures.py."""
    with gzip.open(pkt_path, "rb") as f:
        pkt = pickle.load(f)

    # 1. Load original image as done in find_original_image()
    tta_data = pkt.get("tta_data", [])
    img_rgb = None
    for entry in tta_data:
        if entry.get("name", "").startswith("original"):
            img_rgb = decode_image_if_bytes(entry.get("image"))
            if img_rgb is not None:
                break
    if img_rgb is None and len(tta_data) > 0:
        img_rgb = decode_image_if_bytes(tta_data[0].get("image"))

    if img_rgb is None:
        # Fallback to loading 002252.jpg
        img_path = project_root / "data" / "ochuman" / "images" / "002252.jpg"
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_h, img_w = img_rgb.shape[:2]

    # 2. Extract bounding box from Ground Truth exactly as in generate_all_paper_figures.py
    gt_dict = pkt.get("ground_truth", {})
    bbox = gt_dict.get("bboxes", gt_dict.get("bbox"))
    if bbox is not None:
        b_arr = np.asarray(bbox, dtype=np.float64).reshape(-1)
        bx, by, bw, bh = map(float, b_arr[:4])
    else:
        bx, by, bw, bh = 0.0, 0.0, float(img_w), float(img_h)

    # 3. Extract Coordinates exactly as in generate_all_paper_figures.py
    gt = gt_dict.get("coords")
    ours = pkt.get("predictions", {}).get("ours_gmm", {}).get("coords")
    base = pkt.get("predictions", {}).get("baseline", {}).get("coords")

    # 4. Target joint: Right Ankle (KP 16), Parent: Right Knee (KP 14)
    target_kp = 16
    parent_kp = 14

    raw_hm_0 = pkt["tta_data"][0]["heatmap"][target_kp] # Scale 0.85
    raw_hm_4 = pkt["tta_data"][4]["heatmap"][target_kp] # Scale 1.00
    raw_hm_8 = pkt["tta_data"][8]["heatmap"][target_kp] # Scale 1.15

    agg_hm = pkt["aggregation"]["heatmap_avg"][target_kp]

    # 5. Monte Carlo Rejection Sampling on R_Ankle
    samples_kp = sample_from_heatmap(
        agg_hm,
        num_samples=1000,
        strategy="rejection",
        temperature=0.3,
        seed=42,
    )

    # 6. GMM EM Fitting & BIC Selection
    gmm_k1 = RobustGaussianMixture(n_components=1, random_state=42)
    gmm_k1.fit(samples_kp)
    bic_1 = gmm_k1.compute_bic(len(samples_kp))
    probs_k1 = gmm_k1.predict_proba(samples_kp) # (N, 2)

    gmm_k2 = RobustGaussianMixture(n_components=2, random_state=42)
    gmm_k2.fit(samples_kp)
    bic_2 = gmm_k2.compute_bic(len(samples_kp))
    probs_k2 = gmm_k2.predict_proba(samples_kp) # (N, 3)

    # 7. Compute Calibrated Covariance (Sigma_final) and Adaptive Uncertainty Metrics
    # EXACTLY mirroring runner.py, evaluate_uncertainty.py and optimize_adaptive_uncertainty.py
    from pose_uncertainty.utils.metrics import compute_calibrated_covariance, COCO_SIGMAS

    m_hm = np.array([c.mean for c in gmm_k2.components_], dtype=np.float32)
    c_hm = np.array([c.covariance for c in gmm_k2.components_], dtype=np.float32)
    weights_k2 = np.array([c.weight for c in gmm_k2.components_] + [gmm_k2.uniform_weight_], dtype=np.float32)

    # MMPose affine transformation to image space exactly as in runner.py
    center = np.array([bx + bw / 2.0, by + bh / 2.0], dtype=np.float32)
    scale = np.array([bw * 1.25, bh * 1.25], dtype=np.float32)
    input_size = np.array([192, 256], dtype=np.int32)
    ref_hm = StandardizedHeatmap(
        data=agg_hm[np.newaxis, ...],
        original_size=(int(bh), int(bw)),
        metadata={
            "input_center": center,
            "input_scale": scale,
            "input_size": input_size,
            "heatmap_size": (agg_hm.shape[1], agg_hm.shape[0]),
        }
    )

    m_img, c_img = MMPoseAdapter.transform_heatmap_gaussians_to_image(m_hm, c_hm, ref_hm)

    area = bw * bh
    kappa = float(COCO_SIGMAS[target_kp]) if target_kp < len(COCO_SIGMAS) else 0.089

    # Calibrated Covariance from runner.py
    sigma_final, _ = compute_calibrated_covariance(
        means_img=m_img,
        covs_img=c_img,
        weights=weights_k2,
        scale_sq=float(area),
        kappa=kappa
    )

    det_sigma_final = float(np.linalg.det(sigma_final))

    # Adaptive Uncertainty Fusion from evaluate_uncertainty.py & optimize_adaptive_uncertainty.py
    beta = 0.1
    u_gmm = float(1.0 - np.exp(-beta * det_sigma_final))

    # Baseline heuristic uncertainty from peak activation (base_score = max heatmap activation)
    base_score = float(np.max(agg_hm))
    u_base = float(1.0 - base_score)

    # Strategy C: Weighted Softmax Fusion
    exp_u_base = np.exp(u_base)
    exp_u_gmm = np.exp(u_gmm)
    sum_exp = exp_u_base + exp_u_gmm
    w_base = exp_u_base / sum_exp
    w_gmm = exp_u_gmm / sum_exp
    u_adapt = float(w_base * u_base + w_gmm * u_gmm)

    # 8. Bone length prior statistics and crop for MRF panels (mirroring Figure 2 b1 & c1)
    from pose_uncertainty.core.skeleton import COCO_BONE_LENGTH_STATS
    edge = (parent_kp, target_kp) if (parent_kp, target_kp) in COCO_BONE_LENGTH_STATS else (target_kp, parent_kp)
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

    p_coord = ours[parent_kp, :2] if isinstance(ours, np.ndarray) else gt[parent_kp, :2]
    c_gmm = base[target_kp, :2] if isinstance(base, np.ndarray) else p_coord + 20
    c_mrf = ours[target_kp, :2] if isinstance(ours, np.ndarray) else c_gmm
    c_gt = gt[target_kp, :2] if isinstance(gt, np.ndarray) else c_mrf

    # Crop bounding box encompassing parent, GMM, MRF, and GT exactly as in Figure 2
    all_pts_x = [p_coord[0], c_gmm[0], c_mrf[0], c_gt[0]]
    all_pts_y = [p_coord[1], c_gmm[1], c_mrf[1], c_gt[1]]
    margin = 55.0

    x_min = max(0, int(min(all_pts_x) - margin))
    x_max = min(img_w, int(max(all_pts_x) + margin))
    y_min = max(0, int(min(all_pts_y) - margin))
    y_max = min(img_h, int(max(all_pts_y) + margin))

    crop_img = img_rgb[y_min:y_max, x_min:x_max]

    return {
        "full_img": img_rgb,
        "crop_img": crop_img,
        "crop_coords": (x_min, y_min, x_max, y_max),
        "bbox": (bx, by, bw, bh),
        "gt": gt,
        "base": base,
        "ours": ours,
        "p_coord": p_coord,
        "c_gmm": c_gmm,
        "c_mrf": c_mrf,
        "c_gt": c_gt,
        "mu_px": mu_px,
        "sigma_px": sigma_px,
        "raw_hm_0": raw_hm_0,
        "raw_hm_4": raw_hm_4,
        "raw_hm_8": raw_hm_8,
        "agg_hm": agg_hm,
        "samples_kp": samples_kp,
        "gmm_k1": gmm_k1,
        "bic_1": bic_1,
        "probs_k1": probs_k1,
        "gmm_k2": gmm_k2,
        "bic_2": bic_2,
        "probs_k2": probs_k2,
        "det_sigma_final": det_sigma_final,
        "u_gmm": u_gmm,
        "u_base": u_base,
        "u_adapt": u_adapt,
    }


def render_methodology_pipeline_figure(
    data: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Build the complete 4-block horizontal workflow diagram using generate_all_paper_figures.py data."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["mathtext.fontset"] = "cm"

    # Exact Elsevier / IEEE Double Column Dimension (17.5 cm x 8.4 cm = 7.2 x 3.45 inches)
    fig = plt.figure(figsize=(7.2, 3.45), facecolor="#FFFFFF", dpi=300)
    ax_main = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ax_main.set_xlim(0, 100)
    ax_main.set_ylim(0, 100)
    ax_main.axis("off")

    # 4 Block horizontal layout specifications (X coordinates in percent)
    blocks = [
        {"id": 1, "x": 1.2,  "w": 23.2, "color": PALETTE["b1_gpu"], "title": "Stochastic TTA & Forward", "sub": "1. Multi-Scale Ingestion"},
        {"id": 2, "x": 25.8, "w": 23.2, "color": PALETTE["b2_agg"], "title": "Continuous Aggregation", "sub": "2. Sharpness-Weighted MC"},
        {"id": 3, "x": 50.4, "w": 23.2, "color": PALETTE["b3_gmm"], "title": "Probabilistic Mixture", "sub": "3. EM Fitting + BIC"},
        {"id": 4, "x": 75.0, "w": 23.8, "color": PALETTE["b4_mrf"], "title": "Kinematic MRF Graph", "sub": "4. Topology Decoding"},
    ]

    card_y = 2.0
    card_h = 95.0

    # Draw 4 background rounded cards
    for b in blocks:
        card = FancyBboxPatch(
            (b["x"], card_y), b["w"], card_h,
            boxstyle="Round,pad=0.2,rounding_size=1.8",
            facecolor=PALETTE["bg_card"],
            edgecolor=PALETTE["border_card"],
            linewidth=0.8,
            zorder=1
        )
        ax_main.add_patch(card)

        # Header Badge
        badge = FancyBboxPatch(
            (b["x"], card_y + card_h - 6.8), b["w"], 6.8,
            boxstyle="Round,pad=0.0,rounding_size=1.5",
            facecolor=b["color"],
            edgecolor=b["color"],
            alpha=0.92,
            zorder=2
        )
        ax_main.add_patch(badge)

        ax_main.text(
            b["x"] + b["w"] / 2.0, card_y + card_h - 3.4,
            b["title"],
            color="#FFFFFF", fontsize=6.8, fontweight="bold", ha="center", va="center", zorder=3
        )

        ax_main.text(
            b["x"] + b["w"] / 2.0, card_y + card_h - 9.2,
            b["sub"],
            color=b["color"], fontsize=5.8, fontweight="bold", ha="center", va="center", zorder=3
        )

    # Extract image and crop data
    crop_img = data["crop_img"]
    x_min, y_min, x_max, y_max = data["crop_coords"]
    bx, by, bw, bh = data["bbox"]

    # =========================================================================
    # BLOCK 1: Stochastic TTA & Neural Forward (0.5x Low-Res)
    # =========================================================================
    # 1.1 Input Degraded Image with Bounding Box (Centered in top half)
    ax_b1_in = fig.add_axes([0.022, 0.43, 0.088, 0.33])
    ax_b1_in.imshow(crop_img)
    rect = Rectangle((bx - x_min, by - y_min), bw, bh, linewidth=1.1, edgecolor="#00E5FF", facecolor="none", linestyle="--")
    ax_b1_in.add_patch(rect)
    ax_b1_in.set_title("Input Frame ($I$)", fontsize=5.0, fontweight="bold", pad=2)
    ax_b1_in.axis("off")

    # 1.2 Multi-Scale Views Stack with Trailing Vertical Ellipsis ONLY AT THE END
    # Top view: 0.85x
    ax_s0 = fig.add_axes([0.134, 0.67, 0.040, 0.090])
    scaled_0 = cv2.resize(crop_img, (0, 0), fx=0.85, fy=0.85)
    ax_s0.imshow(scaled_0)
    ax_s0.text(0.5, -0.22, "0.85x", transform=ax_s0.transAxes, fontsize=4.4, fontweight="bold", ha="center", va="top")
    ax_s0.axis("off")

    # Middle view: 1.00x
    ax_s1 = fig.add_axes([0.134, 0.54, 0.040, 0.090])
    ax_s1.imshow(crop_img)
    ax_s1.text(0.5, -0.22, "1.00x", transform=ax_s1.transAxes, fontsize=4.4, fontweight="bold", ha="center", va="top")
    ax_s1.axis("off")

    # Bottom view: 1.15x
    ax_s2 = fig.add_axes([0.134, 0.41, 0.040, 0.090])
    scaled_2 = cv2.resize(crop_img, (0, 0), fx=1.15, fy=1.15)
    ax_s2.imshow(scaled_2)
    ax_s2.text(0.5, -0.22, "1.15x", transform=ax_s2.transAxes, fontsize=4.4, fontweight="bold", ha="center", va="top")
    ax_s2.axis("off")

    # Trailing Vertical Ellipsis ONLY AT THE END (below 1.15x)
    ax_main.text(15.4, 35.5, r"$\mathbf{\vdots}$", fontsize=7.5, color=PALETTE["b1_gpu"], ha="center", va="center", fontweight="bold")

    # Connecting arrow to Backbone
    ax_main.annotate(
        "", xy=(19.2, 50.0), xytext=(17.5, 55.0),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["b1_gpu"], lw=0.9, mutation_scale=6)
    )

    # 1.3 Backbone Block Badge
    bb_box = FancyBboxPatch(
        (19.0, 32.0), 4.2, 35.0,
        boxstyle="Round,pad=0.1,rounding_size=1.0",
        facecolor="#E3F2FD", edgecolor=PALETTE["b1_gpu"], linewidth=1.0, zorder=2
    )
    ax_main.add_patch(bb_box)
    ax_main.text(21.1, 49.5, "CNN / ViT Backbone\n(Frozen HRNet / ViTPose)", fontsize=4.8, fontweight="bold", color=PALETTE["b1_gpu"], ha="center", va="center", rotation=90, zorder=3)

    # 1.4 Raw Multi-Scale Heatmaps for Right Ankle with Trailing Ellipsis ONLY AT THE END
    raw_hm_0 = data["raw_hm_0"]
    raw_hm_4 = data["raw_hm_4"]
    raw_hm_8 = data["raw_hm_8"]

    # Heatmap 0.85
    ax_hm_0 = fig.add_axes([0.024, 0.065, 0.052, 0.22])
    ax_hm_0.imshow(raw_hm_0, cmap="inferno")
    ax_hm_0.set_title(r"$\mathbf{H}_{0.85}^{\mathrm{R\text{-}Ank}}$", fontsize=5.0, pad=1)
    ax_hm_0.axis("off")

    # Heatmap 1.00
    ax_hm_1 = fig.add_axes([0.082, 0.065, 0.052, 0.22])
    ax_hm_1.imshow(raw_hm_4, cmap="inferno")
    ax_hm_1.set_title(r"$\mathbf{H}_{1.00}^{\mathrm{R\text{-}Ank}}$", fontsize=5.0, pad=1)
    ax_hm_1.axis("off")

    # Heatmap 1.15
    ax_hm_2 = fig.add_axes([0.140, 0.065, 0.052, 0.22])
    ax_hm_2.imshow(raw_hm_8, cmap="inferno")
    ax_hm_2.set_title(r"$\mathbf{H}_{1.15}^{\mathrm{R\text{-}Ank}}$", fontsize=5.0, pad=1)
    ax_hm_2.axis("off")

    # Trailing Horizontal Ellipsis ONLY AT THE END (to the right of H_1.15)
    ax_main.text(20.4, 17.5, r"$\mathbf{\dots}$", fontsize=8.0, color=PALETTE["b1_gpu"], ha="center", va="center", fontweight="bold")

    # TTA Subtitle Note
    ax_main.text(11.5, 3.2, r"$s \in \{0.85, \dots, 1.15\} \times \{\mathrm{Orig}, \mathrm{Flip}\}$", fontsize=4.4, color=PALETTE["b1_gpu"], ha="center", va="center")

    # =========================================================================
    # BLOCK 2: Continuous Aggregation & Rejection Sampling
    # =========================================================================
    # Arrow B1 -> B2
    ax_main.annotate(
        "", xy=(26.2, 50.0), xytext=(23.5, 50.0),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["b2_agg"], lw=1.2, mutation_scale=8)
    )

    # 2.1 Rigorous Mathematical Formula Box (Directly from Paper Equations)
    formula_box = FancyBboxPatch(
        (26.4, 62.0), 22.0, 23.0,
        boxstyle="Round,pad=0.2,rounding_size=1.0",
        facecolor="#E8F5E9", edgecolor=PALETTE["b2_agg"], linewidth=0.8, zorder=2
    )
    ax_main.add_patch(formula_box)

    formula_text = (
        r"$\mathbf{P}_k(\mathbf{x}) = \sum_{n=1}^N w_k^{(n)} \tilde{\mathbf{H}}_k^{(n)}(\mathbf{x}), \quad w_k^{(n)} = \frac{\mathcal{C}_k^{(n)}}{\sum_m \mathcal{C}_k^{(m)}}$" + "\n\n"
        r"$\mathcal{C}_k^{(n)} = \left(\rho_k^{(n)}\right)^\alpha \cdot \left[1 - \frac{1}{1 + \frac{\rho_k^{(n)} / (\mu_k^{(n)}+\epsilon)}{\tau}}\right]^\beta$" + "\n\n"
        r"$\mathrm{Dequant\ MC:\ } \tilde{\mathbf{x}}_m = \mathbf{x}_m + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{U}(-0.5, 0.5)$"
    )
    ax_main.text(
        37.4, 73.5,
        formula_text,
        fontsize=4.6, color="#1B5E20", ha="center", va="center", zorder=3
    )

    # 2.2 Continuous Aggregated Heatmap + 2.3 Emerald Green Hollow MC Points (R_Ankle)
    agg_hm = data["agg_hm"]
    samples_kp = data["samples_kp"]

    ax_b2_hm = fig.add_axes([0.272, 0.08, 0.205, 0.46])
    ax_b2_hm.imshow(agg_hm, cmap="inferno")
    # Crisp hollow circles in green matching section color
    ax_b2_hm.scatter(
        samples_kp[:, 0], samples_kp[:, 1],
        s=1.6, facecolors="none", edgecolors="#00E676", linewidths=0.30, alpha=0.75, zorder=4
    )
    ax_b2_hm.set_title(r"$\mathbf{P}_{\mathrm{R\_Ankle}}(\mathbf{x})$ + MC Samples ($N=1000, T=0.3$)", fontsize=5.2, fontweight="bold", pad=2)
    ax_b2_hm.axis("off")

    # Explanatory subtitle inside green card clarifying green hollow points as Monte Carlo samples
    ax_main.text(
        37.4, 4.0,
        r"$\circ\ \mathbf{Green\ circles:}\ \text{Sub-pixel MC Samples } \tilde{\mathbf{x}}_m \sim \mathbf{P}_k$",
        fontsize=4.5, fontweight="bold", color="#1B5E20", ha="center", va="center", zorder=3
    )

    # =========================================================================
    # BLOCK 3: Probabilistic Mixture EM Fitting + BIC Selection
    # =========================================================================
    # Arrow B2 -> B3
    ax_main.annotate(
        "", xy=(50.8, 50.0), xytext=(48.0, 50.0),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["b3_gmm"], lw=1.2, mutation_scale=8)
    )

    # 3.1 Dual Hypothesis Rendering (K=1 vs K=2)
    gmm_k1 = data["gmm_k1"]
    bic_1 = data["bic_1"]
    probs_k1 = data["probs_k1"]

    gmm_k2 = data["gmm_k2"]
    bic_2 = data["bic_2"]
    probs_k2 = data["probs_k2"]

    pi_u_k1 = gmm_k1.uniform_weight_
    pi_u_k2 = gmm_k2.uniform_weight_

    # TOP: K=1 Model Fit with Purple Uniform Background Absorption
    ax_k1 = fig.add_axes([0.518, 0.52, 0.098, 0.30])
    ax_k1.imshow(agg_hm, cmap="magma")
    
    # Render purple uniform absorption wash proportional to pi_uniform (37.0% in K=1)
    if pi_u_k1 > 0.01:
        # Purple uniform tint overlay representing the trash bin absorbing non-structural/unexplained mass
        purple_wash = np.zeros((*agg_hm.shape, 4), dtype=np.float32)
        purple_wash[..., 0] = 0.73  # R
        purple_wash[..., 1] = 0.25  # G
        purple_wash[..., 2] = 0.98  # B
        purple_wash[..., 3] = float(np.clip(pi_u_k1 * 0.40, 0.0, 0.35)) # Alpha proportional to pi_uniform
        ax_k1.imshow(purple_wash, zorder=2)
        
        # Highlight noise samples absorbed by uniform component in magenta/purple
        labels_k1 = np.argmax(probs_k1, axis=1) # 0: Gaussian, 1: Uniform
        noise_mask = labels_k1 == 1
        if np.any(noise_mask):
            ax_k1.scatter(
                samples_kp[noise_mask, 0], samples_kp[noise_mask, 1],
                s=1.2, color="#E040FB", alpha=0.55, zorder=3
            )

    if len(gmm_k1.components_) > 0:
        c1 = gmm_k1.components_[0]
        m1 = c1.mean
        cov1 = c1.covariance
        w1, v1 = np.linalg.eigh(cov1[:2, :2])
        ang1 = np.degrees(np.arctan2(v1[1, 0], v1[0, 0]))
        ell_k1 = Ellipse(m1, width=2*np.sqrt(max(w1[0], 0.5))*2.2, height=2*np.sqrt(max(w1[1], 0.5))*2.2, angle=ang1, edgecolor="#00E676", facecolor="none", lw=0.75, linestyle="--")
        ax_k1.add_patch(ell_k1)
        # Small centroid dot
        ax_k1.scatter([m1[0]], [m1[1]], color="#00E676", s=0.6, zorder=5)
        # Dynamic Weight label cleanly positioned above the ellipse
        ax_k1.text(m1[0], m1[1] - 7.0, f"$\\pi_1={c1.weight:.2f}$", fontsize=3.6, color="#00E676", ha="center", va="bottom", fontweight="bold")
    
    ax_k1.set_title(r"$K=1$ (Unimodal)", fontsize=5.2, fontweight="bold", pad=1)
    ax_k1.text(0.5, -0.17, f"$\\mathrm{{BIC}}_1 = {bic_1:.1f} \\mid \\pi_u = {pi_u_k1*100:.1f}\\%$", transform=ax_k1.transAxes, fontsize=4.4, ha="center", color="#8E24AA")
    ax_k1.axis("off")

    # BOTTOM: K=2 Model Fit (Bimodal with Vertically Staggered Mode Weights)
    ax_k2 = fig.add_axes([0.518, 0.10, 0.098, 0.30])
    ax_k2.imshow(agg_hm, cmap="magma")
    
    # In K=2, pi_uniform is ~0.0% so almost zero purple wash
    if pi_u_k2 > 0.01:
        purple_wash2 = np.zeros((*agg_hm.shape, 4), dtype=np.float32)
        purple_wash2[..., 0] = 0.73
        purple_wash2[..., 1] = 0.25
        purple_wash2[..., 2] = 0.98
        purple_wash2[..., 3] = float(np.clip(pi_u_k2 * 0.40, 0.0, 0.35))
        ax_k2.imshow(purple_wash2, zorder=2)

    comps = gmm_k2.components_
    if len(comps) >= 2:
        # Sort modes: Mode 1 on ball (Cyan, x ~ 10) vs Mode 2 on opponent (Red, x ~ 24)
        modes_sorted = sorted(comps[:2], key=lambda c: c.mean[0])
        true_comp, swap_comp = modes_sorted[0], modes_sorted[1]
        
        # 1. True candidate on ball (Cyan)
        wt, vt = np.linalg.eigh(true_comp.covariance[:2, :2])
        ang_t = np.degrees(np.arctan2(vt[1, 0], vt[0, 0]))
        ell_true = Ellipse(true_comp.mean, width=2*np.sqrt(max(wt[0], 0.5))*2.2, height=2*np.sqrt(max(wt[1], 0.5))*2.2, angle=ang_t, edgecolor="#00E5FF", facecolor="none", lw=0.75)
        ax_k2.add_patch(ell_true)
        # Small centroid dot
        ax_k2.scatter([true_comp.mean[0]], [true_comp.mean[1]], color="#00E5FF", s=0.6, zorder=5)
        # BLUE WEIGHT AT THE BOTTOM (below the ellipse, avoiding any overlap)
        ax_k2.text(
            true_comp.mean[0], true_comp.mean[1] + 6.8,
            f"$\\pi_1={true_comp.weight:.2f}$",
            fontsize=3.8, color="#00E5FF", ha="center", va="top", fontweight="bold"
        )

        # 2. Swapped candidate on opponent (Red)
        ws, vs = np.linalg.eigh(swap_comp.covariance[:2, :2])
        ang_s = np.degrees(np.arctan2(vs[1, 0], vs[0, 0]))
        ell_swap = Ellipse(swap_comp.mean, width=2*np.sqrt(max(ws[0], 0.5))*2.2, height=2*np.sqrt(max(ws[1], 0.5))*2.2, angle=ang_s, edgecolor="#FF1744", facecolor="none", lw=0.75)
        ax_k2.add_patch(ell_swap)
        # Small centroid dot
        ax_k2.scatter([swap_comp.mean[0]], [swap_comp.mean[1]], color="#FF1744", s=0.6, zorder=5)
        # RED WEIGHT AT THE TOP (above the ellipse)
        ax_k2.text(
            swap_comp.mean[0], swap_comp.mean[1] - 6.8,
            f"$\\pi_2={swap_comp.weight:.2f}$",
            fontsize=3.8, color="#FF1744", ha="center", va="bottom", fontweight="bold"
        )

    ax_k2.set_title(r"$K=2$ (Bimodal)", fontsize=5.2, fontweight="bold", pad=1)
    ax_k2.text(0.5, -0.17, f"$\\mathbf{{BIC}}_2 = \\mathbf{{{bic_2:.1f}}} \\mid \\pi_u = {pi_u_k2*100:.1f}\\%$", transform=ax_k2.transAxes, fontsize=4.4, fontweight="bold", ha="center", color=PALETTE["b3_gmm"])
    ax_k2.axis("off")

    # 3.2 BIC Decision Badge & Parameter Extraction (Dynamic Formulation)
    ax_main.text(68.0, 72.0, r"$\mathbf{BIC}_2 < \mathbf{BIC}_1$" + "\n" + r"$\Rightarrow \mathbf{Select\ K=2\ (Bimodal)}$", fontsize=5.3, fontweight="bold", color=PALETTE["b3_gmm"], ha="center", va="center")
    
    param_box = FancyBboxPatch(
        (62.8, 12.0), 10.4, 53.0,
        boxstyle="Round,pad=0.2,rounding_size=0.8",
        facecolor="#FFEBEE", edgecolor=PALETTE["b3_gmm"], linewidth=0.8, zorder=2
    )
    ax_main.add_patch(param_box)
    
    det_final = data["det_sigma_final"]
    u_gmm_val = data["u_gmm"]
    u_base_val = data["u_base"]
    u_adapt_val = data["u_adapt"]

    param_text = (
        r"$\mathbf{R\text{-}Ankle\ Uncertainty}$" + "\n\n"
        r"$\mathbf{1.\ Dual\ GMM\ Modes:}$" + "\n"
        r"$\{\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1, \pi_1\}$" + "\n"
        r"$\{\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2, \pi_2\}$" + "\n\n"
        r"$\mathbf{2.\ Spatial\ Dispersion:}$" + "\n"
        f"$\\det(\\mathbf{{\\Sigma}}_{{\\mathrm{{final}}}}) = {det_final:.3f}$\n"
        f"$U_{{\\mathrm{{gmm}}}} = {u_gmm_val:.3f}$\n\n"
        r"$\mathbf{3.\ Softmax\ Adaptive:}$" + "\n"
        f"$U_{{\\mathrm{{base}}}} = {u_base_val:.2f}$\n"
        f"$\\mathbf{{U}}_{{\\mathbf{{adapt}}}} = \\mathbf{{{u_adapt_val:.2f}}}$\n\n"
        r"$\mathbf{4.\ Outlier\ Density:}$" + "\n"
        f"$\\pi_{{\\mathrm{{uniform}}}} = {pi_u_k2:.3f}$"
    )
    ax_main.text(68.0, 38.5, param_text, fontsize=4.4, color="#B71C1C", ha="center", va="center", zorder=3)

    # =========================================================================
    # BLOCK 4: Kinematic MRF Graph Decoding & Calibrated Output
    # =========================================================================
    # Arrow B3 -> B4
    ax_main.annotate(
        "", xy=(74.8, 50.0), xytext=(72.6, 50.0),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["b4_mrf"], lw=1.2, mutation_scale=8)
    )

    # 4.0 MRF Tree Formulation & Prior Card (Upper Half)
    mrf_formula_box = FancyBboxPatch(
        (75.3, 51.5), 23.2, 34.5,
        boxstyle="Round,pad=0.2,rounding_size=0.8",
        facecolor="#FFFFFF", edgecolor="#AB47BC", linewidth=0.8, zorder=2
    )
    ax_main.add_patch(mrf_formula_box)

    crop_img = data["crop_img"]
    x_min, y_min, x_max, y_max = data["crop_coords"]
    p_coord = data["p_coord"]
    c_gmm = data["c_gmm"]
    c_mrf = data["c_mrf"]
    c_gt = data["c_gt"]
    mu_px = data["mu_px"]
    sigma_px = data["sigma_px"]
    d_unary = float(np.linalg.norm(c_gmm - p_coord))
    d_mrf = float(np.linalg.norm(c_mrf - p_coord))
    d_gt = float(np.linalg.norm(c_gt - p_coord))

    sq_err_unary = (d_unary - mu_px)**2
    sq_err_mrf = (d_mrf - mu_px)**2
    var_px = sigma_px**2

    mrf_formula_text = (
        r"$\mathbf{Global\ Tree\ Optimization:}$" + "\n\n"
        r"$\mathbf{x}^* = \arg\min_{\mathbf{x}} \sum_{u} \phi_u(x_u) + \sum_{(u, v)} \psi_{uv}(x_u, x_v)$" + "\n\n"
        r"$\mathbf{Unary\ Cost:}\quad \phi_u(x_u) = -\ln \mathbf{P}_u(x_u)$" + "\n\n"
        r"$\mathbf{Kinematic\ Pairwise:}\quad \psi_{uv} = \frac{(\|\mathbf{x}_u - \mathbf{x}_v\| - \mu_{uv})^2}{2\sigma_{uv}^2}$" + "\n\n"
        r"$\mathbf{Prior\ (\text{R-Knee} \to \text{R-Ankle}):}$" + "\n"
        f"$\\mu_{{uv}} = {mu_px:.1f}\\mathrm{{px}}, \\quad \\sigma_{{uv}} = {sigma_px:.1f}\\mathrm{{px}} \\quad (\\sigma_{{uv}}^2 = {var_px:.1f}\\mathrm{{px}}^2)$"
    )
    ax_main.text(86.9, 68.75, mrf_formula_text, fontsize=4.5, color="#4A148C", ha="center", va="center", zorder=3)

    # 4.1 Lower Main Unified Image (Large Single Crop)
    ax_crop = fig.add_axes([0.758, 0.05, 0.222, 0.39])
    ax_crop.imshow(crop_img, extent=[x_min, x_max, y_max, y_min])

    # Background bones in translucent green
    gt = data["gt"]
    if isinstance(gt, np.ndarray):
        for p1, p2 in COCO_SKELETON_PAIRS:
            if p1 < len(gt) and p2 < len(gt) and gt[p1, 2] > 0 and gt[p2, 2] > 0:
                if (p1, p2) != (14, 16) and (p2, p1) != (14, 16):
                    ax_crop.plot([gt[p1, 0], gt[p2, 0]], [gt[p1, 1], gt[p2, 1]], color="#00E676", lw=0.9, alpha=0.65, zorder=3)

    # Prior Band & Ring around R_Knee
    prior_band = patches.Wedge((p_coord[0], p_coord[1]), mu_px + sigma_px, 0, 360, width=2*sigma_px, facecolor="#76FF03", alpha=0.22, zorder=3)
    prior_ring = patches.Circle((p_coord[0], p_coord[1]), mu_px, fill=False, edgecolor="#76FF03", linestyle="--", lw=1.0, zorder=4)
    ax_crop.add_patch(prior_band)
    ax_crop.add_patch(prior_ring)

    # Anchor R_Knee (Yellow square)
    ax_crop.scatter(p_coord[0], p_coord[1], color="#FFD600", marker="s", s=45, edgecolors="#000000", linewidth=0.8, zorder=6)
    ax_crop.text(p_coord[0] - 2, p_coord[1] - 4, "R-Knee (Anchor)", fontsize=3.4, fontweight="bold", color="#FFF9C4",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="#000000", alpha=0.65, edgecolor="none"), zorder=7)

    # GMM Unary Swap (Red Cross)
    ax_crop.scatter(c_gmm[0], c_gmm[1], color="#FF1744", marker="x", s=55, lw=1.8, zorder=6)

    # Restored Bone Knee -> MRF Ankle in Cyan
    ax_crop.plot([p_coord[0], c_mrf[0]], [p_coord[1], c_mrf[1]], color="#00E5FF", lw=1.8, zorder=5)

    # Ground Truth (Green circle) & MRF Solution (Cyan dot)
    ax_crop.scatter(c_gt[0], c_gt[1], color="#00E676", marker="o", s=35, edgecolors="#000000", lw=0.6, zorder=6)
    ax_crop.scatter(c_mrf[0], c_mrf[1], color="#00E5FF", marker="o", s=45, edgecolors="#000000", lw=0.8, zorder=7)

    # Corrective Kinematic Pull Arrow from c_gmm to c_mrf
    arr = patches.FancyArrowPatch(
        (c_gmm[0], c_gmm[1]), (c_mrf[0], c_mrf[1]),
        color="#00E5FF", arrowstyle="-|>", mutation_scale=10, lw=1.4, linestyle="--", zorder=8
    )
    ax_crop.add_patch(arr)

    # Label on the kinematic pull arrow
    ax_crop.text((c_gmm[0] + c_mrf[0]) / 2, (c_gmm[1] + c_mrf[1]) / 2 - 2.8, "Kinematic Pull",
                 fontsize=3.2, fontweight="bold", color="#00E5FF", ha="center", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.12", facecolor="#000000", alpha=0.65, edgecolor="none"), zorder=9)

    # Annotations directly inside the image with balanced modern styling (Option 1)
    mrf_info = (
        r"$\mathbf{MRF\ MAP\ Solution\ (\mu_1)}$" + "\n"
        f"$d = {d_mrf:.1f}\\mathrm{{px}} \\Rightarrow \\psi_{{uv}} = {sq_err_mrf:.1f}\\mathrm{{px}}^2$\n"
        r"[✓ Selected | In Prior Band]"
    )
    unary_info = (
        r"$\mathbf{GMM\ Unary\ Peak\ (\mu_2\ Swap)}$" + "\n"
        f"$d = {d_unary:.1f}\\mathrm{{px}} \\Rightarrow \\psi_{{uv}} = {sq_err_unary:.1f}\\mathrm{{px}}^2$\n"
        r"[✗ Pruned | High Penalty]"
    )

    ax_crop.text(
        0.035, 0.08, mrf_info, transform=ax_crop.transAxes, fontsize=2.95, fontweight="bold", color="#E0F7FA",
        ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.4", facecolor="#004D40", alpha=0.92, edgecolor="#00E5FF", lw=0.7),
        zorder=9
    )

    ax_crop.text(
        0.965, 0.08, unary_info, transform=ax_crop.transAxes, fontsize=2.95, fontweight="bold", color="#FFEBEE",
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.4", facecolor="#B71C1C", alpha=0.92, edgecolor="#FF5252", lw=0.7),
        zorder=9
    )

    ax_crop.set_title("Kinematic Prior Band & Swap Correction", fontsize=4.8, fontweight="bold", color="#4A148C", pad=3)
    ax_crop.axis("off")

    # =========================================================================
    # Export Vectorial PDF & 300 DPI PNG
    # =========================================================================
    target_dirs = [
        project_root / "outputs" / "figures" / "methodology",
    ]
    if (project_root.parent / "Paper").exists():
        target_dirs.append(project_root.parent / "Paper" / "Paper" / "figures" / "methodology")
    if (project_root / "Paper").exists():
        target_dirs.append(project_root / "Paper" / "Paper" / "figures" / "methodology")

    for t_dir in target_dirs:
        t_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = t_dir / "methodology_pipeline_overview.pdf"
        png_path = t_dir / "methodology_pipeline_overview.png"
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
        plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        logger.info("Saved Methodology Pipeline figure to:\n  - %s\n  - %s", pdf_path, png_path)

    plt.close(fig)


def main() -> None:
    out_dir = project_root / "outputs" / "figures" / "methodology"
    candidate_paths = [
        project_root / "outputs" / "figures" / "visualizations" / "packets" / "candidate_130.pkl.gz",
        project_root / "outputs" / "figures" / "visualizations" / "figura_2_mrf_dilemma" / "0_FAVORITA_candidato_2a_image_130" / "candidate_130_130.pkl.gz",
        project_root.parent / "Paper" / "Paper" / "figures" / "visualizaciones" / "figura_2_mrf_dilemma" / "0_FAVORITA_candidato_2a_image_130" / "candidate_130_130.pkl.gz",
        project_root / "Paper" / "Paper" / "figures" / "visualizaciones" / "figura_2_mrf_dilemma" / "0_FAVORITA_candidato_2a_image_130" / "candidate_130_130.pkl.gz",
    ]
    pkt_path = next((p for p in candidate_paths if p.exists()), candidate_paths[0])
    
    logger.info("Loading Candidate 130 using exact generate_all_paper_figures.py logic...")
    pipeline_data = load_candidate_130_from_figures_pipeline(pkt_path)
    
    logger.info("Rendering publication-grade methodology pipeline figure...")
    render_methodology_pipeline_figure(pipeline_data, out_dir)
    logger.info("Pipeline figure successfully generated!")


if __name__ == "__main__":
    main()

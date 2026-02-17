"""
FiftyOne Loader – **Grouped Dataset** for deep visual analysis of HPE.

Creates a FiftyOne Grouped Dataset where each analysis packet (``.pkl.gz``)
becomes a *group* containing several slices:

* ``original``          – source image with GT / Baseline / GMM-Ours keypoints
                          and uncertainty ellipses.
* ``mix_analysis``      – 4×5 Matplotlib grid of the **averaged** per-keypoint
                          heatmaps.  Monte-Carlo samples are stored as
                          ``fo.Keypoints`` and GMM confidence ellipses as
                          ``fo.Polylines``, so they can be **toggled
                          independently** in the FiftyOne sidebar.
* ``tta_{i}_image``     – the augmented image produced by TTA transform *i*.
* ``tta_{i}_heatmaps``  – 4×5 Matplotlib grid of that transform's per-keypoint
                          predicted heatmaps (no statistical overlay).

Usage
-----
>>> from pose_uncertainty.visualization.fiftyone_loader import create_evaluation_dataset
>>> ds = create_evaluation_dataset("outputs/2026-02-11/12-40-42")
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_SUBDIR = "fiftyone_cache"
_ELLIPSE_PTS = 64           # vertices per confidence ellipse
_N_COLS = 4                  # columns in the heatmap grid
_GRID_DPI = 100
_CELL_INCHES = (3.0, 3.0)   # (width, height) per subplot cell
_MAX_SAMPLES_VIS = 200       # MC samples shown per keypoint (subsampled)

COCO_KP_NAMES: List[str] = [
    "Nose",       "L_Eye",      "R_Eye",      "L_Ear",       "R_Ear",
    "L_Shoulder", "R_Shoulder", "L_Elbow",    "R_Elbow",
    "L_Wrist",    "R_Wrist",    "L_Hip",      "R_Hip",
    "L_Knee",     "R_Knee",     "L_Ankle",    "R_Ankle",
]
_N_KP = len(COCO_KP_NAMES)  # 17


# ===================================================================
# Public API
# ===================================================================

def create_evaluation_dataset(
    data_dir: str,
    dataset_name: str = "TFG_Evaluation",
) -> Any:
    """Build a **grouped** FiftyOne dataset from Deep-Profiling packets.

    Parameters
    ----------
    data_dir : str
        Folder containing ``.pkl.gz`` packets produced by
        ``EvaluationRunner.run_deep_profiling``.
    dataset_name : str
        Name shown in the FiftyOne app (overwritten if it already exists).

    Returns
    -------
    fiftyone.Dataset
    """
    import fiftyone as fo

    data_path = Path(data_dir)
    pkl_files = sorted(data_path.glob("*.pkl.gz"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl.gz files in {data_path}")
    logger.info("Found %d packets in %s", len(pkl_files), data_path)

    from ..evaluation.storage import load_deep_analysis

    cache_dir = data_path / _CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- create / overwrite grouped dataset --------------------------------
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(name=dataset_name)
    dataset.persistent = True
    dataset.add_group_field("group", default="original")

    # ---- iterate packets ---------------------------------------------------
    n_ok = 0
    for pkl_path in pkl_files:
        try:
            packet = load_deep_analysis(str(pkl_path))
            samples = _packet_to_samples(packet, cache_dir, fo)
            if samples:
                dataset.add_samples(samples)
                n_ok += 1
        except Exception:
            logger.exception("Failed to process %s", pkl_path.name)

    logger.info(
        "Dataset '%s': %d groups, %d total samples",
        dataset_name, n_ok, len(dataset),
    )

    # ---- launch app --------------------------------------------------------
    session = fo.launch_app(dataset, address="0.0.0.0", remote=True)
    print("Servidor activo. Presiona Ctrl+C para salir.")
    session.wait()
    return dataset


# ===================================================================
# Packet -> list of grouped fo.Sample
# ===================================================================

def _packet_to_samples(
    packet: Dict[str, Any],
    cache_dir: Path,
    fo: Any,
) -> List[Any]:
    """Build every group slice for a single analysis packet."""

    meta = packet.get("meta", {})
    image_id = meta.get("image_id", 0)
    sample_type = meta.get("sample_type", "unknown")
    tta_data = packet.get("tta_data", [])

    # ---- recover original image -------------------------------------------
    image_array = _find_original_image(tta_data)
    if image_array is None:
        logger.warning("No image for id=%s; skipping.", image_id)
        return []

    prefix = f"{sample_type}_{image_id}"

    # Shared scalar metrics attached to every slice
    metrics = packet.get("metrics", {})
    scalars: Dict[str, Any] = {
        "image_id":          int(image_id),
        "sample_type":       str(sample_type),
        "oks_delta":         float(metrics.get("delta_oks", 0.0)),
        "nll":               float(metrics.get("nll", 0.0)),
        "entropy":           float(metrics.get("entropy", 0.0)),
        "oks_base":          float(metrics.get("oks_base", 0.0)),
        "oks_ours":          float(metrics.get("oks_ours", 0.0)),
        "covariance_volume": float(metrics.get("covariance_volume", 0.0)),
    }

    group = fo.Group()
    samples: List[Any] = []

    # 1. original -----------------------------------------------------------
    samples.append(
        _build_original_slice(
            packet, image_array, prefix, cache_dir, fo, group, scalars,
        )
    )

    # 2. mix_analysis -------------------------------------------------------
    mix = _build_mix_analysis_slice(
        packet, prefix, cache_dir, fo, group, scalars,
    )
    if mix is not None:
        samples.append(mix)

    # 3 & 4. per-TTA slices -------------------------------------------------
    for idx, entry in enumerate(tta_data):
        name = entry.get("name", f"tta_{idx}")

        s = _build_tta_image_slice(
            entry, idx, name, prefix, cache_dir, fo, group, scalars,
        )
        if s is not None:
            samples.append(s)

        s = _build_tta_heatmap_slice(
            entry, idx, name, prefix, cache_dir, fo, group, scalars,
        )
        if s is not None:
            samples.append(s)

    return samples


# ===================================================================
# Slice builders
# ===================================================================

def _build_original_slice(
    packet: Dict,
    image_array: npt.NDArray,
    prefix: str,
    cache_dir: Path,
    fo: Any,
    group: Any,
    scalars: Dict[str, Any],
) -> Any:
    """Slice ``'original'``: source image + keypoints + ellipses."""

    img_path = cache_dir / f"{prefix}_original.jpg"
    cv2.imwrite(str(img_path), cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

    h, w = image_array.shape[:2]
    sample = fo.Sample(filepath=str(img_path))
    sample["group"] = group.element("original")
    _apply_scalars(sample, scalars)

    # GT coords with visibility (used for filtering)
    gt = _nested_array(packet, "ground_truth", "coords")  # (N, 3): x, y, vis

    # GT keypoints – labelled per body part + visibility
    if gt is not None:
        sample["ground_truth"] = _to_fo_keypoints_labeled(
            gt, w, h, fo, gt_coords_with_vis=gt,
        )

    # Baseline prediction – labelled per body part + GT visibility
    base = _nested_array(packet, "predictions", "baseline", "coords")
    if base is not None:
        sample["prediction_base"] = _to_fo_keypoints_labeled(
            base, w, h, fo, gt_coords_with_vis=gt,
        )

    # Our GMM prediction – labelled per body part + GT visibility
    ours = _nested_array(packet, "predictions", "ours_gmm", "coords")
    if ours is not None:
        sample["prediction_ours"] = _to_fo_keypoints_labeled(
            ours, w, h, fo, gt_coords_with_vis=gt,
        )

    # Uncertainty ellipses – labelled per keypoint (+ visibility tags)
    gmm_per_kp = packet.get("gmm_per_kp")
    if isinstance(gmm_per_kp, list) and len(gmm_per_kp) > 0:
        ell = _ellipses_per_kp_labeled(
            gmm_per_kp, w, h, packet, fo,
            gt_coords_with_vis=gt,
        )
        if ell is not None:
            sample["uncertainty_ellipses"] = ell
    else:
        # Fallback: legacy concatenated GMM (unlabeled)
        gmm = packet.get("gmm_model", {})
        means, covs = gmm.get("means"), gmm.get("covariances")
        if means is not None and covs is not None:
            means_img, covs_img = _map_gmm_to_image_space(
                packet,
                np.asarray(means),
                np.asarray(covs),
                w,
                h,
            )
            polys = _ellipses_normalised(
                means_img, covs_img, w, h, fo,
            )
            if polys:
                sample["uncertainty_ellipses"] = fo.Polylines(polylines=polys)

    return sample


def _build_mix_analysis_slice(
    packet: Dict,
    prefix: str,
    cache_dir: Path,
    fo: Any,
    group: Any,
    scalars: Dict[str, Any],
) -> Optional[Any]:
    """Slice ``'mix_analysis'``: averaged heatmap grid + toggleable overlays."""

    agg = packet.get("aggregation", {})
    avg_hm = agg.get("heatmap_avg")  # expected (17, H, W)
    if avg_hm is None or not isinstance(avg_hm, np.ndarray) or avg_hm.ndim != 3:
        return None

    n_kp = min(avg_hm.shape[0], _N_KP)
    hm_h, hm_w = avg_hm.shape[1], avg_hm.shape[2]
    titles = COCO_KP_NAMES[:n_kp]

    # ---- render 4×5 heatmap grid ------------------------------------------
    grid_rgb, rects = _render_heatmap_grid(avg_hm[:n_kp], titles)
    grid_path = cache_dir / f"{prefix}_mix_analysis.jpg"
    cv2.imwrite(
        str(grid_path),
        cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    grid_h, grid_w = grid_rgb.shape[:2]

    sample = fo.Sample(filepath=str(grid_path))
    sample["group"] = group.element("mix_analysis")
    _apply_scalars(sample, scalars)

    # ---- MC samples --> fo.Keypoints (toggleable) --------------------------
    samp_by_kp = agg.get("sampling_points_by_kp")
    kp_pts: List[List[Tuple[float, float]]] = []
    if isinstance(samp_by_kp, list) and len(samp_by_kp) > 0:
        kp_pts = _samples_by_kp_to_grid_keypoints(
            samp_by_kp, n_kp, hm_w, hm_h, rects, grid_w, grid_h,
        )
    else:
        samp = agg.get("sampling_points")  # legacy: (N_total, 2)
        if (
            samp is not None
            and isinstance(samp, np.ndarray)
            and samp.ndim == 2
            and samp.shape[0] > 0
        ):
            kp_pts = _samples_to_grid_keypoints(
                samp, n_kp, hm_w, hm_h, rects, grid_w, grid_h,
            )

    if kp_pts:
        kp_objs = [
            fo.Keypoint(points=pts, label=COCO_KP_NAMES[k])
            for k, pts in enumerate(kp_pts)
            if pts
        ]
        if kp_objs:
            sample["mc_samples"] = fo.Keypoints(keypoints=kp_objs)

    # ---- GMM ellipses --> fo.Polylines (toggleable) -------------------------
    gmm_per_kp = packet.get("gmm_per_kp")
    if isinstance(gmm_per_kp, list) and len(gmm_per_kp) > 0:
        polys, mode_kps = _gmm_per_kp_to_grid_overlays(
            gmm_per_kp=gmm_per_kp,
            n_kp=n_kp,
            hm_w=hm_w,
            hm_h=hm_h,
            rects=rects,
            grid_w=grid_w,
            grid_h=grid_h,
            fo=fo,
        )
        if polys:
            sample["gmm_ellipses"] = fo.Polylines(polylines=polys)
        if mode_kps:
            sample["gmm_modes"] = fo.Keypoints(keypoints=mode_kps)
    else:
        # Legacy packets: concatenated components only
        gmm = packet.get("gmm_model", {})
        means = gmm.get("means")
        covs = gmm.get("covariances")
        weights = gmm.get("weights")
        if means is not None and covs is not None:
            means_a = np.asarray(means)
            covs_a = np.asarray(covs)
            if means_a.ndim == 2 and covs_a.ndim == 3:
                assignments = _assign_components_to_kp(means_a, avg_hm[:n_kp])
                polys, mode_kps = _legacy_gmm_to_grid_overlays(
                    means=means_a,
                    covs=covs_a,
                    weights=np.asarray(weights) if weights is not None else None,
                    assignments=assignments,
                    n_kp=n_kp,
                    hm_w=hm_w,
                    hm_h=hm_h,
                    rects=rects,
                    grid_w=grid_w,
                    grid_h=grid_h,
                    fo=fo,
                )
                if polys:
                    sample["gmm_ellipses"] = fo.Polylines(polylines=polys)
                if mode_kps:
                    sample["gmm_modes"] = fo.Keypoints(keypoints=mode_kps)

    # ---- GT keypoints on heatmap grid (toggleable) -------------------------
    gt_on_grid = _image_coords_to_grid_keypoints(
        packet, "ground_truth", "coords",
        n_kp=n_kp, hm_w=hm_w, hm_h=hm_h, rects=rects,
        grid_w=grid_w, grid_h=grid_h, fo=fo,
    )
    if gt_on_grid:
        sample["gt_on_heatmap"] = fo.Keypoints(keypoints=gt_on_grid)

    # ---- Baseline prediction on heatmap grid (toggleable) ------------------
    base_on_grid = _image_coords_to_grid_keypoints(
        packet, "predictions", "baseline.coords",
        n_kp=n_kp, hm_w=hm_w, hm_h=hm_h, rects=rects,
        grid_w=grid_w, grid_h=grid_h, fo=fo,
    )
    if base_on_grid:
        sample["baseline_on_heatmap"] = fo.Keypoints(keypoints=base_on_grid)

    return sample


def _build_tta_image_slice(
    entry: Dict,
    idx: int,
    name: str,
    prefix: str,
    cache_dir: Path,
    fo: Any,
    group: Any,
    scalars: Dict[str, Any],
) -> Optional[Any]:
    """Slice ``'tta_{i}_image'``: the augmented image."""

    img = entry.get("image")
    if img is None or not isinstance(img, np.ndarray):
        return None

    tta_label, tta_slug = _tta_label_and_slug(name, entry)
    path = cache_dir / f"{prefix}_tta_{tta_slug}_image.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    sample = fo.Sample(filepath=str(path))
    sample["group"] = group.element(f"tta_{tta_slug}_image")
    sample["tta_index"] = idx
    sample["tta_name"] = name
    sample["tta_label"] = tta_label
    sample["tta_params"] = entry.get("params", {}) if isinstance(entry, dict) else {}
    _apply_scalars(sample, scalars)
    return sample


def _build_tta_heatmap_slice(
    entry: Dict,
    idx: int,
    name: str,
    prefix: str,
    cache_dir: Path,
    fo: Any,
    group: Any,
    scalars: Dict[str, Any],
) -> Optional[Any]:
    """Slice ``'tta_{i}_heatmaps'``: per-keypoint heatmap grid for one TTA."""

    hm = entry.get("heatmap")  # expected (17, H, W)
    if hm is None or not isinstance(hm, np.ndarray) or hm.ndim != 3:
        return None

    n_kp = min(hm.shape[0], _N_KP)
    titles = [f"{COCO_KP_NAMES[k]} [{name}]" for k in range(n_kp)]

    grid_rgb, _ = _render_heatmap_grid(hm[:n_kp], titles)
    tta_label, tta_slug = _tta_label_and_slug(name, entry)
    path = cache_dir / f"{prefix}_tta_{tta_slug}_heatmaps.jpg"
    cv2.imwrite(
        str(path),
        cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    sample = fo.Sample(filepath=str(path))
    sample["group"] = group.element(f"tta_{tta_slug}_heatmaps")
    sample["tta_index"] = idx
    sample["tta_name"] = name
    sample["tta_label"] = tta_label
    sample["tta_params"] = entry.get("params", {}) if isinstance(entry, dict) else {}
    _apply_scalars(sample, scalars)
    return sample


# ===================================================================
# Heatmap grid renderer (Matplotlib)
# ===================================================================

def _render_heatmap_grid(
    heatmaps: npt.NDArray,
    titles: List[str],
    show_axis: bool = False,
) -> Tuple[npt.NDArray, List[Dict[str, float]]]:
    """Render ``(K, H, W)`` heatmaps as a single grid image.

    Parameters
    ----------
    heatmaps : (K, H, W) float array
        Per-keypoint heatmaps.
    titles : list[str]
        One title per subplot (body-part name).
    show_axis : bool
        Whether to display axis ticks.

    Returns
    -------
    rgb : (H_grid, W_grid, 3) uint8 ndarray
        The rendered grid image.
    rects : list[dict]
        Per-subplot data-area rectangles in **grid-image pixel coords**
        (keys: ``x0``, ``y0``, ``w``, ``h``).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = len(heatmaps)
    n_cols = _N_COLS
    n_rows = math.ceil(K / n_cols)

    fw = _CELL_INCHES[0] * n_cols
    fh = _CELL_INCHES[1] * n_rows

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fw, fh),
        dpi=_GRID_DPI,
        squeeze=False,
    )

    for i in range(K):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        ax.imshow(
            heatmaps[i], cmap="hot",
            interpolation="bilinear", aspect="equal",
        )
        ax.set_title(
            titles[i], fontsize=9, fontweight="bold",
            color="white", pad=4,
        )
        if not show_axis:
            ax.set_xticks([])
            ax.set_yticks([])

    # Hide unused cells
    for i in range(K, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r][c].set_visible(False)

    fig.set_facecolor("black")
    fig.tight_layout(pad=0.8)

    # ---- rasterise to numpy -----------------------------------------------
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    w_px, h_px = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h_px, w_px, 4)
    rgb = img_rgba[:, :, :3].copy()

    # ---- extract subplot data-area rects in image-pixel coords ------------
    rects: List[Dict[str, float]] = []
    for i in range(K):
        r, c = divmod(i, n_cols)
        bbox = axes[r][c].get_window_extent(renderer=renderer)
        # bbox uses display coords (origin bottom-left); flip y for image
        rects.append({
            "x0": float(bbox.x0),
            "y0": float(h_px - bbox.y1),      # top of subplot in img coords
            "w":  float(bbox.width),
            "h":  float(bbox.height),
        })

    plt.close(fig)
    return rgb, rects


# ===================================================================
# Coordinate mapping – overlay data onto the grid image
# ===================================================================

def _image_coords_to_grid_keypoints(
    packet: Dict[str, Any],
    *nested_keys: str,
    n_kp: int,
    hm_w: int,
    hm_h: int,
    rects: List[Dict[str, float]],
    grid_w: int,
    grid_h: int,
    fo: Any,
) -> List[Any]:
    """Transform image-space coords to heatmap space and map onto the grid.

    This function navigates the packet with *nested_keys* (supports dotted
    sub-keys, e.g. ``"predictions", "baseline.coords"``), converts the
    resulting ``(N, 2|3)`` array from image coordinates to heatmap
    coordinates using ``MMPoseAdapter.transform_image_coords_to_heatmap``,
    and produces one ``fo.Keypoint`` per body part positioned on the
    corresponding subplot of the heatmap grid image.

    Returns an empty list if any step fails (missing data / metadata).
    """
    # Navigate nested keys, supporting dotted sub-keys
    obj: Any = packet
    for key in nested_keys:
        for sub in key.split("."):
            if not isinstance(obj, dict):
                return []
            obj = obj.get(sub)
            if obj is None:
                return []
    coords_img = obj
    if not isinstance(coords_img, np.ndarray) or coords_img.ndim != 2:
        return []
    coords_img_2d = coords_img[:, :2].astype(np.float32)

    # Also fetch GT visibility (for tags) when available
    gt_raw = _nested_array(packet, "ground_truth", "coords")

    # Build a temporary StandardizedHeatmap to use the adapter's transform
    agg = packet.get("aggregation", {})
    metadata = agg.get("mmpose_metadata") if isinstance(agg, dict) else None
    avg_hm = agg.get("heatmap_avg")
    if metadata is None or not isinstance(avg_hm, np.ndarray) or avg_hm.ndim != 3:
        return []

    try:
        from ..models.adapters import MMPoseAdapter
        from ..utils.types import StandardizedHeatmap

        # We need original image size — derive from GT bbox or use heatmap_avg metadata
        # Use a reasonable fallback; the transform doesn't depend on original_size.
        ref_hm = StandardizedHeatmap(
            data=np.asarray(avg_hm, dtype=np.float32),
            original_size=(1, 1),  # not used by the coordinate transform
            metadata=metadata,
        )
        coords_hm = MMPoseAdapter.transform_image_coords_to_heatmap(
            coords_img_2d, ref_hm,
        )
    except Exception:
        logger.debug(
            "Failed to transform image coords to heatmap space for grid overlay",
            exc_info=True,
        )
        return []

    # Map heatmap coords onto the grid — one fo.Keypoint per keypoint
    kp_objs: List[Any] = []
    n = min(int(coords_hm.shape[0]), n_kp, len(rects))
    for k in range(n):
        x_hm, y_hm = float(coords_hm[k, 0]), float(coords_hm[k, 1])
        rect = rects[k]
        gx = rect["x0"] + (x_hm / hm_w) * rect["w"]
        gy = rect["y0"] + (y_hm / hm_h) * rect["h"]
        nx = float(np.clip(gx / grid_w, 0.0, 1.0))
        ny = float(np.clip(gy / grid_h, 0.0, 1.0))

        kp = fo.Keypoint(
            points=[(nx, ny)],
            label=COCO_KP_NAMES[k],
        )
        # Add visibility info from GT if available
        if gt_raw is not None and k < gt_raw.shape[0] and gt_raw.shape[1] >= 3:
            vis_code = int(gt_raw[k, 2])
            vis_label = _VIS_LABELS.get(vis_code, "unknown")
            kp["visibility"] = vis_label
            kp.tags = [vis_label, COCO_KP_NAMES[k]]
        else:
            kp.tags = [COCO_KP_NAMES[k]]
        kp_objs.append(kp)
    return kp_objs


def _samples_to_grid_keypoints(
    samples_all: npt.NDArray,   # (N_total, 2) in heatmap-pixel coords
    n_kp: int,
    hm_w: int,
    hm_h: int,
    rects: List[Dict[str, float]],
    grid_w: int,
    grid_h: int,
) -> List[List[Tuple[float, float]]]:
    """Split concatenated MC samples per keypoint and map to grid [0,1].

    Assumes equal number of samples per keypoint (the normal case when
    ``sample_from_heatmap`` uses the same ``num_samples`` for every channel).
    """
    n_total = samples_all.shape[0]
    n_per = max(1, n_total // n_kp)
    rng = np.random.default_rng(42)

    result: List[List[Tuple[float, float]]] = []
    for k in range(n_kp):
        start = k * n_per
        end = min(start + n_per, n_total)
        chunk = samples_all[start:end]
        if len(chunk) == 0:
            result.append([])
            continue

        # Subsample for rendering performance
        if len(chunk) > _MAX_SAMPLES_VIS:
            idx = rng.choice(len(chunk), _MAX_SAMPLES_VIS, replace=False)
            chunk = chunk[idx]

        rect = rects[k]
        pts: List[Tuple[float, float]] = []
        for x_hm, y_hm in chunk:
            gx = rect["x0"] + (float(x_hm) / hm_w) * rect["w"]
            gy = rect["y0"] + (float(y_hm) / hm_h) * rect["h"]
            pts.append((
                float(np.clip(gx / grid_w, 0.0, 1.0)),
                float(np.clip(gy / grid_h, 0.0, 1.0)),
            ))
        result.append(pts)

    return result


def _samples_by_kp_to_grid_keypoints(
    samples_by_kp: List[Any],
    n_kp: int,
    hm_w: int,
    hm_h: int,
    rects: List[Dict[str, float]],
    grid_w: int,
    grid_h: int,
) -> List[List[Tuple[float, float]]]:
    """Map per-keypoint MC samples onto the heatmap grid in [0,1]."""
    rng = np.random.default_rng(42)
    result: List[List[Tuple[float, float]]] = []

    for k in range(n_kp):
        if k >= len(rects):
            result.append([])
            continue
        chunk_raw = samples_by_kp[k] if k < len(samples_by_kp) else None
        chunk = np.asarray(chunk_raw) if chunk_raw is not None else np.empty((0, 2))
        if chunk.ndim != 2 or chunk.shape[1] < 2 or chunk.shape[0] == 0:
            result.append([])
            continue

        if len(chunk) > _MAX_SAMPLES_VIS:
            idx = rng.choice(len(chunk), _MAX_SAMPLES_VIS, replace=False)
            chunk = chunk[idx]

        rect = rects[k]
        pts: List[Tuple[float, float]] = []
        for x_hm, y_hm in chunk[:, :2]:
            gx = rect["x0"] + (float(x_hm) / hm_w) * rect["w"]
            gy = rect["y0"] + (float(y_hm) / hm_h) * rect["h"]
            pts.append((
                float(np.clip(gx / grid_w, 0.0, 1.0)),
                float(np.clip(gy / grid_h, 0.0, 1.0)),
            ))
        result.append(pts)

    return result


def _assign_components_to_kp(
    means: npt.NDArray,       # (C_total, 2) in heatmap coords
    avg_hm: npt.NDArray,      # (K, H, W)
) -> npt.NDArray:
    """Assign each GMM component to the keypoint whose heatmap is strongest
    at the component's mean location.

    Returns an int array of shape ``(C_total,)`` with keypoint indices.
    """
    n_kp, hm_h, hm_w = avg_hm.shape
    assignments = np.zeros(len(means), dtype=int)
    for j, mu in enumerate(means):
        x = int(np.clip(round(float(mu[0])), 0, hm_w - 1))
        y = int(np.clip(round(float(mu[1])), 0, hm_h - 1))
        assignments[j] = int(np.argmax(avg_hm[:, y, x]))
    return assignments


def _gmm_to_grid_polylines(
    means: npt.NDArray,
    covs: npt.NDArray,
    assignments: npt.NDArray,
    n_kp: int,
    hm_w: int,
    hm_h: int,
    rects: List[Dict[str, float]],
    grid_w: int,
    grid_h: int,
    fo: Any,
) -> List[Any]:
    """Map concatenated GMM ellipses onto the grid as ``fo.Polyline``."""
    polylines: List[Any] = []

    for j in range(len(means)):
        k = int(assignments[j])
        if k >= n_kp or k >= len(rects):
            continue

        verts = _ellipse_vertices(means[j], covs[j])
        if len(verts) == 0:
            continue

        rect = rects[k]
        mapped: List[Tuple[float, float]] = []
        for x_hm, y_hm in verts:
            gx = rect["x0"] + (float(x_hm) / hm_w) * rect["w"]
            gy = rect["y0"] + (float(y_hm) / hm_h) * rect["h"]
            mapped.append((
                float(np.clip(gx / grid_w, 0.0, 1.0)),
                float(np.clip(gy / grid_h, 0.0, 1.0)),
            ))
        polylines.append(
            fo.Polyline(
                points=[mapped],
                closed=True,
                filled=False,
                label=COCO_KP_NAMES[k],
            )
        )
    return polylines


def _legacy_gmm_to_grid_overlays(
    means: npt.NDArray,
    covs: npt.NDArray,
    weights: Optional[npt.NDArray],
    assignments: npt.NDArray,
    n_kp: int,
    hm_w: int,
    hm_h: int,
    rects: List[Dict[str, float]],
    grid_w: int,
    grid_h: int,
    fo: Any,
) -> Tuple[List[Any], List[Any]]:
    """Legacy overlay builder from concatenated GMM arrays.

    Keeps at most 2 ellipses per keypoint and computes one mode per keypoint.
    """
    polylines: List[Any] = []
    mode_keypoints: List[Any] = []

    w_all = np.asarray(weights, dtype=np.float64).reshape(-1) if weights is not None else None

    for k in range(min(n_kp, len(rects))):
        idx = np.where(assignments == k)[0]
        if idx.size == 0:
            continue

        m = means[idx, :2]
        c = covs[idx, :2, :2]
        if w_all is not None and w_all.shape[0] >= np.max(idx) + 1:
            w = np.clip(w_all[idx], 0.0, None)
        else:
            w = np.ones(len(idx), dtype=np.float64)
        s = float(np.sum(w))
        w = w / s if s > 0 else np.ones(len(idx), dtype=np.float64) / float(len(idx))

        # Keep top-2 components per keypoint
        if len(idx) > 2:
            keep = np.argsort(-w)[:2]
            m = m[keep]
            c = c[keep]
            w = w[keep]
            w = w / np.sum(w)

        rect = rects[k]
        for j in range(m.shape[0]):
            verts = _ellipse_vertices(m[j], c[j])
            if len(verts) == 0:
                continue
            mapped: List[Tuple[float, float]] = []
            for x_hm, y_hm in verts:
                gx = rect["x0"] + (float(x_hm) / hm_w) * rect["w"]
                gy = rect["y0"] + (float(y_hm) / hm_h) * rect["h"]
                mapped.append((
                    float(np.clip(gx / grid_w, 0.0, 1.0)),
                    float(np.clip(gy / grid_h, 0.0, 1.0)),
                ))
            polylines.append(
                fo.Polyline(points=[mapped], closed=True, filled=False, label=COCO_KP_NAMES[k])
            )

        mode_hm = _gmm_mode_on_grid(w, m, c, hm_w, hm_h)
        if mode_hm is not None:
            gx = rect["x0"] + (float(mode_hm[0]) / hm_w) * rect["w"]
            gy = rect["y0"] + (float(mode_hm[1]) / hm_h) * rect["h"]
            mode_keypoints.append(
                fo.Keypoint(
                    points=[(
                        float(np.clip(gx / grid_w, 0.0, 1.0)),
                        float(np.clip(gy / grid_h, 0.0, 1.0)),
                    )],
                    label=COCO_KP_NAMES[k],
                )
            )

    return polylines, mode_keypoints


def _gmm_per_kp_to_grid_overlays(
    gmm_per_kp: List[Any],
    n_kp: int,
    hm_w: int,
    hm_h: int,
    rects: List[Dict[str, float]],
    grid_w: int,
    grid_h: int,
    fo: Any,
) -> Tuple[List[Any], List[Any]]:
    """Create per-keypoint GMM ellipses and mode points on the grid."""
    polylines: List[Any] = []
    mode_keypoints: List[Any] = []

    for k in range(min(n_kp, len(rects))):
        info = gmm_per_kp[k] if k < len(gmm_per_kp) and isinstance(gmm_per_kp[k], dict) else {}
        weights = np.asarray(info.get("weights", []), dtype=np.float64).reshape(-1)
        means = np.asarray(info.get("means", []), dtype=np.float64)
        covs = np.asarray(info.get("covariances", []), dtype=np.float64)

        if means.ndim != 2 or covs.ndim != 3 or means.shape[0] == 0:
            continue

        n = min(means.shape[0], covs.shape[0], max(1, weights.shape[0]))
        means = means[:n, :2]
        covs = covs[:n, :2, :2]
        if weights.shape[0] < n:
            w = np.ones(n, dtype=np.float64) / float(n)
        else:
            w = weights[:n]
            w = np.clip(w, 0.0, None)
            s = float(np.sum(w))
            w = (w / s) if s > 0 else (np.ones(n, dtype=np.float64) / float(n))

        # Safety: if more than 2 are present in legacy/irregular packets, keep top-2 by weight
        if n > 2:
            idx = np.argsort(-w)[:2]
            means = means[idx]
            covs = covs[idx]
            w = w[idx]
            w = w / np.sum(w)

        rect = rects[k]

        # Ellipses
        for j in range(len(means)):
            verts = _ellipse_vertices(means[j], covs[j])
            if len(verts) == 0:
                continue
            mapped: List[Tuple[float, float]] = []
            for x_hm, y_hm in verts:
                gx = rect["x0"] + (float(x_hm) / hm_w) * rect["w"]
                gy = rect["y0"] + (float(y_hm) / hm_h) * rect["h"]
                mapped.append((
                    float(np.clip(gx / grid_w, 0.0, 1.0)),
                    float(np.clip(gy / grid_h, 0.0, 1.0)),
                ))
            polylines.append(
                fo.Polyline(points=[mapped], closed=True, filled=False, label=COCO_KP_NAMES[k])
            )

        # Mixture mode (numerical argmax on heatmap grid)
        mode_hm = _gmm_mode_on_grid(w, means, covs, hm_w, hm_h)
        if mode_hm is not None:
            gx = rect["x0"] + (float(mode_hm[0]) / hm_w) * rect["w"]
            gy = rect["y0"] + (float(mode_hm[1]) / hm_h) * rect["h"]
            mode_keypoints.append(
                fo.Keypoint(
                    points=[(
                        float(np.clip(gx / grid_w, 0.0, 1.0)),
                        float(np.clip(gy / grid_h, 0.0, 1.0)),
                    )],
                    label=COCO_KP_NAMES[k],
                )
            )

    return polylines, mode_keypoints


def _gmm_mode_on_grid(
    weights: npt.NDArray,
    means: npt.NDArray,
    covs: npt.NDArray,
    hm_w: int,
    hm_h: int,
) -> Optional[Tuple[float, float]]:
    """Compute mixture mode in heatmap coords via dense grid evaluation."""
    if means.ndim != 2 or covs.ndim != 3 or means.shape[0] == 0:
        return None

    yy, xx = np.mgrid[0:hm_h, 0:hm_w]
    grid = np.stack([xx, yy], axis=-1).astype(np.float64)
    dens = np.zeros((hm_h, hm_w), dtype=np.float64)

    for i in range(means.shape[0]):
        mu = means[i, :2].astype(np.float64)
        cov = covs[i, :2, :2].astype(np.float64)
        if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(cov)):
            continue
        cov = cov + np.eye(2, dtype=np.float64) * 1e-6
        try:
            inv = np.linalg.inv(cov)
            det = float(np.linalg.det(cov))
        except np.linalg.LinAlgError:
            continue
        det = max(det, 1e-12)
        d = grid - mu
        q = np.einsum("...i,ij,...j->...", d, inv, d)
        norm = 1.0 / (2.0 * math.pi * math.sqrt(det))
        dens += float(weights[i]) * norm * np.exp(-0.5 * q)

    if not np.any(np.isfinite(dens)):
        return None
    y, x = np.unravel_index(int(np.nanargmax(dens)), dens.shape)
    return float(x), float(y)


# ===================================================================
# Geometry helpers
# ===================================================================

def _ellipse_vertices(
    mu: npt.NDArray,
    cov: npt.NDArray,
    n_sigma: float = 2.0,
) -> npt.NDArray:
    """Parametric 2-σ ellipse from *(mean, covariance)* --> ``(N, 2)`` vertices."""
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

    t = np.linspace(0, 2 * math.pi, _ELLIPSE_PTS, endpoint=True)
    xs = a * np.cos(t)
    ys = b * np.sin(t)
    ca, sa = math.cos(angle), math.sin(angle)

    return np.stack(
        [ca * xs - sa * ys + float(mu[0]),
         sa * xs + ca * ys + float(mu[1])],
        axis=-1,
    )


def _ellipses_normalised(
    means: npt.NDArray,
    covs: npt.NDArray,
    img_w: int,
    img_h: int,
    fo: Any,
    n_sigma: float = 2.0,
) -> List[Any]:
    """Build ``fo.Polyline`` ellipses normalised to ``[0,1]`` wrt image dims.

    Used on the **original** slice where overlays sit directly on the image.
    """
    if means.ndim == 1:
        means = means.reshape(1, -1)
    if covs.ndim == 2:
        covs = covs.reshape(1, 2, 2)

    polys: List[Any] = []
    for mu, cov in zip(means, covs):
        if mu.shape[0] < 2 or cov.shape != (2, 2):
            continue
        verts = _ellipse_vertices(mu, cov, n_sigma)
        if len(verts) == 0:
            continue
        pts = [
            (
                float(np.clip(v[0] / img_w, 0.0, 1.0)),
                float(np.clip(v[1] / img_h, 0.0, 1.0)),
            )
            for v in verts
        ]
        polys.append(fo.Polyline(points=[pts], closed=True, filled=False))
    return polys


def _map_gmm_to_image_space(
    packet: Dict[str, Any],
    means: npt.NDArray,
    covs: npt.NDArray,
    img_w: int,
    img_h: int,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Map GMM params to image space for overlays on the original image.

    Aggregated GMMs are typically fitted in heatmap coordinates. For the
    original-image overlay we need image-space means/covariances.
    """
    means_arr = np.asarray(means, dtype=np.float64)
    covs_arr = np.asarray(covs, dtype=np.float64)

    if means_arr.ndim == 1:
        means_arr = means_arr.reshape(1, -1)
    if covs_arr.ndim == 2:
        covs_arr = covs_arr.reshape(1, 2, 2)

    n = min(means_arr.shape[0], covs_arr.shape[0])
    if n == 0:
        return means_arr, covs_arr

    means_arr = means_arr[:n, :2]
    covs_arr = covs_arr[:n]

    # Heatmap resolution used by the fitted GMM
    hm = _nested_array(packet, "aggregation", "heatmap_avg")
    hm_h, hm_w = None, None
    if isinstance(hm, np.ndarray) and hm.ndim == 3:
        hm_h, hm_w = int(hm.shape[1]), int(hm.shape[2])

    # Preferred path: exact MMPose affine mapping using stored metadata
    agg = packet.get("aggregation", {})
    metadata = agg.get("mmpose_metadata") if isinstance(agg, dict) else None
    if isinstance(hm, np.ndarray) and hm.ndim == 3 and isinstance(metadata, dict):
        try:
            from ..models.adapters import MMPoseAdapter
            from ..utils.types import StandardizedHeatmap

            ref_hm = StandardizedHeatmap(
                data=np.asarray(hm, dtype=np.float32),
                original_size=(img_h, img_w),
                metadata=metadata,
            )
            means_img, covs_img = MMPoseAdapter.transform_heatmap_gaussians_to_image(
                means_arr.astype(np.float32),
                covs_arr.astype(np.float32),
                ref_hm,
            )
            return np.asarray(means_img, dtype=np.float64), np.asarray(covs_img, dtype=np.float64)
        except Exception:
            logger.debug(
                "Falling back to bbox scaling for GMM->image mapping",
                exc_info=True,
            )

    # Person box is expected in COCO xywh format in image coordinates.
    # Fallback to full image if missing/invalid.
    x0, y0 = 0.0, 0.0
    box_w, box_h = float(img_w), float(img_h)
    gt = packet.get("ground_truth", {})
    bbox = None
    if isinstance(gt, dict):
        # Historical packets may use either key.
        bbox = gt.get("bboxes", gt.get("bbox"))

    if bbox is not None:
        b = np.asarray(bbox, dtype=np.float64).reshape(-1)
        if b.size >= 4 and np.all(np.isfinite(b[:4])):
            bx, by, bw, bh = map(float, b[:4])
            # Accept normalized xywh too
            if 0.0 <= bx <= 1.0 and 0.0 <= by <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0:
                bx *= float(img_w)
                by *= float(img_h)
                bw *= float(img_w)
                bh *= float(img_h)
            if bw > 0 and bh > 0:
                x0, y0, box_w, box_h = bx, by, bw, bh

    if hm_w is None or hm_h is None or hm_w <= 0 or hm_h <= 0:
        sx = float(img_w)
        sy = float(img_h)
    else:
        sx = box_w / float(hm_w)
        sy = box_h / float(hm_h)

    means_img = means_arr.copy()
    means_img[:, 0] = x0 + means_arr[:, 0] * sx
    means_img[:, 1] = y0 + means_arr[:, 1] * sy

    A = np.array([[sx, 0.0], [0.0, sy]], dtype=np.float64)
    covs_img = np.empty((n, 2, 2), dtype=np.float64)
    for i in range(n):
        c = covs_arr[i]
        if c.shape != (2, 2) or not np.all(np.isfinite(c)):
            covs_img[i] = np.eye(2, dtype=np.float64) * 1e-6
            continue
        covs_img[i] = A @ c @ A.T

    return means_img, covs_img


# ===================================================================
# Small helpers
# ===================================================================

def _find_original_image(
    tta_data: List[Dict[str, Any]],
) -> Optional[npt.NDArray]:
    """Return the first usable RGB image from tta_data (prefer 'original')."""
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


def _slug_tta_name(name: str) -> str:
    """Return a stable filename/group-safe slug from a TTA transform name."""
    if not isinstance(name, str) or not name.strip():
        return "tta"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "tta"


def _tta_label_and_slug(name: str, entry: Dict[str, Any]) -> Tuple[str, str]:
    """Build human-readable and slugified TTA names including key params."""
    base = str(name or "tta")
    params = entry.get("params", {}) if isinstance(entry, dict) else {}
    if not isinstance(params, dict) or not params:
        label = base
        return label, _slug_tta_name(label)

    parts: List[str] = []
    for k in sorted(params.keys()):
        v = params[k]
        if isinstance(v, (int, float, np.floating)):
            parts.append(f"{k}={float(v):.3g}")
        else:
            parts.append(f"{k}={v}")

    label = f"{base} + " + " + ".join(parts)
    return label, _slug_tta_name(label)


def _apply_scalars(sample: Any, scalars: Dict[str, Any]) -> None:
    """Attach shared metric fields to a FiftyOne sample."""
    for k, v in scalars.items():
        sample[k] = v


def _nested_array(d: Dict, *keys: str) -> Optional[npt.NDArray]:
    """Navigate ``d[k1][k2][...][kN]`` safely; return ndarray or ``None``."""
    obj: Any = d
    for k in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(k)
    return obj if isinstance(obj, np.ndarray) else None


def _to_fo_keypoint(
    coords: npt.NDArray,
    img_w: int,
    img_h: int,
    fo: Any,
) -> Any:
    """``(N, 2|3)`` array → ``fiftyone.Keypoint`` (normalised to ``[0,1]``)."""
    pts = [
        (
            float(np.clip(row[0] / img_w, 0.0, 1.0)),
            float(np.clip(row[1] / img_h, 0.0, 1.0)),
        )
        for row in coords
    ]
    return fo.Keypoint(points=pts)


_VIS_LABELS = {0: "not_labeled", 1: "occluded", 2: "visible"}


def _to_fo_keypoints_labeled(
    coords: npt.NDArray,
    img_w: int,
    img_h: int,
    fo: Any,
    *,
    gt_coords_with_vis: Optional[npt.NDArray] = None,
) -> Any:
    """``(N, 2|3)`` array → ``fo.Keypoints`` with **one** ``fo.Keypoint``
    per body part so that each part can be individually filtered by its
    ``label`` in the FiftyOne sidebar.

    If *gt_coords_with_vis* is provided (GT array with visibility in the
    3rd column, COCO convention 0/1/2), a ``visibility`` tag is stored
    on each keypoint so the user can also filter by visibility.
    """
    n_kp = min(coords.shape[0], _N_KP)
    keypoints: List[Any] = []
    for k in range(n_kp):
        x_norm = float(np.clip(coords[k, 0] / img_w, 0.0, 1.0))
        y_norm = float(np.clip(coords[k, 1] / img_h, 0.0, 1.0))

        # Visibility from GT (or from the 3rd col of coords itself)
        vis_code = 2  # default visible
        if gt_coords_with_vis is not None and k < gt_coords_with_vis.shape[0]:
            vis_code = int(gt_coords_with_vis[k, 2]) if gt_coords_with_vis.shape[1] >= 3 else 2
        elif coords.shape[1] >= 3:
            vis_code = int(coords[k, 2])
        vis_label = _VIS_LABELS.get(vis_code, "unknown")

        kp = fo.Keypoint(
            points=[(x_norm, y_norm)],
            label=COCO_KP_NAMES[k],
        )
        kp["visibility"] = vis_label
        kp["visibility_code"] = vis_code
        kp.tags = [vis_label, COCO_KP_NAMES[k]]
        keypoints.append(kp)

    return fo.Keypoints(keypoints=keypoints)


def _ellipses_per_kp_labeled(
    gmm_per_kp: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    packet: Dict[str, Any],
    fo: Any,
    *,
    n_sigma: float = 2.0,
    gt_coords_with_vis: Optional[npt.NDArray] = None,
) -> Any:
    """Build ``fo.Polylines`` with **one polyline per GMM component**,
    each labeled with the body-part name so that filtering a keypoint
    also filters its associated uncertainty ellipse.

    If *gt_coords_with_vis* is provided, each polyline also receives a
    ``visibility`` tag.
    """
    n_kp = min(len(gmm_per_kp), _N_KP)

    # We need means/covs in image space.  Attempt MMPose affine first.
    # Build a combined means/covs array, transform, then split back.
    all_means: List[npt.NDArray] = []
    all_covs: List[npt.NDArray] = []
    comp_kp_idx: List[int] = []  # which keypoint each component belongs to
    for k in range(n_kp):
        kp_gmm = gmm_per_kp[k]
        m = np.asarray(kp_gmm.get("means", np.empty((0, 2))), dtype=np.float64)
        c = np.asarray(kp_gmm.get("covariances", np.empty((0, 2, 2))), dtype=np.float64)
        if m.ndim == 1:
            m = m.reshape(1, -1)
        if c.ndim == 2:
            c = c.reshape(1, 2, 2)
        nc = min(m.shape[0], c.shape[0])
        for j in range(nc):
            all_means.append(m[j])
            all_covs.append(c[j])
            comp_kp_idx.append(k)

    if not all_means:
        return None

    means_cat = np.stack(all_means)       # (C_total, 2)
    covs_cat = np.stack(all_covs)         # (C_total, 2, 2)

    # Transform to image space
    means_img, covs_img = _map_gmm_to_image_space(
        packet, means_cat, covs_cat, img_w, img_h,
    )

    polys: List[Any] = []
    for i, k in enumerate(comp_kp_idx):
        verts = _ellipse_vertices(means_img[i], covs_img[i], n_sigma)
        if len(verts) == 0:
            continue
        pts = [
            (
                float(np.clip(v[0] / img_w, 0.0, 1.0)),
                float(np.clip(v[1] / img_h, 0.0, 1.0)),
            )
            for v in verts
        ]
        tags = [COCO_KP_NAMES[k]]
        if gt_coords_with_vis is not None and k < gt_coords_with_vis.shape[0]:
            vis_code = int(gt_coords_with_vis[k, 2]) if gt_coords_with_vis.shape[1] >= 3 else 2
            tags.append(_VIS_LABELS.get(vis_code, "unknown"))
        poly = fo.Polyline(
            points=[pts],
            closed=True,
            filled=False,
            label=COCO_KP_NAMES[k],
        )
        poly.tags = tags
        polys.append(poly)

    return fo.Polylines(polylines=polys) if polys else None


"""
FiftyOne Loader – visualise Deep Profiling results in the FiftyOne web app.

This module reads the compressed ``.pkl.gz`` packets written by
:func:`storage.save_deep_analysis`, builds a ``fiftyone.Dataset`` with
keypoints, confidence ellipses and scalar metrics, and launches the
interactive web UI.

Visual mapping
--------------
* **Green**  keypoints → ground truth
* **Red**    keypoints → baseline model prediction
* **Blue**   keypoints → our GMM prediction
* **Cyan** polyline ellipses → 2-σ confidence regions (from GMM covariances)

Requirements
------------
``pip install fiftyone`` (>= 0.23).  The first launch may download MongoDB
automatically.

Usage
-----
>>> from pose_uncertainty.visualization.fiftyone_loader import create_evaluation_dataset
>>> dataset = create_evaluation_dataset("outputs/2026-02-09/12-00-00")
"""

from __future__ import annotations

import logging
import math
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
_ELLIPSE_NUM_POINTS = 64  # vertices per confidence ellipse


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_evaluation_dataset(
    data_dir: str,
    dataset_name: str = "TFG_Evaluation",
) -> Any:
    """Load Deep Profiling packets and build a FiftyOne dataset.

    Parameters
    ----------
    data_dir : str
        Directory containing ``.pkl.gz`` files produced by
        ``EvaluationRunner.run_deep_profiling``.
    dataset_name : str
        Name of the FiftyOne dataset (will be overwritten if it
        already exists).

    Returns
    -------
    fiftyone.Dataset
        The populated dataset.  ``fo.launch_app(dataset)`` is called
        automatically before returning.
    """
    import fiftyone as fo

    data_path = Path(data_dir)
    pkl_files = sorted(data_path.glob("*.pkl.gz"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl.gz files found in {data_path}")

    logger.info("Found %d packets in %s", len(pkl_files), data_path)

    # ---- lazy import of project storage module ----------------------------
    from ..evaluation.storage import load_deep_analysis

    # ---- prepare image cache dir ------------------------------------------
    cache_dir = data_path / _CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- create / overwrite dataset ---------------------------------------
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(name=dataset_name)
    dataset.persistent = True

    # ---- iterate packets --------------------------------------------------
    for pkl_path in pkl_files:
        try:
            packet = load_deep_analysis(str(pkl_path))
            sample = _packet_to_sample(packet, cache_dir, fo)
            if sample is not None:
                dataset.add_sample(sample)
        except Exception:
            logger.exception("Failed to load %s", pkl_path.name)

    logger.info("Dataset '%s' contains %d samples", dataset_name, len(dataset))

    # ---- launch app -------------------------------------------------------
    session = fo.launch_app(dataset, address="0.0.0.0", remote=True)
    print("Servidor activo. Presiona Ctrl+C para salir.")
    session.wait()  # <--- INDISPENSABLE

    return dataset


# ---------------------------------------------------------------------------
# Packet → FiftyOne Sample
# ---------------------------------------------------------------------------


def _packet_to_sample(
    packet: Dict[str, Any],
    cache_dir: Path,
    fo: Any,
) -> Any:
    """Convert a single analysis packet into a ``fiftyone.Sample``."""
    meta = packet.get("meta", {})
    image_id = meta.get("image_id", 0)
    sample_type = meta.get("sample_type", "unknown")

    # ---- recover the original image (first TTA entry is 'original') -------
    image_array: Optional[npt.NDArray[np.uint8]] = None
    for tta_entry in packet.get("tta_data", []):
        if tta_entry.get("name", "").startswith("original"):
            img = tta_entry.get("image")
            if isinstance(img, np.ndarray):
                image_array = img
            break

    if image_array is None:
        # Fallback: use any available image
        for tta_entry in packet.get("tta_data", []):
            img = tta_entry.get("image")
            if isinstance(img, np.ndarray):
                image_array = img
                break

    if image_array is None:
        logger.warning("No image found for image_id=%s; skipping.", image_id)
        return None

    # ---- write image to cache ---------------------------------------------
    img_filename = f"{sample_type}_{image_id}.jpg"
    img_path = cache_dir / img_filename
    bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(img_path), bgr)

    img_h, img_w = image_array.shape[:2]

    # ---- create FiftyOne sample -------------------------------------------
    sample = fo.Sample(filepath=str(img_path))

    # Tags: sample_type group
    sample.tags = [sample_type]

    # ---- ground truth keypoints (green) -----------------------------------
    gt = packet.get("ground_truth", {})
    gt_coords = gt.get("coords")  # (N, 3) x,y,vis
    if gt_coords is not None and isinstance(gt_coords, np.ndarray):
        sample["ground_truth"] = _coords_to_fo_keypoints(
            gt_coords, img_w, img_h, fo
        )

    # ---- baseline prediction (red) ----------------------------------------
    preds = packet.get("predictions", {})
    base = preds.get("baseline", {})
    base_coords = base.get("coords")
    if base_coords is not None and isinstance(base_coords, np.ndarray):
        sample["prediction_base"] = _coords_to_fo_keypoints(
            base_coords, img_w, img_h, fo
        )

    # ---- our GMM prediction (blue) ----------------------------------------
    ours = preds.get("ours_gmm", {})
    ours_coords = ours.get("coords")
    if ours_coords is not None and isinstance(ours_coords, np.ndarray):
        sample["prediction_ours"] = _coords_to_fo_keypoints(
            ours_coords, img_w, img_h, fo
        )

    # ---- confidence ellipses (2-sigma) from GMM --------------------------
    gmm_info = packet.get("gmm_model", {})
    gmm_means = gmm_info.get("means")
    gmm_covs = gmm_info.get("covariances")
    if gmm_means is not None and gmm_covs is not None:
        try:
            ellipses = _build_ellipses(
                np.asarray(gmm_means),
                np.asarray(gmm_covs),
                img_w,
                img_h,
                fo,
            )
            if ellipses:
                sample["uncertainty_ellipses"] = fo.Polylines(polylines=ellipses)
        except Exception:
            logger.debug("Could not build ellipses for image_id=%s", image_id)

    # ---- scalar metric fields ---------------------------------------------
    metrics = packet.get("metrics", {})
    sample["oks_delta"] = float(metrics.get("delta_oks", 0.0))
    sample["nll"] = float(metrics.get("nll", 0.0))
    sample["entropy"] = float(metrics.get("entropy", 0.0))
    sample["oks_base"] = float(metrics.get("oks_base", 0.0))
    sample["oks_ours"] = float(metrics.get("oks_ours", 0.0))
    sample["covariance_volume"] = float(metrics.get("covariance_volume", 0.0))

    return sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coords_to_fo_keypoints(
    coords: npt.NDArray[np.float32],
    img_w: int,
    img_h: int,
    fo: Any,
) -> Any:
    """Convert an (N, 2|3) array to a ``fiftyone.Keypoints`` field.

    FiftyOne Keypoint coordinates must be normalised to [0, 1].
    """
    points: List[Tuple[float, float]] = []
    for row in coords:
        x_norm = float(row[0]) / img_w
        y_norm = float(row[1]) / img_h
        points.append((
            float(np.clip(x_norm, 0.0, 1.0)),
            float(np.clip(y_norm, 0.0, 1.0)),
        ))
    return fo.Keypoint(points=points)


def _build_ellipses(
    means: npt.NDArray,
    covariances: npt.NDArray,
    img_w: int,
    img_h: int,
    fo: Any,
    n_sigma: float = 2.0,
) -> List[Any]:
    """Generate FiftyOne ``Polyline`` ellipses from GMM parameters.

    Each (mean, covariance) pair is converted into a closed polygon
    approximating the *n_sigma* confidence ellipse.
    """
    if means.ndim == 1:
        means = means.reshape(1, -1)
    if covariances.ndim == 2:
        covariances = covariances.reshape(1, 2, 2)

    polylines: List[Any] = []

    for mu, cov in zip(means, covariances):
        if mu.shape[0] < 2 or cov.shape != (2, 2):
            continue

        try:
            eigvals, eigvecs = np.linalg.eigh(cov[:2, :2])
        except np.linalg.LinAlgError:
            continue

        # Clamp eigenvalues to avoid degenerate ellipses
        eigvals = np.clip(eigvals, 1e-6, None)

        # Semi-axes lengths (n-sigma)
        a = n_sigma * math.sqrt(eigvals[1])
        b = n_sigma * math.sqrt(eigvals[0])

        # Rotation angle
        angle = math.atan2(eigvecs[1, 1], eigvecs[0, 1])

        # Parametric ellipse points
        t = np.linspace(0, 2 * math.pi, _ELLIPSE_NUM_POINTS, endpoint=True)
        xs = a * np.cos(t)
        ys = b * np.sin(t)

        # Rotate + translate
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x_rot = cos_a * xs - sin_a * ys + mu[0]
        y_rot = sin_a * xs + cos_a * ys + mu[1]

        # Normalise to [0, 1]
        x_norm = np.clip(x_rot / img_w, 0.0, 1.0)
        y_norm = np.clip(y_rot / img_h, 0.0, 1.0)

        pts = [(float(xn), float(yn)) for xn, yn in zip(x_norm, y_norm)]
        polylines.append(
            fo.Polyline(
                points=[pts],
                closed=True,
                filled=False,
            )
        )

    return polylines

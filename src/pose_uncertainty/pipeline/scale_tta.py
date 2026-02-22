"""
Scale Test-Time Augmentation for top-down pose estimation.

Problem
-------
In top-down pose estimation the model receives a **crop of the bounding box**.
Scaling the bbox before cropping effectively zooms in (scale < 1.0) or out
(scale > 1.0).  Zooming in may push keypoints outside the field of view
(FOV), producing diffuse / uniform heatmaps that **poison** a naive average.

Solution
--------
For each scale:
1. Scale the bounding box around its center (clamped to image bounds).
2. Run the full inner TTA pipeline (flip + photometric) using the scaled bbox.
3. Compute a **per-keypoint confidence** from the resulting heatmap
   (based on peak value × sharpness, with NO hard threshold).
4. **Inverse-transform** heatmaps back to the original bbox coordinate space
   using ``F.grid_sample`` with ``align_corners=False`` (DARK-compatible).
5. Aggregate across scales with **confidence-weighted averaging**.

Integration
-----------
Scale TTA wraps *around* the existing flip+photometric pipeline in
``runner._evaluate_single``.  The inner pipeline runs unchanged at every
scale; only the bbox passed to the model changes.

References
----------
* DARK Pose (Zhang et al., 2020) – uses ``align_corners=False``
* MMPose default coordinate convention – half-pixel offset
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


# ===================================================================
# Configuration
# ===================================================================

@dataclass
class ScaleAugConfig:
    """Configuration for Scale TTA.

    Attributes
    ----------
    enabled : bool
        Master switch.  When False the entire module is a no-op.
    scales : list[float]
        Scale factors relative to the original bbox.
        * < 1.0 → zoom in  (tighter crop, fewer keypoints visible)
        * 1.0   → identity
        * > 1.0 → zoom out (more context, lower resolution on person)
    confidence_threshold : float
        **Soft** floor for peak value (deprecated, kept for backward compat).
    aggregation : str
        ``"weighted_mean"`` — confidence-weighted average (recommended).
        ``"max_confidence"`` — pick the scale with highest confidence per kpt.
    
    Confidence weighting parameters:
    --------------------------------
    sharpness_scale : float
        Divisor in the sigmoid sharpness term: (1 - 1/(1 + sharp/S)).
        Larger values make the sigmoid gentler (less penalty for diffuse maps).
        Default: 5.0
    epsilon : float
        Small constant to avoid division by zero in sharpness calculation.
        Default: 1e-10
    peak_exponent : float
        Exponent applied to peak value: conf = peak^α × sharpness_score.
        α > 1 → more aggressive suppression of weak peaks.
        α < 1 → more lenient.
        Default: 1.0 (linear)
    sharpness_exponent : float
        Exponent applied to sharpness score before multiplication.
        β > 1 → stronger emphasis on sharp heatmaps.
        Default: 1.0 (linear)
    """

    enabled: bool = False
    scales: List[float] = field(default_factory=lambda: [0.9, 1.0, 1.1])
    confidence_threshold: float = 0.01  # deprecated
    aggregation: str = "weighted_mean"
    
    # Confidence weighting
    sharpness_scale: float = 5.0
    epsilon: float = 1e-10
    peak_exponent: float = 1.0
    sharpness_exponent: float = 1.0

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "ScaleAugConfig":
        """Build from a plain dict (e.g. resolved Hydra config node)."""
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            scales=list(cfg.get("scales", [0.9, 1.0, 1.1])),
            confidence_threshold=float(cfg.get("confidence_threshold", 0.01)),
            aggregation=str(cfg.get("aggregation", "weighted_mean")),
            sharpness_scale=float(cfg.get("sharpness_scale", 5.0)),
            epsilon=float(cfg.get("epsilon", 1e-10)),
            peak_exponent=float(cfg.get("peak_exponent", 1.0)),
            sharpness_exponent=float(cfg.get("sharpness_exponent", 1.0)),
        )


# ===================================================================
# Bounding-box helpers
# ===================================================================

def scale_bbox(
    bbox: Tuple[float, float, float, float],
    scale_factor: float,
    image_h: int,
    image_w: int,
) -> Tuple[float, float, float, float]:
    """Scale a COCO-format bbox ``(x, y, w, h)`` around its center.

    The result is clamped to the image boundaries so the crop is always
    valid.

    Parameters
    ----------
    bbox : tuple
        ``(x, y, w, h)`` in COCO format.
    scale_factor : float
        Multiplicative factor (1.0 = no change).
    image_h, image_w : int
        Source image dimensions for clamping.

    Returns
    -------
    tuple
        Scaled ``(x, y, w, h)`` bbox.
    """
    x, y, w, h = bbox

    cx = x + w / 2.0
    cy = y + h / 2.0

    new_w = w * scale_factor
    new_h = h * scale_factor

    new_x = cx - new_w / 2.0
    new_y = cy - new_h / 2.0

    # Clamp to image boundaries
    new_x = max(0.0, new_x)
    new_y = max(0.0, new_y)
    # Right / bottom edge
    if new_x + new_w > image_w:
        new_w = image_w - new_x
    if new_y + new_h > image_h:
        new_h = image_h - new_y

    return (float(new_x), float(new_y), float(new_w), float(new_h))


# ===================================================================
# Heatmap confidence (NO hard gating)
# ===================================================================

def compute_heatmap_confidence(
    heatmaps: npt.NDArray[np.float32],
    sharpness_scale: float = 5.0,
    epsilon: float = 1e-10,
    peak_exponent: float = 1.0,
    sharpness_exponent: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Per-keypoint confidence from heatmap sharpness.

    A heatmap with a clear, concentrated peak is "confident".
    A heatmap that is diffuse / nearly uniform (out-of-FOV keypoint)
    gets a confidence close to zero **continuously** — no hard gate.

    The score for keypoint *k* is::

        peak_k  = max(heatmap_k)
        mean_k  = mean(heatmap_k)
        sharp_k = peak_k / (mean_k + ε)
        sharpness_score = (1 − 1/(1 + sharp_k / S))^β
        conf_k  = peak_k^α  ×  sharpness_score

    * ``peak_k`` anchors the value to the actual detection strength.
    * The sigmoid-like sharpness term penalises uniform maps.
    * There is **no** hard threshold — even a peak of 0.02 contributes,
      just with very low weight.

    Parameters
    ----------
    heatmaps : ndarray, shape ``(K, H, W)``
        Normalised heatmaps in [0, 1].
    sharpness_scale : float
        Divisor S in the sigmoid term.
    epsilon : float
        Small constant to avoid division by zero.
    peak_exponent : float
        Exponent α applied to peak value.
    sharpness_exponent : float
        Exponent β applied to sharpness score.

    Returns
    -------
    ndarray, shape ``(K,)``
        Continuous confidence in [0, ~1].
    """
    K = heatmaps.shape[0]
    flat = heatmaps.reshape(K, -1)

    peak = flat.max(axis=1).astype(np.float64)       # (K,)
    mean = flat.mean(axis=1).astype(np.float64)       # (K,)

    sharpness = peak / (mean + epsilon)
    sharpness_score = 1.0 - 1.0 / (1.0 + sharpness / sharpness_scale)
    
    # Apply exponents
    peak_weighted = np.power(peak, peak_exponent)
    sharpness_weighted = np.power(sharpness_score, sharpness_exponent)

    confidence = peak_weighted * sharpness_weighted   # (K,)
    return confidence


# ===================================================================
# Inverse geometric transform  (heatmap space)
# ===================================================================

def inverse_transform_heatmap(
    heatmap: npt.NDArray[np.float32],
    metadata_original: Dict[str, Any],
    metadata_scaled: Dict[str, Any],
) -> npt.NDArray[np.float32]:
    """Map heatmaps from *scaled*-bbox heatmap space to *original*-bbox space.

    Uses the actual MMPose affine coordinate mapping (``input_center``,
    ``input_scale``) instead of raw bbox coordinates.  This correctly
    accounts for the padding and aspect-ratio adjustments that MMPose
    applies when preparing the crop (~1.25× padding).

    The affine mapping from heatmap pixel ``p`` to image coordinate is::

        img = p * (input_scale / hm_size) + input_center - 0.5 * input_scale

    For ``grid_sample`` we compute the *inverse* for each output pixel:

    1. output pixel → image coords   (using *original* metadata)
    2. image coords → input pixel    (inverse of *scaled* metadata)

    When ``metadata_original == metadata_scaled`` the mapping reduces
    to the identity (verified algebraically).

    Parameters
    ----------
    heatmap : ndarray, shape ``(K, H, W)``
        Heatmaps predicted using the *scaled* bbox.
    metadata_original : dict
        Must contain ``input_center`` and ``input_scale`` from the
        prediction with the *original* (identity-scale) bbox.
    metadata_scaled : dict
        Same keys, from the prediction with the *scaled* bbox.

    Returns
    -------
    ndarray, shape ``(K, H, W)``
        Heatmaps warped into the original-bbox heatmap coordinate frame.
    """
    import torch
    import torch.nn.functional as F

    K, H, W = heatmap.shape

    # Extract MMPose crop geometry ------------------------------------------
    center_o = np.array(metadata_original["input_center"], dtype=np.float64)
    scale_o = np.array(metadata_original["input_scale"], dtype=np.float64)
    center_s = np.array(metadata_scaled["input_center"], dtype=np.float64)
    scale_s = np.array(metadata_scaled["input_scale"], dtype=np.float64)

    # Build grid of output pixel centres (0-indexed) ------------------------
    gx = torch.arange(W, dtype=torch.float64)               # [0 .. W-1]
    gy = torch.arange(H, dtype=torch.float64)               # [0 .. H-1]
    gy_2d, gx_2d = torch.meshgrid(gy, gx, indexing="ij")   # (H, W)

    # Step 1: output pixel → image coords (original metadata)
    #   img_x = hm_x * (scale_o_x / W) + center_o_x - 0.5 * scale_o_x
    img_x = gx_2d * (scale_o[0] / W) + center_o[0] - 0.5 * scale_o[0]
    img_y = gy_2d * (scale_o[1] / H) + center_o[1] - 0.5 * scale_o[1]

    # Step 2: image coords → scaled-bbox heatmap pixel (inverse)
    #   hm_x_s = (img_x - center_s_x + 0.5 * scale_s_x) * (W / scale_s_x)
    hm_x_s = (img_x - center_s[0] + 0.5 * scale_s[0]) * (W / scale_s[0])
    hm_y_s = (img_y - center_s[1] + 0.5 * scale_s[1]) * (H / scale_s[1])

    # Step 3: pixel position → grid_sample coords [-1, 1]
    #   align_corners=False:  grid = 2 * (pixel + 0.5) / size - 1
    sample_x = (2.0 * (hm_x_s + 0.5) / W - 1.0).float()
    sample_y = (2.0 * (hm_y_s + 0.5) / H - 1.0).float()

    grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)   # (1, H, W, 2)
    hm_tensor = torch.from_numpy(heatmap).unsqueeze(0).float()      # (1, K, H, W)

    warped = F.grid_sample(
        hm_tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,  # ← DARK Pose / MMPose convention
    )

    return warped.squeeze(0).numpy()  # (K, H, W)


# ===================================================================
# Multi-scale aggregation
# ===================================================================

def aggregate_multiscale_heatmaps(
    heatmaps_per_scale: List[npt.NDArray[np.float32]],
    confidences_per_scale: List[npt.NDArray[np.float64]],
    method: str = "weighted_mean",
) -> npt.NDArray[np.float32]:
    """Aggregate inverse-transformed heatmaps using confidence weights.

    Parameters
    ----------
    heatmaps_per_scale : list of ndarray ``(K, H, W)``
        All arrays must share the same spatial dimensions.
    confidences_per_scale : list of ndarray ``(K,)``
        Per-keypoint confidence for each scale.
    method : str
        * ``"weighted_mean"`` – confidence-weighted average.
        * ``"max_confidence"`` – pick the single best scale per keypoint.

    Returns
    -------
    ndarray ``(K, H, W)``
    """
    S = len(heatmaps_per_scale)
    K, H, W = heatmaps_per_scale[0].shape

    all_hm = np.stack(heatmaps_per_scale, axis=0)          # (S, K, H, W)
    all_conf = np.stack(confidences_per_scale, axis=0)      # (S, K)

    if method == "max_confidence":
        best = all_conf.argmax(axis=0)                       # (K,)
        result = np.empty((K, H, W), dtype=np.float32)
        for k in range(K):
            result[k] = all_hm[best[k], k]
        return result

    if method == "weighted_mean":
        # Normalise weights per keypoint across scales
        w = all_conf.copy()                                  # (S, K)
        w_sum = w.sum(axis=0, keepdims=True)                 # (1, K)
        safe = w_sum.squeeze(0) > 1e-10                      # (K,)
        w[:, safe] /= w_sum[:, safe]
        w[:, ~safe] = 1.0 / S                                # uniform fallback

        # Weighted sum:  (S, K, 1, 1) * (S, K, H, W) → sum over S
        result = (all_hm * w[:, :, None, None]).sum(axis=0)  # (K, H, W)
        return result.astype(np.float32)

    raise ValueError(f"Unknown aggregation method: {method!r}")


# ===================================================================
# Orchestrator
# ===================================================================

class ScaleAugmentor:
    """High-level orchestrator for scale TTA.

    This class does NOT call the model itself — it only provides
    the geometric operations (bbox scaling, inverse transform,
    confidence, aggregation).  The model calls remain in ``runner.py``
    so that the existing flip+photometric inner loop is reused.

    Typical usage inside ``runner._evaluate_single``::

        if scale_aug.enabled:
            scaled_bboxes = scale_aug.get_scaled_bboxes(bbox, img_h, img_w)
            metadata_per_scale = []
            for sbbox in scaled_bboxes:
                # ... run inner TTA with sbbox → get hm_avg + metadata ...
                hms.append(hm_avg)
                confs.append(compute_heatmap_confidence(hm_avg))
                metadata_per_scale.append(scale_metadata)
            heatmap_avg = scale_aug.aggregate(
                hms, confs, metadata_per_scale, metadata_ref
            )
        else:
            # single-scale path (original code unchanged)
    """

    def __init__(self, config: ScaleAugConfig) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def scales(self) -> List[float]:
        return self.config.scales if self.config.enabled else [1.0]

    def get_scaled_bboxes(
        self,
        bbox: Tuple[float, float, float, float],
        image_h: int,
        image_w: int,
    ) -> List[Tuple[float, float, float, float]]:
        """Return one scaled bbox per configured scale factor."""
        return [scale_bbox(bbox, s, image_h, image_w) for s in self.scales]

    def aggregate(
        self,
        heatmaps_per_scale: List[npt.NDArray[np.float32]],
        confidences_per_scale: List[npt.NDArray[np.float64]],
        metadata_per_scale: List[Dict[str, Any]],
        metadata_original: Dict[str, Any],
    ) -> npt.NDArray[np.float32]:
        """Inverse-transform and aggregate heatmaps across scales.

        Parameters
        ----------
        heatmaps_per_scale : list of ``(K, H, W)``
            Averaged heatmaps from each scale's inner TTA pass.
        confidences_per_scale : list of ``(K,)``
            Per-keypoint confidence at each scale.
        metadata_per_scale : list of dict
            MMPose metadata (``input_center``, ``input_scale``) for each
            scale, as returned by ``predict_batch``.
        metadata_original : dict
            Metadata from the identity (1.0×) scale prediction.  This
            defines the target coordinate frame.

        Returns
        -------
        ndarray ``(K, H, W)``
        """
        transformed: List[npt.NDArray[np.float32]] = []

        for hm, meta_s in zip(heatmaps_per_scale, metadata_per_scale):
            # Skip inverse transform for the identity scale
            if meta_s is metadata_original:
                transformed.append(hm)
            else:
                warped = inverse_transform_heatmap(hm, metadata_original, meta_s)
                transformed.append(warped)

        return aggregate_multiscale_heatmaps(
            transformed,
            confidences_per_scale,
            method=self.config.aggregation,
        )

"""
Test-Time Augmentation (TTA) Engine for Pose Estimation.

This module prepares augmented image batches for downstream model inference
via adapters defined in ``pose_uncertainty.models.adapters``.  The design
follows the project's *Separation of Concerns* philosophy:

* **Core Math only** – uses exclusively NumPy and OpenCV (no PyTorch).
* **Immutable metadata** – each augmented image carries a frozen
  ``TTAMetadata`` dataclass so the downstream consumer knows how to
  invert the transformation on the resulting heatmaps.
* **Product-cartesian generation** – geometric variants (flip) are
  combined with photometric variants (brightness, noise, blur) to
  produce the full augmentation batch.

Typical usage
-------------
>>> from pose_uncertainty.pipeline.tta import TTAEngine
>>> engine = TTAEngine(config)
>>> images, metas = engine.prepare_batch(image)
>>> for img, meta in zip(images, metas):
...     hm = adapter.predict(img)
...     if meta.is_flipped:
...         hm.data = TTAEngine.inverse_flip_heatmap(hm.data, adapter.flip_pairs)
...     accumulator.append(hm.data)
>>> mean_heatmap = np.mean(accumulator, axis=0)

Inverse-flip logic
------------------
The static method ``inverse_flip_heatmap`` replicates *exactly* the
MMPose ``flip_heatmaps`` behaviour (spatial flip  → channel swap →
1-pixel shift for alignment) using pure NumPy operations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TTAMetadata:
    """Immutable metadata attached to every augmented image.

    Attributes
    ----------
    is_flipped : bool
        ``True`` when the image has been horizontally flipped.  The
        downstream consumer must call ``inverse_flip_heatmap`` on
        the predicted heatmap before aggregation.
    transform_type : str
        Human-readable label for traceability, e.g.
        ``"original"``, ``"flip"``, ``"flip+noise"``.
    photometric_params : dict | None
        Parameters applied during photometric augmentation (for
        reproducibility / logging).
    """

    is_flipped: bool
    transform_type: str
    photometric_params: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# TTAEngine
# ---------------------------------------------------------------------------

class TTAEngine:
    """Generate a *cartesian-product* TTA batch from a single input image.

    The engine produces every combination of:

    * **Geometric** variants: original, horizontal flip.
    * **Photometric** variants: brightness/contrast, Gaussian noise,
      Gaussian blur.

    Parameters
    ----------
    config : dict
        Hydra-style configuration dictionary.  Recognised keys:

        ``flip.enabled`` (bool, default ``True``)
            Include a horizontally-flipped copy.
        ``photometric.brightness.enabled`` (bool, default ``False``)
        ``photometric.brightness.delta`` (float, default 30.0)
            Maximum absolute brightness shift (in [0, 255] range).
        ``photometric.contrast.enabled`` (bool, default ``False``)
        ``photometric.contrast.range`` (list[float], default ``[0.8, 1.2]``)
            Multiplicative contrast range.
        ``photometric.noise.enabled`` (bool, default ``False``)
        ``photometric.noise.sigma`` (float, default 10.0)
            Standard deviation of additive Gaussian noise.
        ``photometric.blur.enabled`` (bool, default ``False``)
        ``photometric.blur.kernel_size`` (int, default 3)
            Gaussian blur kernel size (must be odd).
        ``seed`` (int | None, default ``None``)
            Random seed for reproducible photometric augmentations.
            When set, all stochastic operations are deterministic.
    """

    # ----- construction ----- #

    def __init__(self, config: Dict[str, Any]) -> None:
        self._cfg = config

        # Geometric
        flip_cfg = self._cfg.get("flip", {})
        self._flip_enabled: bool = flip_cfg.get("enabled", True)

        # Photometric sub-configs
        photo_cfg = self._cfg.get("photometric", {})

        self._brightness_enabled: bool = photo_cfg.get("brightness", {}).get("enabled", False)
        self._brightness_delta: float = float(photo_cfg.get("brightness", {}).get("delta", 30.0))

        self._contrast_enabled: bool = photo_cfg.get("contrast", {}).get("enabled", False)
        self._contrast_range: Tuple[float, float] = tuple(
            photo_cfg.get("contrast", {}).get("range", [0.8, 1.2])
        )

        self._noise_enabled: bool = photo_cfg.get("noise", {}).get("enabled", False)
        self._noise_sigma: float = float(photo_cfg.get("noise", {}).get("sigma", 10.0))

        self._blur_enabled: bool = photo_cfg.get("blur", {}).get("enabled", False)
        self._blur_ksize: int = int(photo_cfg.get("blur", {}).get("kernel_size", 3))
        # Ensure odd kernel
        if self._blur_ksize % 2 == 0:
            self._blur_ksize += 1

        # Seed
        seed = self._cfg.get("seed", None)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def prepare_batch(
        self,
        image: npt.NDArray[np.uint8],
    ) -> Tuple[List[npt.NDArray[np.uint8]], List[TTAMetadata]]:
        """Build the full TTA batch from a single input image.

        The batch is the **cartesian product** of geometric base images
        and photometric variants::

            bases = [original] + ([flipped] if flip enabled)
            photo = [identity] + [each enabled photometric]
            batch = [photo(base) for base in bases for photo in photo_ops]

        Parameters
        ----------
        image : np.ndarray
            Input RGB image of shape ``(H, W, 3)``, dtype ``uint8``.

        Returns
        -------
        images : list[np.ndarray]
            Augmented image copies (always ``uint8``).
        metadata : list[TTAMetadata]
            Corresponding metadata (same length as *images*).
        """
        assert image.ndim == 3 and image.shape[2] == 3, (
            f"Expected (H, W, 3) uint8 image, got shape {image.shape}"
        )

        # 1. Build geometric bases ------------------------------------------
        geo_bases: List[Tuple[npt.NDArray[np.uint8], bool, str]] = [
            (image.copy(), False, "original"),
        ]
        if self._flip_enabled:
            geo_bases.append((self._apply_flip(image), True, "flip"))

        # 2. Build photometric operations list -------------------------------
        #    Each entry: (callable, suffix_label, params_dict)
        photo_ops: List[Tuple[str, Dict[str, Any]]] = [
            ("identity", {}),
        ]
        if self._brightness_enabled:
            photo_ops.append(("brightness", {"delta": self._brightness_delta}))
        if self._contrast_enabled:
            photo_ops.append(("contrast", {"range": self._contrast_range}))
        if self._noise_enabled:
            photo_ops.append(("noise", {"sigma": self._noise_sigma}))
        if self._blur_enabled:
            photo_ops.append(("blur", {"kernel_size": self._blur_ksize}))

        # 3. Cartesian product -----------------------------------------------
        batch_images: List[npt.NDArray[np.uint8]] = []
        batch_meta: List[TTAMetadata] = []

        for base_img, is_flipped, geo_label in geo_bases:
            for photo_label, photo_params in photo_ops:
                if photo_label == "identity":
                    aug_img = base_img.copy()
                    t_label = geo_label
                    p_params: Optional[Dict[str, Any]] = None
                else:
                    aug_img = self._apply_photometric(
                        base_img, kind=photo_label, params=photo_params,
                    )
                    t_label = f"{geo_label}+{photo_label}"
                    p_params = photo_params

                batch_images.append(aug_img)
                batch_meta.append(
                    TTAMetadata(
                        is_flipped=is_flipped,
                        transform_type=t_label,
                        photometric_params=p_params,
                    )
                )

        return batch_images, batch_meta

    # ------------------------------------------------------------------ #
    # Static utility – inverse flip on heatmaps
    # ------------------------------------------------------------------ #

    @staticmethod
    def inverse_flip_heatmap(
        heatmap: npt.NDArray[np.float32],
        flip_pairs: List[Tuple[int, int]],
        shift_heatmap: bool = True,
    ) -> npt.NDArray[np.float32]:
        """Reverse a horizontal flip on a heatmap array.

        Replicates the MMPose ``flip_heatmaps`` logic using pure NumPy:

        1. **Spatial flip** – ``np.flip(..., axis=-1)`` mirrors the width
           dimension.
        2. **Channel swap** – left/right keypoint channels are exchanged
           according to *flip_pairs*.
        3. **1-pixel shift** – ``heatmap[..., 1:] = heatmap[..., :-1]``
           corrects the half-pixel misalignment introduced by the
           discrete flip.

        Parameters
        ----------
        heatmap : np.ndarray
            Shape ``(K, H, W)`` – probability maps for *K* keypoints.
        flip_pairs : list[tuple[int, int]]
            Symmetric keypoint index pairs, e.g. COCO ``[(1,2), (3,4), ...]``.
        shift_heatmap : bool
            Apply the 1-pixel alignment correction (recommended ``True``).

        Returns
        -------
        np.ndarray
            Corrected heatmap of the same shape and dtype.
        """
        heatmap = np.ascontiguousarray(heatmap, dtype=np.float32)

        # Step 1: spatial flip along width axis
        heatmap = np.flip(heatmap, axis=-1).copy()  # copy for contiguity

        # Step 2: swap left ↔ right channels
        if flip_pairs:
            flip_indices = list(range(heatmap.shape[0]))
            for left, right in flip_pairs:
                flip_indices[left] = right
                flip_indices[right] = left
            heatmap = heatmap[flip_indices]

        # Step 3: 1-pixel alignment shift (matches MMPose exactly)
        if shift_heatmap:
            heatmap[..., 1:] = heatmap[..., :-1]
            # Leftmost column zero (information was shifted right)
            heatmap[..., 0] = 0.0

        return heatmap

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply_flip(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Horizontal flip using OpenCV (axis=1)."""
        return cv2.flip(image, 1)

    def _apply_photometric(
        self,
        image: npt.NDArray[np.uint8],
        *,
        kind: str,
        params: Dict[str, Any],
    ) -> npt.NDArray[np.uint8]:
        """Apply a *single* named photometric augmentation.

        Parameters
        ----------
        image : np.ndarray  (H, W, 3) uint8
        kind  : one of "brightness", "contrast", "noise", "blur"
        params: parameters dict specific to *kind*

        Returns
        -------
        np.ndarray (H, W, 3) uint8
        """
        if kind == "brightness":
            return self._apply_brightness(image, delta=params["delta"])
        if kind == "contrast":
            return self._apply_contrast(image, contrast_range=params["range"])
        if kind == "noise":
            return self._apply_noise(image, sigma=params["sigma"])
        if kind == "blur":
            return self._apply_blur(image, ksize=params["kernel_size"])
        raise ValueError(f"Unknown photometric augmentation: {kind!r}")

    # ---- individual photometric transforms ---- #

    def _apply_brightness(
        self,
        image: npt.NDArray[np.uint8],
        delta: float,
    ) -> npt.NDArray[np.uint8]:
        """Deterministic brightness shift using the engine's RNG."""
        d = self._rng.uniform(-delta, delta)
        img_f = image.astype(np.float32) + d
        return np.clip(img_f, 0, 255).astype(np.uint8)

    def _apply_contrast(
        self,
        image: npt.NDArray[np.uint8],
        contrast_range: Tuple[float, float],
    ) -> npt.NDArray[np.uint8]:
        """Deterministic contrast scaling."""
        low, high = contrast_range
        factor = self._rng.uniform(low, high)
        mean = image.mean()
        img_f = (image.astype(np.float32) - mean) * factor + mean
        return np.clip(img_f, 0, 255).astype(np.uint8)

    def _apply_noise(
        self,
        image: npt.NDArray[np.uint8],
        sigma: float,
    ) -> npt.NDArray[np.uint8]:
        """Additive Gaussian noise."""
        noise = self._rng.normal(0, sigma, size=image.shape).astype(np.float32)
        img_f = image.astype(np.float32) + noise
        return np.clip(img_f, 0, 255).astype(np.uint8)

    def _apply_blur(
        self,
        image: npt.NDArray[np.uint8],
        ksize: int,
    ) -> npt.NDArray[np.uint8]:
        """Gaussian blur via OpenCV."""
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    # ------------------------------------------------------------------ #
    # Repr
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:  # pragma: no cover
        parts = [f"TTAEngine(flip={self._flip_enabled}"]
        for name, enabled in [
            ("brightness", self._brightness_enabled),
            ("contrast", self._contrast_enabled),
            ("noise", self._noise_enabled),
            ("blur", self._blur_enabled),
        ]:
            if enabled:
                parts.append(f" {name}=True")
        return ",".join(parts) + ")"

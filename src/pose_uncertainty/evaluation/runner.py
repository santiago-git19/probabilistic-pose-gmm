"""
Evaluation Runner – orchestrates mass evaluation and deep profiling.

This module provides :class:`EvaluationRunner`, the central class that drives
the two-stage evaluation workflow:

1. **Mass evaluation** (``run_mass_evaluation``):  iterates the full dataset,
   runs inference + TTA + sampling + mixture on every sample, collects scalar
   metrics and persists them to a Parquet file.

2. **Deep profiling** (``run_deep_profiling``):  re-processes a curated subset
   of images (selected by ``diagnostics.select_focus_groups``) while capturing
   every intermediate artefact (images, heatmaps, GMM parameters, …) and
   stores them via :func:`storage.save_deep_analysis`.

Dependencies
------------
* ``pose_uncertainty.models.adapters`` – model inference (``predict``, ``predict_batch``)
* ``pose_uncertainty.pipeline.tta``    – Test-Time Augmentation engine
* ``pose_uncertainty.core.sampling``   – heatmap → Monte-Carlo samples
* ``pose_uncertainty.core.mixture``    – GMM fitting & model selection
* ``pose_uncertainty.utils.metrics``   – OKS, NLL, entropy, covariance volume …
* ``pose_uncertainty.evaluation.storage`` – compressed packet I/O

All configuration is passed via a Hydra ``DictConfig`` / plain dict.
"""

from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# ---- project imports -------------------------------------------------------
from ..models.adapters import create_model_adapter, MMPoseAdapter
from ..pipeline.tta import TTAEngine, TTAMetadata
from ..pipeline.scale_tta import (
    ScaleAugConfig,
    ScaleAugmentor,
    compute_heatmap_confidence,
)
from ..core.sampling import sample_from_heatmap
from ..core.mixture import select_best_model, fit_with_outer_loop
from ..utils.metrics import (
    compute_oks,
    compute_nll,
    compute_entropy,
    compute_covariance_volume,
    count_active_components,
)
from ..utils.types import ImageSample, MixtureResult
from ..data_loader import COCOLoader, CrowdPoseLoader, OCHumanLoader
from . import storage
from .storage import AnalysisPacket

logger = logging.getLogger(__name__)

# COCO keypoint names (17 keypoints)
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATASET_LOADERS = {
    "coco": COCOLoader,
    "crowdpose": CrowdPoseLoader,
    "ochuman": OCHumanLoader,
}


def _build_dataloader(cfg: DictConfig) -> Any:
    """Instantiate the dataset loader dictated by ``cfg.dataset``."""
    ds_cfg = cfg.dataset
    name = ds_cfg.name.lower()
    loader_cls = _DATASET_LOADERS.get(name)
    if loader_cls is None:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(_DATASET_LOADERS)}"
        )
    return loader_cls(
        data_root=ds_cfg.root_dir,
        ann_file=ds_cfg.annotations,
        image_dir=ds_cfg.images,
    )


def _gt_to_arrays(
    sample: ImageSample,
    target_keypoint_names: Optional[List[str]] = None,
    target_num_keypoints: Optional[int] = None,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], float]:
    """Extract (coords_Nx2, visibility_N, area) aligned to the target keypoint space.

    If ``target_keypoint_names`` is provided, keypoints are matched by name.
    This is required for cross-dataset evaluation (e.g., CrowdPose 14-point GT
    against COCO-17 model outputs), where keypoint indices are not compatible.
    """
    kps = sample.ground_truth_keypoints or []

    if target_keypoint_names is not None:
        n = len(target_keypoint_names)
        name_to_idx = {name: idx for idx, name in enumerate(target_keypoint_names)}
    elif target_num_keypoints is not None:
        n = int(target_num_keypoints)
        name_to_idx = None
    else:
        n = (max((kp.id for kp in kps), default=-1) + 1) if kps else 0
        name_to_idx = None

    coords = np.zeros((n, 2), dtype=np.float32)
    vis = np.zeros(n, dtype=np.int32)

    for kp in kps:
        if name_to_idx is not None:
            idx = name_to_idx.get(kp.name)
            if idx is None:
                continue
        else:
            idx = int(kp.id)
            if idx < 0 or idx >= n:
                continue

        coords[idx] = [kp.x, kp.y]
        if kp.confidence >= 0.9:
            vis[idx] = 2
        elif kp.confidence > 0.0:
            vis[idx] = 1
        else:
            vis[idx] = 0

    # Area from bbox (x, y, w, h)
    _, _, w, h = sample.bbox
    area = float(w * h)
    return coords, vis, area


def _flip_bbox(
    bbox: Tuple[float, float, float, float],
    image_width: int,
) -> Tuple[float, float, float, float]:
    """Mirror a COCO-format bbox (x, y, w, h) horizontally."""
    x, y, w, h = bbox
    return (float(image_width - x - w), y, w, h)


def _build_tta_bboxes(
    aug_metas: List[TTAMetadata],
    bbox: Tuple[float, float, float, float],
    image_width: int,
) -> List[Tuple[float, float, float, float]]:
    """Return one bbox per augmentation, flipping when needed."""
    bboxes: List[Tuple[float, float, float, float]] = []
    for meta in aug_metas:
        if meta.is_flipped:
            bboxes.append(_flip_bbox(bbox, image_width))
        else:
            bboxes.append(bbox)
    return bboxes


def _resolve_output_dir(cfg: DictConfig) -> Path:
    """Return the output directory from config, creating it if needed."""
    out_str: str = OmegaConf.select(cfg, "logging.output_dir", default="outputs")
    out = Path(out_str)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_to_plain(cfg: Any) -> Any:
    """Convert OmegaConf to a plain dict (pickle-safe)."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def _as_plain_dict(node: Any) -> Dict[str, Any]:
    """Convert a config node to a plain dict; return {} for non-mappings."""
    if isinstance(node, DictConfig):
        node = OmegaConf.to_container(node, resolve=True)
    return node if isinstance(node, dict) else {}


def _extract_tta_cfg(cfg: DictConfig) -> Dict[str, Any]:
    """Build TTA config supporting both namespaced and root-level Hydra layouts.

    Historically, ``defaults: [tta]`` merges keys from ``tta.yaml`` at root.
    Some callers may instead provide ``cfg.tta`` as a nested node.
    """
    # 1) Preferred: namespaced node (cfg.tta)
    tta_node = OmegaConf.select(cfg, "tta", default=None)
    tta_cfg = _as_plain_dict(tta_node)
    if tta_cfg:
        return tta_cfg

    # 2) Backward-compatible fallback: root-level keys from tta.yaml
    keys = (
        "enabled",
        "flip",
        "photometric",
        "seed",
        "aggregation",
        "post_processing",
        "rotation",
        "scale",
    )
    root_cfg: Dict[str, Any] = {}
    for key in keys:
        value = OmegaConf.select(cfg, key, default=None)
        if value is None:
            continue
        if isinstance(value, DictConfig):
            root_cfg[key] = OmegaConf.to_container(value, resolve=True)
        else:
            root_cfg[key] = value
    return root_cfg


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class EvaluationRunner:
    """Two-stage evaluation pipeline.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration tree.
    dataloader : iterable of ImageSample, optional
        If *None*, one is built from ``cfg.dataset``.
    """

    def __init__(
        self,
        cfg: DictConfig,
        dataloader: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.output_dir = _resolve_output_dir(cfg)

        # ---- model ---------------------------------------------------------
        model_name: str = OmegaConf.select(cfg, "model.name", default="hrnet_w32")
        device: str = OmegaConf.select(cfg, "model.device", default="cuda")
        self.model = create_model_adapter(model_name, device=device)
        logger.info("Model: %s on %s", model_name, device)

        # ---- TTA engine ---------------------------------------------------
        tta_cfg = _extract_tta_cfg(cfg)
        self.tta_engine = TTAEngine(tta_cfg)

        # ---- Scale TTA ---------------------------------------------------
        scale_cfg_raw = tta_cfg.get("scale", {})
        self.scale_aug = ScaleAugmentor(ScaleAugConfig.from_dict(scale_cfg_raw))
        if self.scale_aug.enabled:
            logger.info(
                "Scale TTA enabled: scales=%s, aggregation=%s",
                self.scale_aug.config.scales,
                self.scale_aug.config.aggregation,
            )

        # ---- sampling / mixture config ------------------------------------
        sampling_node = OmegaConf.select(cfg, "sampling", default=None) or OmegaConf.select(cfg, "math_core.sampling", default=None)
        if isinstance(sampling_node, DictConfig):
            self.sampling_cfg = OmegaConf.to_container(sampling_node, resolve=True)
        else:
            self.sampling_cfg = sampling_node or {}

        mixture_node = OmegaConf.select(cfg, "mixture_model", default=None) or OmegaConf.select(cfg, "math_core.mixture_model", default=None)
        if isinstance(mixture_node, DictConfig):
            self.mixture_cfg = OmegaConf.to_container(mixture_node, resolve=True)
        else:
            self.mixture_cfg = mixture_node or {}

        # ---- dataloader ----------------------------------------------------
        self.dataloader = dataloader or _build_dataloader(cfg)

    # ====================================================================
    # Stage 1 – Mass Evaluation (scalars only)
    # ====================================================================

    def run_mass_evaluation(self) -> pd.DataFrame:
        """Iterate the entire dataset, compute scalar metrics, save Parquet.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per image.
        """
        records: List[Dict[str, Any]] = []
        logger.info("=== Mass Evaluation START ===")

        for sample in tqdm(self.dataloader, desc="Mass Evaluation", unit="img"):
            try:
                row = self._evaluate_single(sample)
                if row is not None:
                    records.append(row)
            except Exception:
                logger.exception("Error processing image_id=%s", sample.image_id)

        if not records:
            logger.warning("No valid results collected.")
            return pd.DataFrame()

        df = pd.DataFrame(records)

        parquet_path = self.output_dir / "results_metadata.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info("Saved %d results to %s", len(df), parquet_path)

        logger.info("=== Mass Evaluation END ===")
        return df

    # ---- per-sample pipeline (scalars) ------------------------------------

    def _evaluate_single(self, sample: ImageSample) -> Optional[Dict[str, Any]]:
        """Run full pipeline on one sample, return scalar metrics dict."""
        image = sample.image_array
        bbox = sample.bbox
        gt_coords, vis, area = _gt_to_arrays(
            sample,
            target_keypoint_names=self.model.keypoint_names,
            target_num_keypoints=self.model.num_keypoints,
        )

        if area < 1.0:
            return None  # degenerate bbox

        # 1) Baseline prediction (single forward pass) ---------------------
        base_kps, base_scores = self.model.predict_keypoints(image, bbox=bbox)
        base_coords_2d = base_kps[:, :2] if base_kps.ndim == 2 else base_kps
        oks_base, per_kp_oks_base = compute_oks(base_coords_2d, gt_coords, vis, area)

        # 2) TTA -----------------------------------------------------------
        img_h, img_w = image.shape[:2]
        scaled_bboxes = self.scale_aug.get_scaled_bboxes(bbox, img_h, img_w)

        heatmaps_per_scale: List[npt.NDArray[np.float32]] = []
        confs_per_scale: List[npt.NDArray[np.float64]] = []
        metadata_per_scale: List[Optional[Dict[str, Any]]] = []
        metadata_ref = None

        for sbbox in scaled_bboxes:
            aug_images, aug_metas = self.tta_engine.prepare_batch(image)
            tta_bboxes = _build_tta_bboxes(aug_metas, sbbox, img_w)
            heatmaps_list = self.model.predict_batch(aug_images, bboxes=tta_bboxes)

            # Inverse-flip and average within this scale
            accum: List[npt.NDArray[np.float32]] = []
            scale_metadata = None
            for std_hm, meta in zip(heatmaps_list, aug_metas):
                hm = std_hm.data.copy()
                if meta.is_flipped:
                    hm = TTAEngine.inverse_flip_heatmap(
                        hm, self.model.flip_pairs,
                        shift_heatmap=self.model.shift_heatmap,
                    )
                accum.append(hm)
                # Prefer metadata from a non-flipped prediction for this scale
                if scale_metadata is None and not meta.is_flipped and std_hm.metadata is not None:
                    scale_metadata = std_hm.metadata
            # Fallback: any metadata from this scale
            if scale_metadata is None:
                for std_hm in heatmaps_list:
                    if std_hm.metadata is not None:
                        scale_metadata = std_hm.metadata
                        break

            metadata_per_scale.append(scale_metadata)

            # Track identity-scale metadata for final coord transform
            is_identity = all(abs(a - b) < 1.0 for a, b in zip(sbbox, bbox))
            if is_identity and scale_metadata is not None:
                metadata_ref = scale_metadata

            scale_avg = np.mean(accum, axis=0).astype(np.float32)
            heatmaps_per_scale.append(scale_avg)
            confs_per_scale.append(compute_heatmap_confidence(
                scale_avg,
                sharpness_scale=self.scale_aug.config.sharpness_scale,
                epsilon=self.scale_aug.config.epsilon,
                peak_exponent=self.scale_aug.config.peak_exponent,
                sharpness_exponent=self.scale_aug.config.sharpness_exponent,
            ))

        # Fallback: use first available metadata if no identity scale found
        if metadata_ref is None:
            metadata_ref = next((m for m in metadata_per_scale if m is not None), None)

        # Aggregate across scales (confidence-weighted)
        if self.scale_aug.enabled and len(scaled_bboxes) > 1:
            heatmap_avg = self.scale_aug.aggregate(
                heatmaps_per_scale, confs_per_scale, metadata_per_scale, metadata_ref
            )
        else:
            heatmap_avg = heatmaps_per_scale[0]

        # 3) Per-keypoint sampling + GMM -----------------------------------
        num_kp = heatmap_avg.shape[0]
        gmm_results: List[MixtureResult] = []
        ours_coords = np.zeros((num_kp, 2), dtype=np.float32)
        ours_scores = np.zeros(num_kp, dtype=np.float32)

        for k in range(num_kp):
            hm_k = heatmap_avg[k]
            try:
                samples = sample_from_heatmap(
                    hm_k,
                    num_samples=int(self.sampling_cfg.get("num_samples", 1000)),
                    strategy=self.sampling_cfg.get("method", "rejection"),
                    temperature=float(self.sampling_cfg.get("temperature", 0.1)),
                    use_dequantization=bool(self.sampling_cfg.get("use_dequantization", True)),
                    seed=self.sampling_cfg.get("seed", 42),
                )
                mixture_res = select_best_model(
                    samples.astype(np.float64),
                    aic_weight=float(self.mixture_cfg.get("aic_weight", 0)),
                    bic_weight=float(self.mixture_cfg.get("bic_weight", 1)),
                    reg_covar=float(self.mixture_cfg.get("regularization_strength", 1e-4)),
                    max_iter=int(self.mixture_cfg.get("max_iterations", 100)),
                    tol=float(self.mixture_cfg.get("convergence_threshold", 1e-4)),
                )
                gmm_results.append(mixture_res)
                ours_coords[k] = mixture_res.best_mean.astype(np.float32)
                ours_scores[k] = float(
                    max((c.weight for c in mixture_res.components), default=0.0)
                )
            except Exception:
                logger.debug("GMM failed for keypoint %d, fallback to heatmap argmax", k)
                y, x = np.unravel_index(np.argmax(hm_k), hm_k.shape)
                ours_coords[k] = [float(x), float(y)]
                ours_scores[k] = float(hm_k[y, x])

        # Transform heatmap coords → image coords if metadata available
        if metadata_ref is not None:
            from ..utils.types import StandardizedHeatmap

            ref_hm = StandardizedHeatmap(
                data=heatmap_avg,
                original_size=(image.shape[0], image.shape[1]),
                metadata=metadata_ref,
            )
            try:
                ours_coords = MMPoseAdapter.transform_heatmap_coords_to_image(
                    ours_coords, ref_hm
                )
            except Exception:
                logger.debug("Coord transform failed; using raw heatmap coords")

        # 4) Scalar metrics ------------------------------------------------
        oks_ours, per_kp_oks_ours = compute_oks(ours_coords[:, :2], gt_coords, vis, area)
        delta_oks = oks_ours - oks_base
        
        # Per-keypoint delta OKS
        per_kp_delta_oks = per_kp_oks_ours - per_kp_oks_base

        # Aggregate GMM metrics over keypoints
        all_weights: List[npt.NDArray] = []
        all_covs: List[npt.NDArray] = []
        for mr in gmm_results:
            w = np.array([c.weight for c in mr.components], dtype=np.float32)
            c = np.array([c.covariance for c in mr.components], dtype=np.float32)
            all_weights.append(w)
            all_covs.append(c)

        if all_weights:
            avg_weights = np.mean(
                [w for w in all_weights], axis=0
            ) if all(len(w) == len(all_weights[0]) for w in all_weights) else all_weights[0]
            avg_covs = np.mean(
                [c for c in all_covs], axis=0
            ) if all(c.shape == all_covs[0].shape for c in all_covs) else all_covs[0]
            entropy_val = compute_entropy(avg_weights, avg_covs)
            cov_vol = compute_covariance_volume(avg_covs, avg_weights)
            n_components = int(np.mean([count_active_components(w) for w in all_weights]))
        else:
            entropy_val = 0.0
            cov_vol = 0.0
            n_components = 0

        # NLL (use first usable GMM result as a proxy)
        nll_val = 0.0
        for k, mr in enumerate(gmm_results):
            if mr.components:
                try:
                    # Build a simple sklearn-like scorer from our custom GMM
                    from ..core.mixture import RobustGaussianMixture

                    proxy = RobustGaussianMixture(n_components=len(mr.components))
                    proxy.components_ = mr.components
                    proxy.uniform_weight_ = mr.uniform_weight
                    gt_pt = gt_coords[k].reshape(1, 2)
                    nll_val += compute_nll(proxy, gt_pt)
                except Exception:
                    pass
        nll_val = nll_val / max(len(gmm_results), 1)

        # Build metrics dict with per-keypoint OKS values
        metrics_dict = {
            "image_id": sample.image_id,
            "dataset": sample.dataset_source,
            "oks_base": float(oks_base),
            "oks_ours": float(oks_ours),
            "delta_oks": float(delta_oks),
            "nll": float(nll_val),
            "entropy": float(entropy_val),
            "covariance_vol": float(cov_vol),
            "n_components": int(n_components),
        }
        
        # Add per-keypoint OKS values with COCO keypoint names
        for i, kp_name in enumerate(COCO_KEYPOINT_NAMES):
            if i < len(per_kp_oks_base):
                metrics_dict[f"oks_base_{kp_name}"] = float(per_kp_oks_base[i]) if not np.isnan(per_kp_oks_base[i]) else None
                metrics_dict[f"oks_ours_{kp_name}"] = float(per_kp_oks_ours[i]) if not np.isnan(per_kp_oks_ours[i]) else None
                metrics_dict[f"delta_oks_{kp_name}"] = float(per_kp_delta_oks[i]) if not np.isnan(per_kp_delta_oks[i]) else None
        
        return metrics_dict

    # ====================================================================
    # Stage 2 – Deep Profiling (full artefacts)
    # ====================================================================

    def run_deep_profiling(
        self, target_ids_dict: Dict[str, List[int]]
    ) -> List[Path]:
        """Re-run the pipeline on selected images, capturing all artefacts.

        Parameters
        ----------
        target_ids_dict : dict[str, list[int]]
            Mapping ``sample_type → list_of_image_ids``, produced by
            :func:`diagnostics.select_focus_groups`.

        Returns
        -------
        list[pathlib.Path]
            Paths to the saved ``.pkl.gz`` files.
        """
        # ---- collect unique IDs and their group labels --------------------
        id_to_groups: Dict[int, str] = {}
        unique_ids: Set[int] = set()
        for group_name, ids in target_ids_dict.items():
            for img_id in ids:
                if img_id not in id_to_groups:
                    id_to_groups[img_id] = group_name
                unique_ids.add(img_id)

        logger.info(
            "Deep profiling: %d unique images across %d groups",
            len(unique_ids),
            len(target_ids_dict),
        )

        saved_paths: List[Path] = []

        for sample in tqdm(self.dataloader, desc="Deep Profiling", unit="img"):
            if sample.image_id not in unique_ids:
                continue

            try:
                packet = self._build_deep_packet(sample, id_to_groups)
                path = storage.save_deep_analysis(packet, str(self.output_dir))
                saved_paths.append(path)
            except Exception:
                logger.exception(
                    "Deep profiling error for image_id=%s", sample.image_id
                )

            # Early exit when every target has been processed
            unique_ids.discard(sample.image_id)
            if not unique_ids:
                break

        logger.info("Deep profiling complete: %d packets saved", len(saved_paths))
        return saved_paths

    # ---- per-sample full artefact capture ---------------------------------

    def _build_deep_packet(
        self, sample: ImageSample, id_to_groups: Dict[int, str]
    ) -> AnalysisPacket:
        """Build a complete analysis packet for one image."""
        image = sample.image_array
        bbox = sample.bbox
        gt_coords, vis, area = _gt_to_arrays(
            sample,
            target_keypoint_names=self.model.keypoint_names,
            target_num_keypoints=self.model.num_keypoints,
        )
        timestamp = datetime.datetime.now().isoformat()

        # 1) Baseline -------------------------------------------------------
        base_kps, base_scores = self.model.predict_keypoints(image, bbox=bbox)
        base_coords_2d = base_kps[:, :2] if base_kps.ndim == 2 else base_kps
        oks_base, per_kp_oks_base = compute_oks(base_coords_2d, gt_coords, vis, area)
        base_score_mean = float(np.mean(base_scores))

        # 2) TTA (capture everything) ---------------------------------------
        img_h, img_w = image.shape[:2]
        scaled_bboxes = self.scale_aug.get_scaled_bboxes(bbox, img_h, img_w)

        tta_data: List[Dict[str, Any]] = []
        heatmaps_per_scale: List[npt.NDArray[np.float32]] = []
        confs_per_scale: List[npt.NDArray[np.float64]] = []
        metadata_per_scale: List[Optional[Dict[str, Any]]] = []
        metadata_ref = None

        for scale_idx, sbbox in enumerate(scaled_bboxes):
            aug_images, aug_metas = self.tta_engine.prepare_batch(image)
            tta_bboxes = _build_tta_bboxes(aug_metas, sbbox, img_w)
            std_heatmaps = self.model.predict_batch(
                aug_images, bboxes=tta_bboxes
            )

            accum: List[npt.NDArray[np.float32]] = []
            scale_metadata = None

            for idx, (aug_img, meta, std_hm) in enumerate(
                zip(aug_images, aug_metas, std_heatmaps)
            ):
                hm = std_hm.data.copy()
                if meta.is_flipped:
                    hm = TTAEngine.inverse_flip_heatmap(
                        hm, self.model.flip_pairs,
                        shift_heatmap=self.model.shift_heatmap,
                    )
                accum.append(hm)
                # Prefer metadata from a non-flipped prediction for this scale
                if scale_metadata is None and not meta.is_flipped and std_hm.metadata is not None:
                    scale_metadata = std_hm.metadata

                # Get scale factor directly from config (not computed from bbox)
                scale_factor = self.scale_aug.scales[scale_idx] if scale_idx < len(self.scale_aug.scales) else 1.0
                
                # Include scale factor in params for FiftyOne display
                params_with_scale = dict(meta.photometric_params or {})
                if len(scaled_bboxes) > 1:  # Only add scale info if multiple scales
                    params_with_scale["scale"] = scale_factor
                
                tta_entry: Dict[str, Any] = {
                    "aug_id": len(tta_data),
                    "scale_idx": scale_idx,
                    "scale_bbox": list(sbbox),
                    "scale_factor": float(scale_factor),
                    "name": meta.transform_type,
                    "params": params_with_scale,
                    "image": aug_img,
                    "heatmap": hm,
                    "pred_coords": std_hm.data.mean(axis=(1, 2)),
                }
                tta_data.append(tta_entry)

            # Fallback: any metadata from this scale
            if scale_metadata is None:
                for std_hm in std_heatmaps:
                    if std_hm.metadata is not None:
                        scale_metadata = std_hm.metadata
                        break

            metadata_per_scale.append(scale_metadata)

            # Track identity-scale metadata for final coord transform
            is_identity = all(abs(a - b) < 1.0 for a, b in zip(sbbox, bbox))
            if is_identity and scale_metadata is not None:
                metadata_ref = scale_metadata

            scale_avg = np.mean(accum, axis=0).astype(np.float32)
            heatmaps_per_scale.append(scale_avg)
            confs_per_scale.append(compute_heatmap_confidence(
                scale_avg,
                sharpness_scale=self.scale_aug.config.sharpness_scale,
                epsilon=self.scale_aug.config.epsilon,
                peak_exponent=self.scale_aug.config.peak_exponent,
                sharpness_exponent=self.scale_aug.config.sharpness_exponent,
            ))

        # Fallback: use first available metadata if no identity scale found
        if metadata_ref is None:
            metadata_ref = next((m for m in metadata_per_scale if m is not None), None)

        # Aggregate across scales
        if self.scale_aug.enabled and len(scaled_bboxes) > 1:
            heatmap_avg = self.scale_aug.aggregate(
                heatmaps_per_scale, confs_per_scale, metadata_per_scale, metadata_ref
            )
        else:
            heatmap_avg = heatmaps_per_scale[0]

        # 3) Per-keypoint sampling + GMM -----------------------------------
        num_kp = heatmap_avg.shape[0]
        ours_coords = np.zeros((num_kp, 2), dtype=np.float32)
        ours_scores = np.zeros(num_kp, dtype=np.float32)
        all_sampling_points: List[npt.NDArray[np.float32]] = []
        sampling_points_by_kp: List[npt.NDArray[np.float32]] = []

        # Aggregate GMM params across keypoints
        gmm_weights_list: List[npt.NDArray] = []
        gmm_means_list: List[npt.NDArray] = []
        gmm_covs_list: List[npt.NDArray] = []
        n_comp_total = 0
        gmm_results: List[MixtureResult] = []

        for k in range(num_kp):
            hm_k = heatmap_avg[k]
            try:
                samples = sample_from_heatmap(
                    hm_k,
                    num_samples=int(self.sampling_cfg.get("num_samples", 1000)),
                    strategy=self.sampling_cfg.get("method", "rejection"),
                    temperature=float(self.sampling_cfg.get("temperature", 0.1)),
                    use_dequantization=bool(self.sampling_cfg.get("use_dequantization", True)),
                    seed=self.sampling_cfg.get("seed", 42),
                )
                all_sampling_points.append(samples)
                sampling_points_by_kp.append(samples.astype(np.float32))

                mixture_res = select_best_model(
                    samples.astype(np.float64),
                    aic_weight=float(self.mixture_cfg.get("aic_weight", 0)),
                    bic_weight=float(self.mixture_cfg.get("bic_weight", 1)),
                    reg_covar=float(self.mixture_cfg.get("regularization_strength", 1e-4)),
                    max_iter=int(self.mixture_cfg.get("max_iterations", 100)),
                    tol=float(self.mixture_cfg.get("convergence_threshold", 1e-4)),
                )
                gmm_results.append(mixture_res)
                ours_coords[k] = mixture_res.best_mean.astype(np.float32)
                ours_scores[k] = float(
                    max((c.weight for c in mixture_res.components), default=0.0)
                )

                w = np.array([c.weight for c in mixture_res.components])
                m = np.array([c.mean for c in mixture_res.components])
                c = np.array([c.covariance for c in mixture_res.components])
                gmm_weights_list.append(w)
                gmm_means_list.append(m)
                gmm_covs_list.append(c)
                n_comp_total += len(mixture_res.components)

            except Exception:
                logger.debug("GMM failed for keypoint %d; argmax fallback", k)
                y, x = np.unravel_index(np.argmax(hm_k), hm_k.shape)
                ours_coords[k] = [float(x), float(y)]
                ours_scores[k] = float(hm_k[y, x])
                all_sampling_points.append(np.empty((0, 2), dtype=np.float32))
                sampling_points_by_kp.append(np.empty((0, 2), dtype=np.float32))

        # Transform heatmap → image coordinates
        if metadata_ref is not None:
            from ..utils.types import StandardizedHeatmap

            ref_hm = StandardizedHeatmap(
                data=heatmap_avg,
                original_size=(image.shape[0], image.shape[1]),
                metadata=metadata_ref,
            )
            try:
                ours_coords = MMPoseAdapter.transform_heatmap_coords_to_image(
                    ours_coords, ref_hm
                )
            except Exception:
                logger.debug("Coord transform failed; raw heatmap coords used")

        oks_ours, per_kp_oks_ours = compute_oks(ours_coords[:, :2], gt_coords, vis, area)
        ours_score_mean = float(np.mean(ours_scores))

        # Aggregate GMM info -----------------------------------------------
        if gmm_weights_list:
            agg_w = np.concatenate(gmm_weights_list)
            agg_m = np.concatenate(gmm_means_list)
            agg_c = np.concatenate(gmm_covs_list)
        else:
            agg_w = np.array([1.0])
            agg_m = np.zeros((1, 2))
            agg_c = np.eye(2).reshape(1, 2, 2)

        # Per-keypoint avg metrics ------------------------------------------
        kp_weights = [np.array([c.weight for c in mr.components], dtype=np.float32) for mr in gmm_results]
        kp_covs = [np.array([c.covariance for c in mr.components], dtype=np.float32) for mr in gmm_results]

        if kp_weights:
            avg_w = np.mean(kp_weights, axis=0) if all(
                len(w) == len(kp_weights[0]) for w in kp_weights
            ) else kp_weights[0]
            avg_c = np.mean(kp_covs, axis=0) if all(
                c.shape == kp_covs[0].shape for c in kp_covs
            ) else kp_covs[0]
            entropy_val = compute_entropy(avg_w, avg_c)
            cov_vol = compute_covariance_volume(avg_c, avg_w)
        else:
            entropy_val = 0.0
            cov_vol = 0.0

        nll_val = 0.0
        for k, mr in enumerate(gmm_results):
            if mr.components:
                try:
                    from ..core.mixture import RobustGaussianMixture

                    proxy = RobustGaussianMixture(n_components=len(mr.components))
                    proxy.components_ = mr.components
                    proxy.uniform_weight_ = mr.uniform_weight
                    nll_val += compute_nll(proxy, gt_coords[k].reshape(1, 2))
                except Exception:
                    pass
        nll_val /= max(len(gmm_results), 1)

        delta_oks = oks_ours - oks_base

        # Make coords Nx3 (x, y, score) for packet --------------------------
        base_coords_3 = np.column_stack(
            [base_coords_2d, base_scores]
        ).astype(np.float32) if base_scores.shape[0] == base_coords_2d.shape[0] else base_coords_2d
        ours_coords_3 = np.column_stack(
            [ours_coords[:, :2], ours_scores]
        ).astype(np.float32)

        # GT as Nx3
        gt_coords_3 = np.column_stack(
            [gt_coords, vis.astype(np.float32)]
        ).astype(np.float32)

        # Sampling points aggregated
        sampling_all = (
            np.concatenate(all_sampling_points, axis=0).astype(np.float32)
            if all_sampling_points
            else np.empty((0, 2), dtype=np.float32)
        )

        sample_type = id_to_groups.get(sample.image_id, "unknown")

        model_name = OmegaConf.select(self.cfg, "model.name", default="unknown")

        packet: AnalysisPacket = {  # type: ignore[typeddict-item]
            "meta": {
                "image_id": int(sample.image_id),
                "file_name": str(sample.image_path),
                "dataset": sample.dataset_source,
                "model_name": str(model_name),
                "sample_type": sample_type,
                "timestamp": timestamp,
            },
            "ground_truth": {
                "coords": gt_coords_3,
                "bboxes": list(sample.bbox),
                "area": float(area),
            },
            "predictions": {
                "baseline": {
                    "coords": base_coords_3,
                    "score": float(base_score_mean),
                },
                "ours_gmm": {
                    "coords": ours_coords_3,
                    "score": float(ours_score_mean),
                },
            },
            "metrics": {
                "oks_base": float(oks_base),
                "oks_ours": float(oks_ours),
                "delta_oks": float(delta_oks),
                "nll": float(nll_val),
                "entropy": float(entropy_val),
                "covariance_volume": float(cov_vol),
                "per_kp_oks_ours": per_kp_oks_ours,
                "per_kp_oks_base": per_kp_oks_base,
            },
            "tta_data": tta_data,
            "aggregation": {
                "heatmap_avg": heatmap_avg,
                "sampling_points": sampling_all,
                "sampling_points_by_kp": sampling_points_by_kp,
                "mmpose_metadata": metadata_ref,
            },
            "gmm_model": {
                "n_components": int(n_comp_total),
                "weights": agg_w,
                "means": agg_m,
                "covariances": agg_c,
            },
            "gmm_per_kp": [
                {
                    "weights": np.array([c.weight for c in mr.components], dtype=np.float64),
                    "means": np.array([c.mean for c in mr.components], dtype=np.float64),
                    "covariances": np.array([c.covariance for c in mr.components], dtype=np.float64),
                }
                for mr in gmm_results
            ],
            "config_used": _cfg_to_plain(self.cfg),
        }

        return packet

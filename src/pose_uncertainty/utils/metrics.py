"""
Evaluation metrics for human pose estimation.

Implements standard benchmarking metrics used in COCO, MPII, and other
pose estimation datasets. All metrics operate on numpy arrays for efficiency.

Key Metrics:
-----------
1. **OKS (Object Keypoint Similarity)**: COCO's primary metric, accounting for
   keypoint-specific variances (e.g., wrists are harder than hips).

2. **PCK (Percentage of Correct Keypoints)**: Binary correctness within a threshold,
   normalized by person scale (torso size or head size).

3. **AUC (Area Under Curve)**: PCK@[α] integrated over multiple thresholds.

4. **NLL (Negative Log-Likelihood)**: Evaluates GMM model fit to ground truth.

5. **Entropy**: Measures uncertainty in Gaussian mixture distributions.

References:
    - COCO Evaluation: https://cocodataset.org/#keypoints-eval
    - MPII Pose: http://human-pose.mpi-inf.mpg.de/
"""

from typing import Optional, Dict, List, Union, Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from .types import RefinedKeypoint, PoseEstimationResult


# COCO keypoint sigmas (17 keypoints: nose, eyes, ears, shoulders, elbows, 
# wrists, hips, knees, ankles)
COCO_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079,  # head (0-6)
    0.072, 0.072, 0.062, 0.062,                        # arms (7-10)
    0.107, 0.107, 0.087, 0.087, 0.089, 0.089          # legs (11-16)
], dtype=np.float32)


def compute_l2_distance(
    pred_coords: npt.NDArray[np.float32],
    gt_coords: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Compute vectorized Euclidean (L2) distance between predictions and ground truth.
    
    Mathematical Definition:
    -----------------------
        d = ||pred - gt||₂ = √((x_pred - x_gt)² + (y_pred - y_gt)²)
    
    Args:
        pred_coords: Array of shape (num_keypoints, 2) with [x, y] coordinates.
        gt_coords: Array of shape (num_keypoints, 2) with [x, y] coordinates.
    
    Returns:
        Array of shape (num_keypoints,) with L2 distances for each keypoint.
    
    Example:
        >>> pred = np.array([[100.0, 200.0], [150.0, 250.0]])
        >>> gt = np.array([[101.0, 201.0], [148.0, 248.0]])
        >>> distances = compute_l2_distance(pred, gt)
        >>> # distances ≈ [1.414, 3.606]
    """
    assert pred_coords.shape == gt_coords.shape, \
        f"Shape mismatch: pred {pred_coords.shape} vs gt {gt_coords.shape}"
    assert pred_coords.shape[1] == 2, \
        f"Expected (N, 2) coordinates, got {pred_coords.shape}"
    
    # Vectorized computation: sqrt(sum((pred - gt)^2, axis=1))
    return np.linalg.norm(pred_coords - gt_coords, axis=1).astype(np.float32)


def compute_oks(
    pred_coords: npt.NDArray[np.float32],
    gt_coords: npt.NDArray[np.float32],
    visible_flags: npt.NDArray[np.int32],
    area: float,
    sigmas: Optional[npt.NDArray[np.float32]] = None
) -> tuple[float, npt.NDArray[np.float32]]:
    """
    Compute Object Keypoint Similarity (OKS) between predicted and ground truth poses.
    
    Mathematical Definition:
    -----------------------
    OKS measures spatial accuracy weighted by keypoint-specific uncertainty:
    
        OKS = Σᵢ exp(-dᵢ² / (2 s² κᵢ²)) δ(vᵢ > 0) / Σᵢ δ(vᵢ > 0)
    
    Where:
        - dᵢ: Euclidean distance for keypoint i
        - s²: Object scale (√area, typically from bbox)
        - κᵢ: Per-keypoint constant reflecting difficulty (from COCO paper)
        - vᵢ: Visibility flag (0=unlabeled, 1=occluded, 2=visible)
    
    Intuition:
        OKS ∈ [0, 1] where 1 = perfect alignment. It's analogous to IoU for boxes,
        but adapted for sparse keypoint sets with varying localization difficulty.
    
    Args:
        pred_coords: Array of shape (num_keypoints, 2) with [x, y] predictions.
        gt_coords: Array of shape (num_keypoints, 2) with [x, y] ground truth.
        visible_flags: Array of shape (num_keypoints,) with visibility 
                      (0=unlabeled, 1=occluded, 2=visible). Only >0 are evaluated.
        area: Bounding box area for scale normalization (s² = area).
        sigmas: Per-keypoint standard deviations (κᵢ). If None, uses COCO defaults.
    
    Returns:
        Tuple of (mean_oks, per_keypoint_oks):
            - mean_oks: OKS score ∈ [0, 1] averaged over visible keypoints.
            - per_keypoint_oks: Array of shape (num_keypoints,) with OKS for each keypoint.
                               Non-visible keypoints are set to NaN.
    
    COCO Defaults (17 keypoints):
        σ = [.026, .025, .025, .035, .035, .079, .079, .072, .072, .062, .062, 
             .107, .107, .087, .087, .089, .089] / 10.0
    
    Example:
        >>> pred = np.array([[100.0, 200.0], [150.0, 250.0]])
        >>> gt = np.array([[100.5, 200.2], [148.0, 248.0]])
        >>> visible = np.array([2, 2])
        >>> area = 10000.0
        >>> mean_oks, per_kp_oks = compute_oks(pred, gt, visible, area)
        >>> # mean_oks ≈ 0.99 (very close predictions)
        >>> # per_kp_oks ≈ [0.99, 0.98]
    """
    # Use COCO defaults if not provided
    if sigmas is None:
        sigmas = COCO_SIGMAS
    
    # Ensure correct shapes
    num_keypoints = pred_coords.shape[0]
    assert gt_coords.shape[0] == num_keypoints, "Prediction and GT must have same length"
    assert visible_flags.shape[0] == num_keypoints, "Visibility flags must match keypoints"
    assert sigmas.shape[0] == num_keypoints, f"Sigmas length {len(sigmas)} != {num_keypoints}"
    
    # Filter visible keypoints (v > 0)
    visible_mask = visible_flags > 0
    if not np.any(visible_mask):
        # No visible keypoints -> OKS = 0, all per-keypoint scores are NaN
        per_keypoint_oks = np.full(num_keypoints, np.nan, dtype=np.float32)
        return 0.0, per_keypoint_oks
    
    # Compute squared distances
    distances = compute_l2_distance(pred_coords, gt_coords)
    squared_distances = distances ** 2
    
    # Scale normalization: s = sqrt(area)
    scale = np.sqrt(area)
    
    # OKS formula: exp(-d² / (2 * s² * κ²))
    # Note: 2 * s² * κ² is the variance in the exponential
    variance = 2 * (scale ** 2) * (sigmas ** 2)
    oks_per_keypoint = np.exp(-squared_distances / variance)
    
    # Set non-visible keypoints to NaN in the per-keypoint array
    per_keypoint_oks = oks_per_keypoint.copy()
    per_keypoint_oks[~visible_mask] = np.nan
    
    # Average over visible keypoints only
    oks = np.sum(oks_per_keypoint[visible_mask]) / np.sum(visible_mask)
    
    return float(oks), per_keypoint_oks


def compute_nll(
    gmm_model: Any,
    gt_coords: npt.NDArray[np.float32]
) -> float:
    """
    Compute Negative Log-Likelihood (NLL) of ground truth under GMM model.
    
    Mathematical Definition:
    -----------------------
    The log-likelihood of a Gaussian Mixture Model is:
    
        log P(x) = log(Σⱼ wⱼ 𝒩(x | μⱼ, Σⱼ))
    
    NLL is simply the negative of this value:
        NLL = -log P(x)
    
    Lower NLL indicates that the GMM assigns higher probability to the GT,
    suggesting better model fit and calibration.
    
    Args:
        gmm_model: A fitted GaussianMixture object (sklearn API) with .score_samples() method.
        gt_coords: Ground truth coordinates, shape (num_keypoints, 2) or (2,) for single point.
    
    Returns:
        Negative log-likelihood (scalar). Lower is better.
    
    Example:
        >>> from sklearn.mixture import GaussianMixture
        >>> gmm = GaussianMixture(n_components=2)
        >>> samples = np.random.randn(100, 2)
        >>> gmm.fit(samples)
        >>> gt = np.array([[0.0, 0.0]])
        >>> nll = compute_nll(gmm, gt)
        >>> # nll ≈ 2.0 (depends on fit)
    
    Note:
        - Handles reshaping for single point (2,) -> (1, 2)
        - For multiple keypoints, computes average NLL
    """
    # Ensure 2D array (N, 2)
    if gt_coords.ndim == 1:
        gt_coords = gt_coords.reshape(1, -1)
    
    assert gt_coords.shape[1] == 2, f"Expected (N, 2) coordinates, got {gt_coords.shape}"
    
    # Use sklearn's .score_samples() which returns log P(X)
    log_likelihood = gmm_model.score_samples(gt_coords)
    
    # Return negative log-likelihood (average over all points)
    nll = -np.mean(log_likelihood)
    
    return float(nll)


def compute_entropy(
    weights: npt.NDArray[np.float32],
    covariances: npt.NDArray[np.float32]
) -> float:
    """
    Compute entropy of a Gaussian Mixture Model as a measure of uncertainty.
    
    Mathematical Definition:
    -----------------------
    For a GMM with weights wⱼ and covariances Σⱼ, there's no closed-form entropy.
    We use the upper bound approximation:
    
        H(GMM) ≈ -Σⱼ wⱼ log(wⱼ) + Σⱼ wⱼ · H(𝒩ⱼ)
    
    Where the entropy of a single Gaussian is:
        H(𝒩) = (D/2) log(2πe) + (1/2) log(det(Σ))
        
    For D=2 (2D keypoints):
        H(𝒩) = log(2πe) + (1/2) log(det(Σ))
    
    Args:
        weights: Array of shape (n_components,) with mixture weights (must sum to 1).
        covariances: Array of shape (n_components, 2, 2) with covariance matrices.
    
    Returns:
        Entropy (nats). Higher entropy indicates more uncertainty.
    
    Example:
        >>> weights = np.array([0.7, 0.3])
        >>> cov1 = np.eye(2) * 0.1
        >>> cov2 = np.eye(2) * 10.0
        >>> covariances = np.array([cov1, cov2])
        >>> entropy = compute_entropy(weights, covariances)
        >>> # High entropy because of the second wide Gaussian
    
    Note:
        - Uses np.linalg.slogdet for numerical stability
        - Handles edge cases: weights near 0, singular matrices
    """
    # Normalize weights (should already be normalized, but ensure)
    weights = weights / np.sum(weights)
    
    # Entropy of mixing distribution: H(W) = -Σ w log w
    # Handle w=0 case (0 log 0 = 0 by convention)
    weight_entropy = -np.sum(weights * np.log(weights + 1e-10))
    
    # Entropy of each Gaussian component: H(N) = log(2πe) + 0.5 log(det(Σ))
    # For 2D: H = log(2πe) + 0.5 log(det)
    D = 2  # Dimension
    log_2pi_e = np.log(2 * np.pi * np.e)
    
    gaussian_entropies = []
    for cov in covariances:
        # Use slogdet for numerical stability (avoids overflow/underflow)
        sign, logdet = np.linalg.slogdet(cov)
        
        # Handle singular or ill-conditioned matrices
        if sign <= 0 or logdet < -20:  # Very small or negative determinant
            # Apply stronger regularization
            cov_reg = cov + 1e-3 * np.eye(D)
            sign, logdet = np.linalg.slogdet(cov_reg)
        
        # Ensure entropy is non-negative by clipping logdet
        # Minimum entropy for numerical stability
        logdet = max(logdet, -20.0)
        
        h_gaussian = (D / 2) * log_2pi_e + 0.5 * logdet
        gaussian_entropies.append(h_gaussian)
    
    gaussian_entropies = np.array(gaussian_entropies)
    
    # Total entropy (upper bound approximation)
    total_entropy = weight_entropy + np.sum(weights * gaussian_entropies)
    
    # Ensure non-negative entropy (handle numerical edge cases)
    # Entropy should theoretically be non-negative, but numerical issues
    # with near-singular matrices can cause small negative values
    total_entropy = max(total_entropy, 1e-6)
    
    return float(total_entropy)


def compute_covariance_volume(
    covariances: npt.NDArray[np.float32],
    weights: Optional[npt.NDArray[np.float32]] = None,
    method: str = "dominant"
) -> float:
    """
    Compute generalized variance (volume) of covariance matrices.
    
    Mathematical Definition:
    -----------------------
    The generalized variance is the determinant of the covariance matrix:
    
        V = √(det(Σ))
    
    For 2D covariances, this represents the area of the uncertainty ellipse.
    
    For multiple components (GMM), we have two options:
        - "dominant": Volume of the component with highest weight
        - "weighted": Weighted average of volumes
    
    Args:
        covariances: Array of shape (n_components, 2, 2) or (2, 2) for single matrix.
        weights: Optional array of shape (n_components,) for weighted average.
                Required if method="weighted".
        method: "dominant" (use max weight component) or "weighted" (average).
    
    Returns:
        Generalized variance (scalar). Higher values indicate more uncertainty.
    
    Example:
        >>> cov = np.array([[[1.0, 0.0], [0.0, 1.0]], 
        ...                 [[4.0, 0.0], [0.0, 4.0]]])
        >>> weights = np.array([0.8, 0.2])
        >>> volume = compute_covariance_volume(cov, weights, method="weighted")
        >>> # volume ≈ 1.4 (weighted average of sqrt(1) and sqrt(16))
    """
    # Handle single covariance matrix
    if covariances.ndim == 2:
        covariances = covariances[np.newaxis, ...]
    
    assert covariances.ndim == 3, f"Expected (N, 2, 2) or (2, 2), got {covariances.shape}"
    
    # Compute volumes (sqrt of determinant)
    volumes = []
    for cov in covariances:
        det = np.linalg.det(cov)
        if det < 1e-10:  # Nearly singular
            det = 1e-10
        volumes.append(np.sqrt(det))
    
    volumes = np.array(volumes)
    
    if method == "dominant":
        if weights is None:
            # Return max volume if no weights provided
            return float(np.max(volumes))
        else:
            # Return volume of dominant component
            dominant_idx = np.argmax(weights)
            return float(volumes[dominant_idx])
    
    elif method == "weighted":
        if weights is None:
            raise ValueError("Weights required for method='weighted'")
        
        # Weighted average
        weights = weights / np.sum(weights)  # Normalize
        weighted_volume = np.sum(weights * volumes)
        return float(weighted_volume)
    
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'dominant' or 'weighted'.")


def count_active_components(
    weights: npt.NDArray[np.float32],
    threshold: float = 1e-3
) -> int:
    """
    Count number of active (non-negligible) components in a GMM.
    
    Definition:
    ----------
    Count how many mixture components have weight above a threshold:
    
        N_active = Σⱼ 𝟙(wⱼ > τ)
    
    Where τ is the threshold (default 0.001 = 0.1%).
    
    Args:
        weights: Array of shape (n_components,) with mixture weights.
        threshold: Minimum weight to consider a component "active" (default 1e-3).
    
    Returns:
        Number of active components (integer).
    
    Example:
        >>> weights = np.array([0.7, 0.25, 0.04, 0.009, 0.001])
        >>> count = count_active_components(weights, threshold=0.01)
        >>> # count = 3 (first three components > 0.01)
    
    Interpretation:
        - N_active = 1: Unimodal (single clear mode)
        - N_active = 2+: Multimodal (ambiguous, e.g., left/right confusion)
    """
    return int(np.sum(weights > threshold))


def classify_failure_mode(
    metrics_dict: Dict[str, float],
    oks_base: Optional[float] = None
) -> str:
    """
    Classify prediction into failure modes based on metrics.
    
    Categories:
    ----------
    1. **Robust**: High accuracy, low uncertainty (OKS > 0.8, entropy < 2.0)
    2. **Denoised**: Our method improves over baseline (OKS_ours > OKS_base + 0.1)
    3. **Ambiguous**: High entropy, multiple modes (entropy > 3.0, N_components > 1)
    4. **Overconfident**: Low uncertainty but wrong prediction (entropy < 1.5, OKS < 0.5)
    5. **Honest_Uncertainty**: High uncertainty and wrong (entropy > 2.5, OKS < 0.6)
    6. **Failure**: Very low accuracy (OKS < 0.3)
    
    Args:
        metrics_dict: Dictionary with keys:
            - "oks": Object Keypoint Similarity ∈ [0, 1]
            - "entropy": GMM entropy (optional)
            - "nll": Negative log-likelihood (optional)
            - "n_active_components": Number of active modes (optional)
        oks_base: Baseline OKS for comparison (optional). If provided, enables
                 "Denoised" classification.
    
    Returns:
        Category label as string.
    
    Example:
        >>> metrics = {"oks": 0.85, "entropy": 1.2, "n_active_components": 1}
        >>> classify_failure_mode(metrics)
        'Robust'
        
        >>> metrics = {"oks": 0.4, "entropy": 3.5, "n_active_components": 2}
        >>> classify_failure_mode(metrics)
        'Ambiguous'
    
    Usage:
        This function enables systematic error analysis for pose estimation systems,
        particularly useful for:
        - Dataset curation (filter failure cases)
        - Active learning (sample ambiguous cases)
        - Model diagnostics (identify overconfident predictions)
    """
    oks = metrics_dict.get("oks", 0.0)
    entropy = metrics_dict.get("entropy", 0.0)
    n_active = metrics_dict.get("n_active_components", 1)
    
    # Define thresholds
    OKS_ROBUST = 0.8
    OKS_GOOD = 0.6
    OKS_POOR = 0.5
    OKS_FAILURE = 0.3
    ENTROPY_LOW = 1.5
    ENTROPY_MID = 2.0
    ENTROPY_HIGH = 2.5
    ENTROPY_VERY_HIGH = 3.0
    DENOISED_IMPROVEMENT = 0.1
    
    # Priority order matters for correct classification
    
    # 1. Check for denoising improvement (if baseline provided)
    if oks_base is not None and oks > oks_base + DENOISED_IMPROVEMENT:
        return "Denoised"
    
    # 2. Check for complete failure
    if oks < OKS_FAILURE:
        return "Failure"
    
    # 3. Check for robust predictions
    if oks > OKS_ROBUST and entropy < ENTROPY_MID:
        return "Robust"
    
    # 4. Check for ambiguous predictions (multimodal + high entropy)
    if entropy > ENTROPY_VERY_HIGH and n_active > 1:
        return "Ambiguous"
    
    # 5. Check for overconfident predictions (low uncertainty but wrong)
    if entropy < ENTROPY_LOW and oks < OKS_POOR:
        return "Overconfident"
    
    # 6. Check for honest uncertainty (high uncertainty + moderate/low OKS)
    if entropy > ENTROPY_HIGH and oks < OKS_GOOD:
        return "Honest_Uncertainty"
    
    # 7. Default: Robust (catches edge cases)
    return "Robust"


# ============================================================================
# Legacy/Additional Metrics (Optional)
# ============================================================================

def compute_pck(
    predicted: List[RefinedKeypoint],
    ground_truth: npt.NDArray[np.float32],
    threshold: float = 0.2,
    normalize: str = "torso"
) -> float:
    """
    Compute Percentage of Correct Keypoints (PCK) metric.
    
    Definition:
    ----------
    A keypoint is "correct" if its Euclidean distance to ground truth is below
    a threshold relative to a normalizing factor (body scale):
    
        PCK@α = (1/K) Σᵢ 𝟙(||pᵢ - gᵢ|| < α · scale)
    
    Where:
        - α: Threshold multiplier (e.g., 0.2 for PCK@0.2)
        - scale: Normalizing factor (torso length, head size, or bbox diagonal)
    
    Args:
        predicted: List of predicted keypoints.
        ground_truth: Array of shape (num_keypoints, 3).
        threshold: Multiplier for the scale (e.g., 0.2 = 20% of scale).
        normalize: Scale measure - "torso" (shoulder-hip distance), 
                  "head" (head bbox diagonal), or "bbox" (person bbox diagonal).
    
    Returns:
        PCK score ∈ [0, 1] (percentage in decimal form).
    
    Normalization Strategies:
        - "torso" (MPII): Distance from neck to pelvis
        - "head" (300W face): Head bounding box diagonal
        - "bbox" (general): Person detection box diagonal
    
    Usage:
        PCK@0.2 is common for whole-body pose (MPII, COCO).
        PCK@0.05 is typical for face landmarks (300W, WFLW).
    """
    # Convert RefinedKeypoint list to numpy array
    pred_coords = np.array([[kp.x, kp.y] for kp in predicted], dtype=np.float32)
    gt_coords = ground_truth[:, :2].astype(np.float32)
    visible = ground_truth[:, 2].astype(np.int32)
    
    # Filter visible keypoints
    visible_mask = visible > 0
    if not np.any(visible_mask):
        return 0.0
    
    # Compute scale based on normalization method
    if normalize == "torso":
        # Assume COCO skeleton: shoulders (5, 6), hips (11, 12)
        if len(gt_coords) >= 13:
            shoulder_center = (gt_coords[5] + gt_coords[6]) / 2
            hip_center = (gt_coords[11] + gt_coords[12]) / 2
            scale = np.linalg.norm(shoulder_center - hip_center)
        else:
            # Fallback to bbox diagonal
            scale = np.linalg.norm(np.max(gt_coords, axis=0) - np.min(gt_coords, axis=0))
    elif normalize == "bbox":
        scale = np.linalg.norm(np.max(gt_coords, axis=0) - np.min(gt_coords, axis=0))
    else:
        raise ValueError(f"Unknown normalization: {normalize}")
    
    # Compute distances
    distances = compute_l2_distance(pred_coords, gt_coords)
    
    # Check correctness
    correct = distances[visible_mask] < (threshold * scale)
    pck = np.mean(correct)
    
    return float(pck)


def compute_pck_auc(
    predicted: List[RefinedKeypoint],
    ground_truth: npt.NDArray[np.float32],
    thresholds: Optional[npt.NDArray[np.float32]] = None
) -> float:
    """
    Compute Area Under the PCK curve (PCK-AUC or AUC metric).
    
    Mathematical Definition:
    -----------------------
    Integrate PCK over a range of thresholds:
    
        AUC = ∫₀^{α_max} PCK(α) dα
    
    Discretized as:
        AUC ≈ (1/N) Σᵢ PCK(αᵢ)
    
    Why AUC?
        - Single-number summary robust to threshold choice
        - Captures full accuracy vs. tolerance trade-off
        - More informative than PCK at a single threshold
    
    Args:
        predicted: List of predicted keypoints.
        ground_truth: Array of shape (num_keypoints, 3).
        thresholds: Array of α values (e.g., np.linspace(0, 0.5, 50)).
                   If None, defaults to [0.0, 0.05, ..., 0.5].
    
    Returns:
        AUC score (higher is better, max = 1.0).
    
    Typical Ranges:
        - State-of-the-art models: AUC > 0.6 on COCO
        - Good performance: AUC > 0.5
        - Random baseline: AUC ≈ 0.0
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 0.5, 50)
    
    pck_values = []
    for thresh in thresholds:
        pck = compute_pck(predicted, ground_truth, threshold=thresh)
        pck_values.append(pck)
    
    auc = np.mean(pck_values)
    return float(auc)


def compute_nme(
    predicted: List[RefinedKeypoint],
    ground_truth: npt.NDArray[np.float32],
    normalize: str = "interocular"
) -> float:
    """
    Compute Normalized Mean Error (NME) for facial landmark evaluation.
    
    Definition:
    ----------
    Mean Euclidean error normalized by face size:
    
        NME = (1/K) Σᵢ ||pᵢ - gᵢ|| / d
    
    Where d is the normalizing distance (interocular distance, face bbox diagonal).
    
    Args:
        predicted: List of predicted landmarks.
        ground_truth: Array of shape (num_keypoints, 3).
        normalize: "interocular" (eye center distance) or "bbox" (face box diagonal).
    
    Returns:
        NME percentage (lower is better). Typical threshold: NME < 8% is good.
    
    Usage:
        Primary metric for face alignment (300W, WFLW, COFW datasets).
    """
    pred_coords = np.array([[kp.x, kp.y] for kp in predicted], dtype=np.float32)
    gt_coords = ground_truth[:, :2].astype(np.float32)
    visible = ground_truth[:, 2].astype(np.int32)
    
    visible_mask = visible > 0
    if not np.any(visible_mask):
        return float('inf')
    
    # Compute normalizing distance
    if normalize == "interocular":
        # Assume eyes are keypoints 1 and 2 (or 0 and 1)
        if len(gt_coords) >= 2:
            eye_distance = np.linalg.norm(gt_coords[0] - gt_coords[1])
        else:
            eye_distance = 1.0  # Fallback
    elif normalize == "bbox":
        eye_distance = np.linalg.norm(np.max(gt_coords, axis=0) - np.min(gt_coords, axis=0))
    else:
        raise ValueError(f"Unknown normalization: {normalize}")
    
    # Compute distances
    distances = compute_l2_distance(pred_coords, gt_coords)
    
    # Normalized mean error
    nme = np.mean(distances[visible_mask]) / eye_distance
    
    return float(nme * 100)  # Return as percentage


def compute_uncertainty_calibration(
    predictions: List[PoseEstimationResult],
    ground_truths: List[npt.NDArray[np.float32]],
    num_bins: int = 10
) -> Dict[str, float]:
    """
    Evaluate calibration of uncertainty estimates (Expected Calibration Error).
    
    Motivation:
    ----------
    Our pipeline produces covariance matrices (σ) as uncertainty measures.
    We want to verify that predicted uncertainty correlates with actual error.
    
    Expected Calibration Error (ECE):
        ECE = Σⱼ (nⱼ/n) |acc(Bⱼ) - conf(Bⱼ)|
    
    Where predictions are binned by confidence, and we check if empirical
    accuracy matches the predicted confidence.
    
    Args:
        predictions: List of PoseEstimationResult with uncertainty info.
        ground_truths: List of ground truth arrays.
        num_bins: Number of bins for confidence discretization.
    
    Returns:
        Dictionary with:
            - "ece": Expected Calibration Error
            - "mce": Maximum Calibration Error
            - "reliability_diagram": Per-bin accuracy vs. confidence
    
    Interpretation:
        - ECE < 0.1: Well-calibrated
        - ECE > 0.2: Over/under-confident
    
    Why This Matters:
        If our covariance estimates are meaningful, low σ should predict
        low error, and vice versa. This validates our mixture model approach.
    """
    # Placeholder implementation
    # Full implementation would require detailed analysis of covariance vs error
    return {
        "ece": 0.0,
        "mce": 0.0,
        "reliability_diagram": {}
    }

"""
Data Transfer Objects and Type Definitions.

This module defines the "universal currency" for data flow across the pipeline.
All model adapters must convert their outputs to these standardized formats,
enabling framework-agnostic processing.

Design Rationale:
----------------
Different pose estimation frameworks (MMPose, Detectron2, OpenPose) produce
heterogeneous outputs. These data classes normalize representations to enable
a unified mathematical processing pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import numpy.typing as npt


from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Tuple

@dataclass
class Keypoint:
    """Agnostic representation of a single keypoint."""
    id: int
    x: float
    y: float
    confidence: float  # Ground truth visibility or confidence
    name: str          # E.g., "left_elbow"

@dataclass
class ImageSample:
    """
    Data Transfer Object (DTO) flowing through the evaluation pipeline.
    Standardizes inputs across different datasets (COCO, CrowdPose, OCHuman).
    """
    image_id: int or str
    image_path: str
    image_array: np.ndarray        # RGB image array (H, W, 3)
    bbox: Tuple[float, float, float, float] # (x, y, w, h)
    ground_truth_keypoints: Optional[List[Keypoint]] = None
    dataset_source: str = "coco"   # Dataset metadata for provenance tracking


@dataclass(frozen=True)
class StandardizedHeatmap:
    """
    Normalized heatmap representation across different pose estimation models.
    
    This is the **universal currency** of our pipeline. All model adapters must
    transform their native outputs (e.g., MMPose's PixelData, Detectron2's Instances)
    into this format.
    
    Mathematical Context:
    --------------------
    Heatmaps represent probability distributions P(k|I) over spatial locations
    for each keypoint k given image I. Shape: (num_keypoints, H, W).
    
    Attributes:
        data: Heatmap tensor of shape (num_keypoints, height, width).
              Values in [0, 1] representing location probabilities.
        original_size: (height, width) of the input image before any resizing.
                      Required for inverse affine transformations during TTA.
        confidence_map: Optional per-keypoint confidence scores of shape (num_keypoints,).
                       Some models provide this separately from heatmaps.
        scale_factor: Ratio between heatmap resolution and original image.
        offset: Spatial offset applied during preprocessing (for padding).
        reconstructed: Boolean indicating if the heatmap was reconstructed
            (True) or obtained directly from the model (False).
    
    Invariants:
        - data.ndim == 3
        - np.all((data >= 0) & (data <= 1))
        - len(original_size) == 2
    """
    data: npt.NDArray[np.float32]
    original_size: Tuple[int, int]
    confidence_map: Optional[npt.NDArray[np.float32]] = None
    scale_factor: float = 1.0
    offset: Tuple[float, float] = (0.0, 0.0)
    reconstructed: bool = False
    metadata: Optional[Dict[str, Any]] = None  # Store transformation info for decode
    
    def __post_init__(self) -> None:
        """Validate data integrity."""
        assert self.data.ndim == 3, f"Expected 3D heatmap, got shape {self.data.shape}"
        assert len(self.original_size) == 2, "original_size must be (height, width)"
        assert np.all((self.data >= 0) & (self.data <= 1)), "Heatmap values must be in [0, 1]"
        ...


@dataclass
class RefinedKeypoint:
    """
    A single keypoint with uncertainty quantification after stochastic refinement.
    
    Mathematical Formulation:
    ------------------------
    After fitting a Gaussian Mixture Model to Monte Carlo samples drawn from
    the heatmap, each keypoint is represented as:
    
        (x, y) ~ argmax_{μ} P(μ | samples)
        Σ: Covariance matrix (2×2) quantifying spatial uncertainty
    
    For bimodal distributions (e.g., left/right limb swaps), we enforce
    a shared covariance constraint (Σ₁ = Σ₂) and select the mode with
    higher mixing coefficient.
    
    Attributes:
        x: X-coordinate in original image space.
        y: Y-coordinate in original image space.
        sigma: 2×2 covariance matrix encoding uncertainty ellipse.
               Eigenvalues represent variance along principal axes.
        confidence: Model's confidence score ∈ [0, 1].
        is_outlier: Flag indicating if this keypoint was marked as outlier
                   (e.g., via Mahalanobis distance threshold).
        mixture_weight: For bimodal cases, the weight of the selected mode.
        keypoint_id: Index in the skeleton (e.g., 0=nose, 5=left_shoulder).
    
    Geometric Interpretation:
        The uncertainty ellipse can be visualized by computing eigenvectors
        of sigma. Major/minor axes lengths: sqrt(eigenvalues).
    """
    x: float
    y: float
    sigma: npt.NDArray[np.float32]  # Shape: (2, 2)
    confidence: float
    is_outlier: bool = False
    mixture_weight: float = 1.0
    keypoint_id: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate covariance matrix properties."""
        assert self.sigma.shape == (2, 2), f"Covariance must be 2×2, got {self.sigma.shape}"
        assert 0.0 <= self.confidence <= 1.0, "Confidence must be in [0, 1]"
        # Check positive semi-definite (eigenvalues ≥ 0)
        ...


@dataclass
class PoseEstimationResult:
    """
    Complete pose estimation output for a single person instance.
    
    Aggregates all keypoints along with metadata for downstream processing
    (visualization, evaluation, dataset creation).
    
    Attributes:
        keypoints: List of refined keypoints (typically 17 for COCO format).
        bbox: Optional bounding box (x1, y1, x2, y2) for the person instance.
        instance_id: Unique identifier for multi-person scenarios.
        metadata: Flexible dict for storing auxiliary information
                 (e.g., TTA statistics, fitting convergence metrics).
    """
    keypoints: List[RefinedKeypoint]
    bbox: Optional[Tuple[float, float, float, float]] = None
    instance_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    
    def to_coco_format(self) -> npt.NDArray[np.float32]:
        """
        Convert to COCO evaluation format: [x1, y1, v1, ..., xK, yK, vK].
        
        Where v_i ∈ {0: not labeled, 1: labeled but not visible, 2: labeled and visible}.
        We use v_i = 2 for non-outliers with confidence > 0.3.
        """
        ...


@dataclass
class AugmentationParams:
    """
    Parameters defining a geometric augmentation applied during TTA.
    
    Mathematical Representation:
    ---------------------------
    Each augmentation can be represented as an affine transformation matrix A:
    
        [x']   [a  b  tx] [x]
        [y'] = [c  d  ty] [y]
        [1 ]   [0  0  1 ] [1]
    
    This class stores the semantic parameters (rotation angle, scale, flip)
    which are converted to matrix form during application.
    
    Attributes:
        rotation_angle: Rotation in degrees (counter-clockwise).
        scale: Uniform scale factor (1.0 = no scaling).
        horizontal_flip: Boolean flag for mirroring across vertical axis.
        vertical_flip: Boolean flag (rarely used for pose).
        center: Rotation center (x, y) in image coordinates.
    
    Usage:
        These parameters enable **inverse transformations** to map augmented
        predictions back to the original image space for averaging.
    """
    rotation_angle: float = 0.0
    scale: float = 1.0
    horizontal_flip: bool = False
    vertical_flip: bool = False
    center: Optional[Tuple[float, float]] = None
    
    def to_matrix(self) -> npt.NDArray[np.float32]:
        """
        Construct 3×3 affine transformation matrix.
        
        Order of operations: Scale → Rotate → Flip → Translate
        """
        ...
    
    def inverse_matrix(self) -> npt.NDArray[np.float32]:
        """
        Compute inverse transformation for mapping predictions back.
        
        Critical for TTA: We apply T to images, then T⁻¹ to predictions.
        """
        ...




@dataclass
class MixtureComponent:
    """
    Single component of a Gaussian Mixture Model.
    
    Attributes:
        mean: Component mean (μₖ) of shape (D,).
        covariance: Covariance matrix (Σₖ) of shape (D, D).
        weight: Mixing coefficient (πₖ) where Σₖ πₖ = 1.
        n_samples: Number of samples currently assigned to this component.
    """
    mean: npt.NDArray[np.float32]
    covariance: npt.NDArray[np.float32]
    weight: float
    n_samples: int = 0


@dataclass
class MixtureResult:
    """
    Result container for mixture model fitting.
    
    Attributes:
        model_type: Type of model fitted ('unimodal' or 'bimodal').
        best_mean: Best estimate of the keypoint location.
        best_covariance: Uncertainty covariance matrix.
        log_likelihood: Final log-likelihood of the model.
        aic: Akaike Information Criterion.
        bic: Bayesian Information Criterion.
        n_iterations: Number of EM iterations performed.
        converged: Whether the algorithm converged.
        components: List of fitted mixture components.
        uniform_weight: Weight of the uniform (outlier) component.
    """
    model_type: str
    best_mean: npt.NDArray[np.float64]
    best_covariance: npt.NDArray[np.float64]
    log_likelihood: float
    aic: float
    bic: float
    n_iterations: int
    converged: bool
    components: List[MixtureComponent]
    uniform_weight: float

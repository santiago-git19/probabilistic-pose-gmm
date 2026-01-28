"""
Stochastic Pose Refinement Pipeline.

This is the main orchestrator that combines:
1. Test-Time Augmentation (TTA)
2. Model inference
3. Heatmap aggregation
4. Monte Carlo sampling
5. Mixture model fitting
6. Refined keypoint extraction with uncertainty

This class represents the culmination of the entire research approach,
implementing the complete CVIU paper algorithm.

Architecture Pattern: Dependency Injection
-----------------------------------------
The StochasticPoseRefiner depends on abstractions (BasePoseModel, Augmenter,
RestrictedGaussianMixture) rather than concrete implementations. This enables:
    - Testing with mock components
    - Swapping models/algorithms at runtime
    - Clear separation of concerns

Usage Example:
-------------
```python
# 1. Initialize components
model = MMPoseAdapter(config, checkpoint)
augmenter = Augmenter(types=["rotation", "flip"], num_aug=8)
gmm = RestrictedGaussianMixture(n_components=2)

# 2. Create pipeline
refiner = StochasticPoseRefiner(
    model_adapter=model,
    augmenter=augmenter,
    mixture_model=gmm,
    n_samples=500
)

# 3. Process image
image = cv2.imread("person.jpg")
result = refiner.process_image(image)

# 4. Access refined keypoints
for kpt in result.keypoints:
    print(f"({kpt.x:.1f}, {kpt.y:.1f}) ± σ={kpt.sigma}")
```
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from ..models.base import BasePoseModel
from ..core.mixture import RestrictedGaussianMixture
from ..core.sampling import sample_from_heatmap
from ..utils.types import (
    StandardizedHeatmap,
    RefinedKeypoint,
    PoseEstimationResult
)
from .tta import Augmenter


@dataclass
class RefinerConfig:
    """
    Configuration for StochasticPoseRefiner.
    
    Centralizes all hyperparameters for easy experiment management.
    Can be loaded from Hydra YAML configs.
    
    Attributes:
        n_samples: Number of Monte Carlo samples per keypoint.
        sampling_strategy: Method for drawing samples from heatmaps.
        temperature: Softmax temperature for heatmap sampling.
        use_tta: Whether to enable Test-Time Augmentation.
        aggregation_method: How to combine TTA predictions.
        outlier_detection: Enable outlier filtering via Mahalanobis distance.
        outlier_threshold: Standard deviations for outlier cutoff.
        min_confidence: Minimum heatmap confidence to process keypoint.
    """
    n_samples: int = 500
    sampling_strategy: str = "importance"
    temperature: float = 1.0
    use_tta: bool = True
    aggregation_method: str = "mean"
    outlier_detection: bool = True
    outlier_threshold: float = 3.0
    min_confidence: float = 0.3


class StochasticPoseRefiner:
    """
    Main pipeline for robust pose estimation with uncertainty quantification.
    
    This class implements the complete algorithm from the CVIU paper:
    
    Algorithm Pipeline:
    ------------------
    Input: RGB image I
    
    1. TTA Generation:
       Generate N augmented images: {T_i(I)}_{i=1}^N
    
    2. Batch Prediction:
       For each T_i(I):
           H_i = Model(T_i(I))  // Get heatmap
           H_i' = T_i^{-1}(H_i)  // Transform back to original space
    
    3. Heatmap Aggregation:
       H_final = Aggregate({H_i'}_{i=1}^N)  // e.g., mean
    
    4. Monte Carlo Sampling:
       For each keypoint k:
           S_k = {(x_j, y_j)}_{j=1}^M ~ P_k(x, y | H_final)
    
    5. Mixture Model Fitting:
       For each keypoint k:
           θ_k = fit_mixture(S_k)  // (μ_1, μ_2, Σ, π)
           (x_k, y_k), Σ_k = get_mode(θ_k)
    
    6. Outlier Detection:
       Mark keypoints with d_Mahalanobis > threshold as outliers
    
    Output: List[RefinedKeypoint] with (x, y, Σ, confidence, is_outlier)
    
    Mathematical Contributions:
    --------------------------
    1. **TTA for Variance Reduction**: Averaging N predictions reduces variance by √N
    2. **Mixture Models for Ambiguity**: Handles bimodal distributions (left/right swaps)
    3. **Shared Covariance Constraint**: Regularizes mixture for better identifiability
    4. **Uncertainty Calibration**: Covariance matrices reflect true localization error
    
    Parameters:
        model_adapter: Any model implementing BasePoseModel interface.
        augmenter: TTA augmentation generator (can be None to disable TTA).
        mixture_model: Gaussian mixture model for uncertainty estimation.
        config: RefinerConfig with hyperparameters.
    
    Attributes:
        model: The pose estimation model adapter.
        augmenter: TTA augmentation engine (optional).
        mixture_model: Statistical model for uncertainty.
        config: Configuration object.
    """
    
    def __init__(
        self,
        model_adapter: BasePoseModel,
        augmenter: Optional[Augmenter] = None,
        mixture_model: Optional[RestrictedGaussianMixture] = None,
        config: Optional[RefinerConfig] = None
    ) -> None:
        """
        Initialize the stochastic pose refiner.
        
        Dependency Injection Pattern:
            All major components are injected, enabling easy testing and swapping.
        
        Args:
            model_adapter: Pose estimation model (MMPose, Detectron2, etc.).
            augmenter: TTA engine. If None, TTA is disabled.
            mixture_model: GMM for uncertainty. If None, uses default 2-component.
            config: Hyperparameters. If None, uses default RefinerConfig.
        
        Design Decision:
            We inject dependencies rather than creating them internally.
            This follows SOLID principles and enables:
                - Unit testing with mocks
                - Runtime configuration
                - Experimentation with different components
        """
        ...
    
    def process_image(
        self,
        image: npt.NDArray[np.uint8],
        bbox: Optional[tuple] = None,
        visualize: bool = False
    ) -> PoseEstimationResult:
        """
        Process a single image and return refined pose with uncertainty.
        
        This is the main public API for the pipeline. It orchestrates all steps:
        
        Step-by-Step Execution:
        ----------------------
        1. **TTA Generation** (if enabled):
           - Generate N augmented versions of input image
           - Store augmentation parameters for inverse transforms
        
        2. **Batch Inference**:
           - Predict heatmaps for each augmented image
           - Apply inverse transforms to align predictions
        
        3. **Heatmap Aggregation**:
           - Average (or median) aligned heatmaps
           - Result: Single "consensus" heatmap
        
        4. **Monte Carlo Sampling**:
           - For each keypoint, draw M samples from aggregated heatmap
           - Samples form empirical distribution
        
        5. **Mixture Model Fitting**:
           - Fit Restricted GMM to samples
           - Extract mode (μ) and covariance (Σ)
        
        6. **Outlier Detection**:
           - Compute Mahalanobis distance for each keypoint
           - Mark outliers if distance > threshold
        
        7. **Result Packaging**:
           - Create RefinedKeypoint objects
           - Assemble into PoseEstimationResult
        
        Args:
            image: Input RGB image (H, W, 3), uint8.
            bbox: Optional person bounding box (x1, y1, x2, y2).
            visualize: If True, stores intermediate visualizations in result.metadata.
        
        Returns:
            PoseEstimationResult containing:
                - keypoints: List[RefinedKeypoint] with uncertainty
                - bbox: Input bounding box (if provided)
                - metadata: Dict with debug info (TTA stats, convergence, etc.)
        
        Raises:
            RuntimeError: If model prediction fails.
            ValueError: If heatmap has unexpected shape.
        
        Performance:
            - Without TTA: ~50ms (depends on model)
            - With TTA (N=8): ~400ms
            - Bottleneck: Model inference, not our math
        
        Example:
            ```python
            refiner = StochasticPoseRefiner(model, augmenter, gmm)
            result = refiner.process_image(image)
            
            # Access refined keypoints
            for kpt in result.keypoints:
                if not kpt.is_outlier:
                    print(f"Keypoint {kpt.keypoint_id}: ({kpt.x}, {kpt.y})")
                    print(f"  Uncertainty: {np.linalg.det(kpt.sigma):.3f}")
            ```
        """
        ...
    
    def _generate_tta_predictions(
        self,
        image: npt.NDArray[np.uint8],
        bbox: Optional[tuple]
    ) -> List[StandardizedHeatmap]:
        """
        Generate predictions for all TTA augmentations.
        
        Orchestration:
            1. augmenter.augment_image() yields augmented images
            2. model.predict() runs on each augmented image
            3. augmenter.inverse_transform_heatmap() aligns predictions
            4. Return list of aligned heatmaps
        
        Args:
            image: Input image.
            bbox: Optional bounding box.
        
        Returns:
            List of StandardizedHeatmap objects (all in original image space).
        """
        ...
    
    def _aggregate_heatmaps(
        self,
        heatmaps: List[StandardizedHeatmap]
    ) -> StandardizedHeatmap:
        """
        Aggregate multiple heatmaps into a single consensus heatmap.
        
        Delegates to self.augmenter.aggregate_heatmaps() with configured method.
        
        Args:
            heatmaps: List of heatmaps to aggregate.
        
        Returns:
            Single aggregated heatmap.
        """
        ...
    
    def _sample_keypoints(
        self,
        heatmap: StandardizedHeatmap
    ) -> List[npt.NDArray[np.float32]]:
        """
        Draw Monte Carlo samples for each keypoint from the heatmap.
        
        For each keypoint k:
            S_k = {(x_j, y_j)}_{j=1}^M ~ P_k(x, y)
        
        Args:
            heatmap: Aggregated heatmap (num_keypoints, H, W).
        
        Returns:
            List of sample arrays, one per keypoint.
            Each array has shape (n_samples, 2).
        """
        ...
    
    def _fit_mixture_per_keypoint(
        self,
        samples: npt.NDArray[np.float32],
        keypoint_id: int
    ) -> RefinedKeypoint:
        """
        Fit mixture model to samples and extract refined keypoint.
        
        Process:
            1. Check if samples are sufficient (> 10)
            2. Fit RestrictedGaussianMixture
            3. Extract mode (dominant component)
            4. Detect outliers via Mahalanobis distance
            5. Package into RefinedKeypoint
        
        Args:
            samples: Monte Carlo samples (n_samples, 2).
            keypoint_id: Index of this keypoint in skeleton.
        
        Returns:
            RefinedKeypoint with mean, covariance, confidence, outlier flag.
        
        Edge Cases:
            - Too few samples: Fall back to simple mean/covariance
            - Singular covariance: Add regularization
            - Non-convergence: Use k-means result
        """
        ...
    
    def _detect_outliers(
        self,
        keypoints: List[RefinedKeypoint]
    ) -> List[RefinedKeypoint]:
        """
        Mark outlier keypoints based on Mahalanobis distance.
        
        For each keypoint, compute distance to its own distribution:
            d_M = √[(x - μ)^T Σ^{-1} (x - μ)]
        
        Mark as outlier if d_M > threshold.
        
        Args:
            keypoints: List of RefinedKeypoint objects.
        
        Returns:
            Same list with is_outlier flags updated.
        
        Rationale:
            Outliers typically indicate:
            - Occluded keypoints (model hallucinating)
            - Out-of-distribution poses
            - Augmentation artifacts
        """
        ...
    
    def process_batch(
        self,
        images: List[npt.NDArray[np.uint8]],
        bboxes: Optional[List[tuple]] = None,
        show_progress: bool = True
    ) -> List[PoseEstimationResult]:
        """
        Process multiple images in a batch.
        
        Efficiently processes a list of images by:
        1. Batching model inference (if model supports it)
        2. Parallelizing independent operations
        3. Showing progress bar
        
        Args:
            images: List of input images.
            bboxes: Optional list of bounding boxes (one per image).
            show_progress: Whether to display tqdm progress bar.
        
        Returns:
            List of PoseEstimationResult objects (one per image).
        
        Performance:
            Batched inference is typically 2-3× faster than sequential processing.
        """
        ...
    
    def evaluate_uncertainty_calibration(
        self,
        predictions: List[PoseEstimationResult],
        ground_truths: List[npt.NDArray[np.float32]]
    ) -> Dict[str, float]:
        """
        Evaluate how well predicted uncertainty matches actual error.
        
        Computes Expected Calibration Error (ECE) to validate that:
            - Small σ → Small actual error
            - Large σ → Large actual error
        
        Args:
            predictions: List of our pose predictions with uncertainty.
            ground_truths: List of ground truth keypoint arrays.
        
        Returns:
            Dict with calibration metrics:
                - "ece": Expected Calibration Error
                - "correlation": Correlation between σ and actual error
        
        Why This Matters:
            If calibration is poor, our uncertainty estimates are not meaningful.
            Well-calibrated uncertainty enables:
                - Confidence-based filtering
                - Active learning (label uncertain samples)
                - Risk-aware decision making
        """
        ...

"""
Test-Time Augmentation (TTA) for Robust Pose Estimation.

This module implements geometric augmentations and inverse transformations
to enable ensemble predictions at test time.

TTA Methodology:
---------------
1. Generate N augmented versions of input image:
   - Horizontal flips
   - Rotations (±15°)
   - Scales (0.85× to 1.15×)
   - Combinations

2. Predict on each augmented image

3. Apply inverse transforms to predictions

4. Aggregate (average) transformed predictions

Mathematical Intuition:
----------------------
For a model f and augmentation T, we have:

    E[f(T(x))] ≈ f(x)  (model equivariance assumption)

By averaging over multiple augmentations:

    y_TTA = (1/N) Σᵢ T_i^(-1) [f(T_i(x))]

We reduce prediction variance at the cost of N× inference time.

References:
    - "Test-Time Augmentation for Pose Estimation" (various papers)
    - Krizhevsky et al. "ImageNet Classification with Deep CNNs" (original TTA)
"""

from typing import List, Tuple, Optional, Generator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import cv2

from ..utils.types import AugmentationParams, StandardizedHeatmap
from ..utils.geometry import (
    construct_affine_matrix,
    apply_affine_to_heatmap,
    get_inverse_transform
)


class Augmenter:
    """
    Test-Time Augmentation generator for pose estimation.
    
    Generates a set of geometric augmentations (transformations) and provides
    methods to apply them to images and inverse-transform predictions.
    
    Design Principles:
    -----------------
    1. **Reproducibility**: Seeded random generation for deterministic augmentations
    2. **Invertibility**: Every augmentation T has inverse T^(-1)
    3. **Composability**: Augmentations can be chained (rotation + scale + flip)
    4. **Pose-Awareness**: Special handling for horizontal flip (swap left/right)
    
    Parameters:
        augmentation_types: List of augmentation names to use.
                           Options: ["horizontal_flip", "rotation", "scale", "identity"]
        rotation_range: (min_angle, max_angle) in degrees for rotation sampling.
        scale_range: (min_scale, max_scale) for uniform scale sampling.
        num_augmentations: Total number of augmented views to generate.
        include_original: If True, always include identity transform (no aug).
        random_state: Random seed for reproducibility.
    
    Attributes:
        augmentations: List of AugmentationParams defining all transforms.
    
    Example:
        ```python
        aug = Augmenter(
            augmentation_types=["horizontal_flip", "rotation"],
            rotation_range=(-15, 15),
            num_augmentations=8
        )
        
        # Generate augmented images
        for aug_img, params in aug.augment_image(image):
            heatmap = model.predict(aug_img)
            # Later: apply inverse to heatmap
        ```
    """
    
    def __init__(
        self,
        augmentation_types: List[str],
        rotation_range: Tuple[float, float] = (-15.0, 15.0),
        scale_range: Tuple[float, float] = (0.85, 1.15),
        num_augmentations: int = 8,
        include_original: bool = True,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize TTA augmenter with specified parameters.
        
        Args:
            augmentation_types: List of augmentation types to apply.
                Available: ["horizontal_flip", "vertical_flip", "rotation", "scale", "identity"]
            rotation_range: Min/max rotation angles in degrees (counter-clockwise).
            scale_range: Min/max uniform scale factors (> 1 = zoom in).
            num_augmentations: Target number of augmentations to generate.
            include_original: If True, ensures identity transform is included.
            random_state: Random seed for reproducible augmentation sampling.
        
        Raises:
            ValueError: If augmentation_types contains unsupported augmentations.
        """
        ...
    
    def generate_augmentations(self) -> List[AugmentationParams]:
        """
        Generate a diverse set of augmentation parameters.
        
        Strategy:
        --------
        1. If include_original=True, add identity transform
        2. For each augmentation type:
           - Sample random parameters (angle, scale)
           - Optionally combine with other types
        3. Ensure num_augmentations total
        
        Returns:
            List of AugmentationParams objects defining all transforms.
        
        Design Consideration:
            We balance diversity (cover parameter space) with practicality
            (avoid extreme transforms that hurt accuracy).
        """
        ...
    
    def augment_image(
        self,
        image: npt.NDArray[np.uint8]
    ) -> Generator[Tuple[npt.NDArray[np.uint8], AugmentationParams], None, None]:
        """
        Apply all augmentations to an image, yielding (augmented_image, params).
        
        This is a generator to enable lazy evaluation and memory efficiency.
        Augmented images are not stored all at once.
        
        Args:
            image: Input RGB image (H, W, 3), uint8.
        
        Yields:
            Tuple of:
                - Augmented image (same shape and type as input)
                - AugmentationParams used for this augmentation
        
        Implementation:
            For each AugmentationParams in self.augmentations:
            1. Construct affine matrix from params
            2. Apply cv2.warpAffine to image
            3. Yield result with params (for later inverse transform)
        
        Example:
            ```python
            for aug_img, params in augmenter.augment_image(image):
                heatmap_aug = model.predict(aug_img)
                heatmap_orig = inverse_transform_heatmap(heatmap_aug, params)
            ```
        """
        ...
    
    def inverse_transform_heatmap(
        self,
        heatmap: StandardizedHeatmap,
        aug_params: AugmentationParams,
        output_size: Tuple[int, int]
    ) -> StandardizedHeatmap:
        """
        Apply inverse augmentation to transform heatmap back to original space.
        
        Critical for TTA: We apply T to images, predict, then apply T^(-1) to
        predictions to align all predictions in the same coordinate frame.
        
        Mathematical Form:
        -----------------
        If we predict on T(image), we get heatmap_aug.
        To align with original image space:
        
            heatmap_orig = T^(-1)(heatmap_aug)
        
        Args:
            heatmap: Prediction on augmented image.
            aug_params: AugmentationParams that were applied to image.
            output_size: (height, width) of target space (typically original image size).
        
        Returns:
            Transformed heatmap in original image coordinate frame.
        
        Special Handling:
            - Horizontal flip: Swap left/right keypoints (use model.flip_pairs)
            - Rotation: Rotate heatmaps by -angle
            - Scale: Rescale heatmaps by 1/scale
        """
        ...
    
    def aggregate_heatmaps(
        self,
        heatmaps: List[StandardizedHeatmap],
        method: str = "mean"
    ) -> StandardizedHeatmap:
        """
        Aggregate multiple heatmaps from different augmentations into one.
        
        Aggregation Strategy:
        --------------------
        After inverse-transforming all predictions to original space, we combine
        them to get a robust final prediction.
        
        Methods:
            - "mean": Element-wise average (default, most common)
            - "median": Element-wise median (robust to outliers)
            - "max": Element-wise maximum (preserves peaks)
            - "weighted_mean": Weight by confidence scores
        
        Args:
            heatmaps: List of StandardizedHeatmap objects (all same size).
            method: Aggregation method name.
        
        Returns:
            Single StandardizedHeatmap representing ensemble prediction.
        
        Mathematical Form (mean):
            H_final(x, y, k) = (1/N) Σᵢ H_i(x, y, k)
        
        Variance Reduction:
            By averaging, we reduce variance by factor of √N:
                Var[H_final] = Var[H_i] / N
        
        Implementation Notes:
            - Ensure all heatmaps have same spatial dimensions
            - Renormalize after aggregation to maintain probability distribution
        """
        ...
    
    def visualize_augmentations(
        self,
        image: npt.NDArray[np.uint8],
        save_path: Optional[str] = None
    ) -> npt.NDArray[np.uint8]:
        """
        Create a grid visualization of all augmented images.
        
        Useful for debugging and understanding the augmentation strategy.
        
        Args:
            image: Input image to augment.
            save_path: Optional path to save visualization (e.g., "aug_grid.png").
        
        Returns:
            Grid image showing all augmentations side-by-side.
        """
        ...


def sample_rotation_angle(
    min_angle: float,
    max_angle: float,
    rng: np.random.RandomState
) -> float:
    """
    Sample a rotation angle from a specified range.
    
    Strategy: Uniform sampling from [min_angle, max_angle].
    
    Alternative Strategies:
        - Discrete: Sample from fixed set {-15, -10, -5, 0, 5, 10, 15}
        - Gaussian: Sample from N(0, σ²) truncated to range
    
    Args:
        min_angle: Minimum rotation in degrees.
        max_angle: Maximum rotation in degrees.
        rng: NumPy random state for reproducibility.
    
    Returns:
        Sampled angle in degrees.
    """
    ...


def sample_scale_factor(
    min_scale: float,
    max_scale: float,
    rng: np.random.RandomState
) -> float:
    """
    Sample a scale factor from a specified range.
    
    Sampling Strategy:
        Log-uniform distribution to ensure symmetric treatment of zoom in/out:
        
            log(scale) ~ Uniform(log(min_scale), log(max_scale))
        
    Why Log-Uniform?
        - 0.5× (zoom out) and 2× (zoom in) have equal probability
        - With uniform sampling, small scales would be under-represented
    
    Args:
        min_scale: Minimum scale factor (e.g., 0.85).
        max_scale: Maximum scale factor (e.g., 1.15).
        rng: NumPy random state.
    
    Returns:
        Sampled scale factor.
    """
    ...


def apply_flip_to_keypoint_indices(
    heatmap: npt.NDArray[np.float32],
    flip_pairs: List[Tuple[int, int]]
) -> npt.NDArray[np.float32]:
    """
    Swap left/right keypoint indices after horizontal flip.
    
    When an image is flipped horizontally, left becomes right and vice versa.
    For pose heatmaps, we must swap the channels corresponding to symmetric
    keypoints.
    
    Example:
        Before flip: Channel 5 = left_shoulder, Channel 6 = right_shoulder
        After flip + swap: Channel 5 = right_shoulder, Channel 6 = left_shoulder
    
    Args:
        heatmap: Heatmap array (num_keypoints, H, W).
        flip_pairs: List of (left_idx, right_idx) tuples to swap.
    
    Returns:
        Heatmap with swapped channels (same shape).
    
    Implementation:
        ```python
        swapped = heatmap.copy()
        for left, right in flip_pairs:
            swapped[left], swapped[right] = heatmap[right], heatmap[left]
        ```
    """
    ...

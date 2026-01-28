"""
Geometric transformations for image and heatmap manipulation.

This module provides NumPy-based implementations of affine transformations,
coordinate mappings, and spatial warping operations. All transformations
are invertible to support Test-Time Augmentation workflows.

Mathematical Foundation:
-----------------------
We represent transformations in homogeneous coordinates using 3×3 matrices:

    T = [R  t]  where R ∈ ℝ^{2×2} (rotation + scale), t ∈ ℝ^2 (translation)
        [0  1]

Key Operations:
    1. Forward: x' = T @ x (apply augmentation to image)
    2. Inverse: x = T⁻¹ @ x' (map predictions back to original space)
"""

from typing import Tuple, List

import numpy as np
import numpy.typing as npt

from .types import StandardizedHeatmap, AugmentationParams


def construct_affine_matrix(
    rotation: float,
    scale: float,
    translation: Tuple[float, float],
    center: Tuple[float, float]
) -> npt.NDArray[np.float32]:
    """
    Construct 3×3 affine transformation matrix for 2D geometric operations.
    
    Mathematical Formulation:
    ------------------------
    The transformation is composed as: T_center⁻¹ @ R(θ) @ S @ T_center @ T_translate
    
    Where:
        - T_center: Translate rotation center to origin
        - S: Uniform scaling matrix diag(s, s, 1)
        - R(θ): Rotation matrix with angle θ
        - T_translate: Final translation
    
    Args:
        rotation: Angle in degrees (counter-clockwise positive).
        scale: Uniform scale factor (> 1 zooms in, < 1 zooms out).
        translation: (tx, ty) translation vector.
        center: (cx, cy) point around which rotation/scaling occurs.
    
    Returns:
        3×3 affine matrix in homogeneous coordinates.
    
    Example:
        >>> # 45° rotation around image center with 1.2× zoom
        >>> T = construct_affine_matrix(45.0, 1.2, (0, 0), (128, 128))
        >>> # Apply to point: T @ [x, y, 1]
    """
    ...


def apply_affine_to_heatmap(
    heatmap: StandardizedHeatmap,
    transform: npt.NDArray[np.float32],
    output_size: Tuple[int, int]
) -> StandardizedHeatmap:
    """
    Apply affine transformation to heatmap tensor with proper interpolation.
    
    Implementation Strategy:
    -----------------------
    Unlike image warping, heatmap transformation requires careful handling:
    1. Use bilinear interpolation to preserve probability distributions
    2. Renormalize after transformation to maintain ∑P(x,y) ≈ 1 per keypoint
    3. Handle boundary conditions (zeros outside valid region)
    
    Why Not Standard cv2.warpAffine?
        - Heatmaps represent probability densities, not pixel intensities
        - Need to preserve peak locations accurately (sub-pixel precision)
        - Must maintain statistical properties after warping
    
    Args:
        heatmap: Input heatmap in StandardizedHeatmap format.
        transform: 3×3 affine matrix (or 2×3 for OpenCV compatibility).
        output_size: (height, width) of the output heatmap.
    
    Returns:
        Transformed heatmap with updated metadata (scale_factor, offset).
    
    Mathematical Note:
        For inverse TTA, we apply T⁻¹ to predictions:
            heatmap_original = apply_affine_to_heatmap(heatmap_aug, T_inv, original_size)
    """
    ...


def transform_keypoints(
    keypoints: npt.NDArray[np.float32],
    transform: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Apply affine transformation to keypoint coordinates.
    
    Efficient batch operation for transforming multiple keypoints:
        [x', y'] = (T @ [x, y, 1]^T)[:2]
    
    Args:
        keypoints: Array of shape (N, 2) containing (x, y) coordinates.
        transform: 3×3 affine matrix or 2×3 matrix.
    
    Returns:
        Transformed coordinates of shape (N, 2).
    
    Usage in TTA:
        # Forward: augment image
        img_aug = apply_transform(img, T)
        # Predict on augmented image
        kpts_aug = model.predict(img_aug)
        # Inverse: map back to original space
        kpts_orig = transform_keypoints(kpts_aug, T_inv)
    """
    ...


def create_flip_matrix(
    image_width: int,
    horizontal: bool = True
) -> npt.NDArray[np.float32]:
    """
    Create affine matrix for horizontal/vertical flipping.
    
    Mathematical Form:
        Horizontal flip: x' = w - x, y' = y
        Matrix: [[-1, 0, w], [0, 1, 0], [0, 0, 1]]
    
    Args:
        image_width: Width of the image (height for vertical flip).
        horizontal: If True, flip horizontally; else vertically.
    
    Returns:
        3×3 flip transformation matrix.
    
    Important for Pose:
        Horizontal flip requires swapping left/right keypoint labels
        (e.g., left_shoulder ↔ right_shoulder). Handle in adapter layer.
    """
    ...


def get_inverse_transform(
    transform: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Compute inverse of an affine transformation matrix.
    
    For affine matrices T = [A | t], the inverse is:
                            [0 | 1]
    
        T⁻¹ = [A⁻¹  | -A⁻¹t]
              [0    |   1  ]
    
    Args:
        transform: 3×3 or 2×3 affine matrix.
    
    Returns:
        Inverse transformation matrix (same shape as input).
    
    Numerical Stability:
        Uses NumPy's linalg.inv with built-in conditioning checks.
        Raises LinAlgError if matrix is singular (det ≈ 0).
    """
    ...


def rotate_covariance_matrix(
    sigma: npt.NDArray[np.float32],
    rotation: float
) -> npt.NDArray[np.float32]:
    """
    Rotate a 2×2 covariance matrix by a given angle.
    
    Mathematical Derivation:
    -----------------------
    Under rotation R(θ), a covariance matrix transforms as:
        Σ' = R(θ) @ Σ @ R(θ)^T
    
    This is required when propagating uncertainty through geometric transforms.
    
    Args:
        sigma: 2×2 covariance matrix.
        rotation: Rotation angle in degrees.
    
    Returns:
        Rotated 2×2 covariance matrix.
    
    Application in TTA:
        When averaging predictions from rotated images, we must rotate
        their covariance matrices back to the canonical orientation.
    """
    ...

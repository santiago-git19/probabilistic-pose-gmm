"""
Robust Pose Estimation with Test-Time Adaptation and Uncertainty Quantification.

This package implements a research framework for human pose estimation that combines:
1. Test-Time Adaptation (TTA) via geometric augmentations
2. Uncertainty quantification via Restricted Gaussian Mixture Models
3. Outlier-robust refinement via stochastic optimization

Architecture Philosophy:
----------------------
- **Separation of Concerns**: Pure mathematical core (NumPy) isolated from 
  deep learning adapters (PyTorch/MMPose).
- **Dependency Injection**: Model adapters are interchangeable via abstract interfaces.
- **Data Contracts**: Standardized representations enable cross-framework compatibility.

Mathematical Foundation:
-----------------------
Based on the CVIU paper's approach to handle bimodal distributions in pose estimation,
particularly for symmetric keypoints (left/right limb swaps) using a two-component
Gaussian mixture with shared covariance constraint.

References:
----------
- Restricted Gaussian Mixture Models for pose refinement
- Distribution-Aware Representation of Keypoints (DARK)
- Test-Time Augmentation for robust predictions

"""

__version__ = "0.1.0"

from typing import List

__all__: List[str] = [
    "core",
    "models",
    "pipeline",
    "utils",
]

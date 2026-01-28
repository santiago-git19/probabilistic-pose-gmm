"""
Pipeline: Orchestration of TTA and Stochastic Refinement.

This subpackage contains the high-level components that tie together:
1. Test-Time Augmentation (TTA) generation
2. Model inference on augmented images
3. Heatmap aggregation and inverse transformation
4. Monte Carlo sampling and mixture model fitting
5. Refined keypoint extraction with uncertainty

Core Classes:
------------
- Augmenter: Generates geometric augmentations
- StochasticPoseRefiner: Main pipeline orchestrator
"""

from typing import List

__all__: List[str] = ["tta", "refiner"]

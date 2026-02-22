"""
Pipeline: Orchestration of TTA and Stochastic Refinement.

This subpackage contains the high-level components that tie together:
1. Test-Time Augmentation (TTA) generation
2. Scale TTA with confidence-weighted aggregation
3. Model inference on augmented images
4. Heatmap aggregation and inverse transformation
5. Monte Carlo sampling and mixture model fitting
6. Refined keypoint extraction with uncertainty

Core Modules:
-------------
- tta:       Flip and photometric augmentation engine
- scale_tta: Multi-scale bounding-box augmentation with FOV-aware fusion
- refiner:   Main pipeline orchestrator
"""

from typing import List

__all__: List[str] = ["tta", "scale_tta", "refiner"]

"""
Model Adapter Layer: Bridge between Deep Learning Frameworks and Math Core.

This subpackage implements the **Adapter Pattern** to enable framework-agnostic
pose estimation. Different models (MMPose, Detectron2, MediaPipe) are wrapped
with a unified interface that produces StandardizedHeatmap objects.

Design Philosophy:
-----------------
Following SOLID principles, particularly:
- **Dependency Inversion**: High-level pipeline depends on abstractions (BasePoseModel),
  not concrete implementations (MMPoseAdapter).
- **Open/Closed**: Add new models by extending BasePoseModel, without modifying pipeline.

Architecture:
------------
    User Code
        ↓
    StochasticPoseRefiner (pipeline)
        ↓
    BasePoseModel (abstract interface)
        ↓ ↓ ↓
    MMPoseAdapter  Detectron2Adapter  MediaPipeAdapter
        ↓               ↓                    ↓
    MMPose Lib    Detectron2 Lib      MediaPipe Lib

Benefits:
    1. Test pipeline with mock adapters (no GPU needed)
    2. Switch models via config change (no code modification)
    3. Isolate framework bugs from core algorithms
"""

from typing import List

__all__: List[str] = ["base", "adapters"]

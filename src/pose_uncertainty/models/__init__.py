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
    EvaluationRunner (pipeline)
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

Usage:
------
    >>> from pose_uncertainty.models import create_model_adapter, list_available_models
    >>> 
    >>> # List available models
    >>> print(list_available_models())
    ['resnet50', 'hrnet_w32', 'vitpose_small']
    >>> 
    >>> # Create model adapter
    >>> model = create_model_adapter("hrnet_w32", device="cuda")
    >>> 
    >>> # Run prediction
    >>> heatmap = model.predict(image, bbox=(x, y, w, h))
"""

from .base import (
    BasePoseModel,
    MockPoseModel,
    COCO_KEYPOINT_NAMES,
    COCO_FLIP_PAIRS,
)
from .adapters import (
    MMPoseAdapter,
    ModelPathResolver,
    create_model_adapter,
    list_available_models,
)

__all__ = [
    # Base classes
    "BasePoseModel",
    "MockPoseModel",
    # Adapters
    "MMPoseAdapter",
    "ModelPathResolver",
    # Factory functions
    "create_model_adapter",
    "list_available_models",
    # Constants
    "COCO_KEYPOINT_NAMES",
    "COCO_FLIP_PAIRS",
]

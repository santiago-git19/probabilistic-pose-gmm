"""
Abstract Base Class for Pose Estimation Models.

Defines the contract that all model adapters must implement. This enables
dependency injection and framework-agnostic pipeline design.

Design Pattern: Adapter + Strategy
----------------------------------
The BasePoseModel acts as both:
1. **Adapter**: Translates framework-specific APIs to our StandardizedHeatmap format
2. **Strategy**: Allows runtime swapping of different pose estimation algorithms

Key Responsibilities:
--------------------
1. Image preprocessing (resize, normalize, pad)
2. Model inference (forward pass)
3. Output standardization (heatmap extraction, denormalization)
4. Metadata preservation (scales, offsets for inverse transforms)
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import numpy.typing as npt

from ..utils.types import StandardizedHeatmap


class BasePoseModel(ABC):
    """
    Abstract interface for pose estimation models.
    
    All concrete adapters (MMPose, Detectron2, etc.) must implement this interface
    to be compatible with the StochasticPoseRefiner pipeline.
    
    Contract:
    --------
    1. **Input**: Raw image (numpy array, H×W×3, uint8, RGB)
    2. **Output**: StandardizedHeatmap with metadata for inverse transforms
    3. **Transparency**: Handle all preprocessing internally (user provides raw images)
    4. **Immutability**: Models should be stateless (no side effects between calls)
    
    Attributes:
        model_name: Identifier for the model (e.g., "rtmpose-m", "vitpose-b").
        input_size: (height, width) expected by the model.
        num_keypoints: Number of keypoints detected (e.g., 17 for COCO).
        device: Computation device ("cpu", "cuda", "mps").
    
    Example Implementation:
        ```python
        class MyAdapter(BasePoseModel):
            def predict(self, image):
                # 1. Preprocess
                img_tensor = self._preprocess(image)
                # 2. Inference
                heatmap = self.model(img_tensor)
                # 3. Standardize
                return StandardizedHeatmap(
                    data=heatmap.cpu().numpy(),
                    original_size=image.shape[:2],
                    ...
                )
        ```
    """
    
    def __init__(
        self,
        model_name: str,
        input_size: Tuple[int, int],
        num_keypoints: int,
        device: str = "cuda"
    ) -> None:
        """
        Initialize base model attributes.
        
        Args:
            model_name: Human-readable model identifier.
            input_size: (height, width) for model input (e.g., (256, 192)).
            num_keypoints: Number of keypoints detected by model.
            device: Computation device string.
        """
        ...
    
    @abstractmethod
    def predict(
        self,
        image: npt.NDArray[np.uint8],
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> StandardizedHeatmap:
        """
        Predict pose heatmaps from a raw image.
        
        This is the core method that every adapter must implement. It should:
        1. Preprocess image (resize/pad to self.input_size)
        2. Run model inference
        3. Extract heatmaps from model output
        4. Package into StandardizedHeatmap with inverse transform metadata
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format, uint8.
            bbox: Optional bounding box (x1, y1, x2, y2) for cropping.
                 If None, use entire image.
        
        Returns:
            StandardizedHeatmap with:
                - data: Heatmaps of shape (num_keypoints, h, w)
                - original_size: (H, W) of input image
                - scale_factor: Resize ratio applied
                - offset: (dx, dy) padding offset
        
        Guarantees:
            - Output heatmaps are normalized: values ∈ [0, 1]
            - Spatial dimensions match model output (not original image)
            - Metadata enables reconstruction of original-space coordinates
        
        Implementation Notes:
            - Use self._preprocess() helper for common preprocessing
            - Handle batched vs. single-image models internally
            - Catch framework-specific exceptions and re-raise with context
        """
        ...
    
    @abstractmethod
    def warmup(self, iterations: int = 3) -> None:
        """
        Warm up model with dummy inputs for accurate timing.
        
        Many deep learning frameworks (CUDA, TensorRT) perform lazy initialization
        and kernel compilation. This method:
        1. Creates dummy input matching self.input_size
        2. Runs forward passes (without gradients)
        3. Synchronizes GPU to ensure compilation is complete
        
        Args:
            iterations: Number of warmup iterations (default: 3).
        
        Usage:
            Call once after model initialization to ensure stable inference times.
        """
        ...
    
    def preprocess(
        self,
        image: npt.NDArray[np.uint8],
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> Tuple[npt.NDArray[np.float32], Dict[str, Any]]:
        """
        Preprocess image with padding to maintain aspect ratio.
        
        Standard preprocessing pipeline:
        1. Crop to bbox (if provided)
        2. Resize to self.input_size with aspect ratio preservation
        3. Pad to square with reflection padding
        4. Normalize (ImageNet stats or model-specific)
        5. Convert to CHW format for PyTorch
        
        Args:
            image: Raw RGB image (H, W, 3), uint8.
            bbox: Optional crop region (x1, y1, x2, y2).
        
        Returns:
            Tuple of:
                - Preprocessed image (C, H, W), float32, normalized
                - Metadata dict with keys: "scale", "offset", "original_size"
        
        Metadata Usage:
            Stored in StandardizedHeatmap for inverse transformation:
            ```python
            x_original = (x_heatmap - offset_x) / scale
            ```
        """
        ...
    
    def postprocess(
        self,
        heatmap: npt.NDArray[np.float32],
        metadata: Dict[str, Any]
    ) -> StandardizedHeatmap:
        """
        Package raw heatmap into StandardizedHeatmap with metadata.
        
        This method wraps the model output in our data contract format,
        attaching the information needed for coordinate space conversions.
        
        Args:
            heatmap: Raw heatmap from model (num_keypoints, h, w).
            metadata: Preprocessing metadata from self.preprocess().
        
        Returns:
            StandardizedHeatmap instance ready for pipeline consumption.
        """
        ...
    
    @property
    @abstractmethod
    def keypoint_names(self) -> List[str]:
        """
        Return list of keypoint names in order.
        
        Example for COCO format:
            ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
             "left_shoulder", "right_shoulder", ...]
        
        Used for:
            - Visualization (labeling keypoints)
            - Left/right swap detection (horizontal flip augmentation)
            - Evaluation (matching predictions to ground truth)
        """
        ...
    
    @property
    @abstractmethod
    def flip_pairs(self) -> List[Tuple[int, int]]:
        """
        Return indices of left/right symmetric keypoint pairs.
        
        Example for COCO:
            [(1, 2),   # left_eye ↔ right_eye
             (3, 4),   # left_ear ↔ right_ear
             (5, 6),   # left_shoulder ↔ right_shoulder
             ...]
        
        Usage:
            When applying horizontal flip augmentation, swap these keypoint indices:
            ```python
            flipped_heatmap = heatmap.copy()
            for left, right in model.flip_pairs:
                flipped_heatmap[[left, right]] = flipped_heatmap[[right, left]]
            ```
        """
        ...


class MockPoseModel(BasePoseModel):
    """
    Mock model for testing without GPU dependencies.
    
    Generates synthetic heatmaps with controllable properties (unimodal, bimodal,
    noisy) for unit testing the mathematical pipeline.
    
    Features:
        - Instant "prediction" (no actual model)
        - Deterministic output for reproducible tests
        - Configurable number/type of peaks
    
    Example:
        ```python
        # Test bimodal case (left/right ambiguity)
        mock_model = MockPoseModel(mode="bimodal", noise_level=0.1)
        heatmap = mock_model.predict(dummy_image)
        # heatmap will have two peaks with configurable separation
        ```
    """
    
    def __init__(
        self,
        mode: str = "unimodal",
        noise_level: float = 0.05,
        **kwargs: Any
    ) -> None:
        """
        Initialize mock model.
        
        Args:
            mode: Heatmap type - "unimodal", "bimodal", "uniform", "noisy".
            noise_level: Gaussian noise std (as fraction of peak height).
            **kwargs: Additional args passed to BasePoseModel.__init__.
        """
        ...
    
    def predict(
        self,
        image: npt.NDArray[np.uint8],
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> StandardizedHeatmap:
        """Generate synthetic heatmap based on self.mode."""
        ...
    
    def warmup(self, iterations: int = 3) -> None:
        """No-op for mock model."""
        pass
    
    @property
    def keypoint_names(self) -> List[str]:
        """COCO keypoint names."""
        ...
    
    @property
    def flip_pairs(self) -> List[Tuple[int, int]]:
        """COCO flip pairs."""
        ...

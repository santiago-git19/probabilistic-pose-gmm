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


# COCO keypoint names (standard 17-point format)
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# COCO flip pairs for horizontal augmentation (left_idx, right_idx)
COCO_FLIP_PAIRS = [
    (1, 2),   # left_eye ↔ right_eye
    (3, 4),   # left_ear ↔ right_ear
    (5, 6),   # left_shoulder ↔ right_shoulder
    (7, 8),   # left_elbow ↔ right_elbow
    (9, 10),  # left_wrist ↔ right_wrist
    (11, 12), # left_hip ↔ right_hip
    (13, 14), # left_knee ↔ right_knee
    (15, 16), # left_ankle ↔ right_ankle
]


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
        self._model_name = model_name
        self._input_size = input_size
        self._num_keypoints = num_keypoints
        self._device = device
    
    @property
    def model_name(self) -> str:
        """Model identifier string."""
        return self._model_name
    
    @property
    def input_size(self) -> Tuple[int, int]:
        """Expected input size (height, width)."""
        return self._input_size
    
    @property
    def num_keypoints(self) -> int:
        """Number of keypoints detected."""
        return self._num_keypoints
    
    @property
    def device(self) -> str:
        """Computation device."""
        return self._device
    
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
        raise NotImplementedError("Subclasses must implement predict()")
    
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
        raise NotImplementedError("Subclasses must implement warmup()")
    
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
        import cv2
        
        original_size = (image.shape[0], image.shape[1])
        
        # Step 1: Crop to bbox if provided
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            # Ensure bounds are within image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            image = image[y1:y2, x1:x2]
        
        # Step 2: Calculate resize ratio to fit input_size while preserving aspect ratio
        target_h, target_w = self.input_size
        src_h, src_w = image.shape[:2]
        
        scale = min(target_h / src_h, target_w / src_w)
        new_h, new_w = int(src_h * scale), int(src_w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Step 3: Pad to target size (center padding)
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        
        # Step 4: Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        normalized = (padded.astype(np.float32) / 255.0 - mean) / std
        
        # Step 5: Convert to CHW format
        chw = normalized.transpose(2, 0, 1)
        
        metadata = {
            "scale": scale,
            "offset": (pad_left, pad_top),
            "original_size": original_size,
            "crop_bbox": bbox
        }
        
        return chw, metadata
    
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
        # Ensure heatmap is normalized to [0, 1]
        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        if heatmap_max > heatmap_min:
            normalized_heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            normalized_heatmap = heatmap
        
        return StandardizedHeatmap(
            data=normalized_heatmap.astype(np.float32),
            original_size=metadata["original_size"],
            scale_factor=metadata["scale"],
            offset=metadata["offset"]
        )
    
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
        raise NotImplementedError("Subclasses must implement keypoint_names property")
    
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
        raise NotImplementedError("Subclasses must implement flip_pairs property")


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
        input_size: Tuple[int, int] = (64, 48),
        num_keypoints: int = 17,
        **kwargs: Any
    ) -> None:
        """
        Initialize mock model.
        
        Args:
            mode: Heatmap type - "unimodal", "bimodal", "uniform", "noisy".
            noise_level: Gaussian noise std (as fraction of peak height).
            input_size: Output heatmap size (h, w).
            num_keypoints: Number of keypoints to generate.
            **kwargs: Additional args passed to BasePoseModel.__init__.
        """
        super().__init__(
            model_name=f"mock_{mode}",
            input_size=input_size,
            num_keypoints=num_keypoints,
            device="cpu"
        )
        self.mode = mode
        self.noise_level = noise_level
        self._rng = np.random.default_rng(42)  # Deterministic seed
    
    def predict(
        self,
        image: npt.NDArray[np.uint8],
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> StandardizedHeatmap:
        """Generate synthetic heatmap based on self.mode."""
        h, w = self.input_size
        heatmap = np.zeros((self.num_keypoints, h, w), dtype=np.float32)
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        for k in range(self.num_keypoints):
            if self.mode == "unimodal":
                # Single Gaussian at center
                cx, cy = w // 2, h // 2
                sigma = min(h, w) / 8
                heatmap[k] = np.exp(-((x_coords - cx)**2 + (y_coords - cy)**2) / (2 * sigma**2))
                
            elif self.mode == "bimodal":
                # Two Gaussians (simulates left/right ambiguity)
                sigma = min(h, w) / 10
                cx1, cy1 = w // 3, h // 2
                cx2, cy2 = 2 * w // 3, h // 2
                g1 = np.exp(-((x_coords - cx1)**2 + (y_coords - cy1)**2) / (2 * sigma**2))
                g2 = np.exp(-((x_coords - cx2)**2 + (y_coords - cy2)**2) / (2 * sigma**2))
                heatmap[k] = 0.5 * g1 + 0.5 * g2
                
            elif self.mode == "uniform":
                heatmap[k] = np.ones((h, w), dtype=np.float32) / (h * w)
                
            elif self.mode == "noisy":
                # Noisy unimodal
                cx, cy = w // 2 + self._rng.integers(-5, 6), h // 2 + self._rng.integers(-5, 6)
                sigma = min(h, w) / 8
                heatmap[k] = np.exp(-((x_coords - cx)**2 + (y_coords - cy)**2) / (2 * sigma**2))
        
        # Add noise
        if self.noise_level > 0:
            noise = self._rng.normal(0, self.noise_level, heatmap.shape).astype(np.float32)
            heatmap = np.clip(heatmap + noise, 0, 1)
        
        # Normalize each keypoint heatmap
        for k in range(self.num_keypoints):
            max_val = heatmap[k].max()
            if max_val > 0:
                heatmap[k] /= max_val
        
        return StandardizedHeatmap(
            data=heatmap,
            original_size=image.shape[:2] if image is not None else (256, 192),
            scale_factor=1.0,
            offset=(0.0, 0.0)
        )
    
    def warmup(self, iterations: int = 3) -> None:
        """No-op for mock model."""
        pass
    
    @property
    def keypoint_names(self) -> List[str]:
        """COCO keypoint names."""
        return COCO_KEYPOINT_NAMES
    
    @property
    def flip_pairs(self) -> List[Tuple[int, int]]:
        """COCO flip pairs."""
        return COCO_FLIP_PAIRS

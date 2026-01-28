"""
Concrete Model Adapters for Popular Pose Estimation Frameworks.

This module implements adapters for:
1. MMPose (OpenMMLab's pose estimation library)
2. Detectron2 (Facebook's detection framework)
3. MediaPipe (Google's lightweight pose)

Each adapter inherits from BasePoseModel and implements framework-specific
initialization, inference, and output conversion logic.

Implementation Strategy:
-----------------------
- Lazy imports (import mmpose only when MMPoseAdapter is instantiated)
- Exception wrapping (convert framework errors to our custom exceptions)
- Config validation (ensure checkpoint matches expected architecture)
"""

from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from .base import BasePoseModel
from ..utils.types import StandardizedHeatmap


class MMPoseAdapter(BasePoseModel):
    """
    Adapter for OpenMMLab's MMPose framework.
    
    MMPose is a state-of-the-art pose estimation library supporting models like:
    - RTMPose (real-time models)
    - ViTPose (Vision Transformer-based)
    - HRNet (high-resolution networks)
    - TopDown/BottomUp approaches
    
    This adapter handles:
    1. Config loading from .py files
    2. Checkpoint loading with device mapping
    3. Inference with MMPose's pipeline API
    4. Heatmap extraction from various output formats
    
    Key Challenges:
    --------------
    - MMPose uses custom data structures (PixelData, InstanceData)
    - Different models output heatmaps in different formats
    - Some models use SimCC (regression) instead of heatmaps
    
    Args:
        config_path: Path to MMPose config file (.py).
        checkpoint_path: Path to model weights (.pth).
        device: Device for inference ("cpu", "cuda", "cuda:0").
        backend: Inference backend - "pytorch" (default) or "onnxruntime".
    
    Attributes:
        model: MMPose model instance (loaded from checkpoint).
        cfg: MMPose Config object with model architecture.
        pipeline: Data preprocessing pipeline from config.
    
    Example:
        ```python
        adapter = MMPoseAdapter(
            config_path="configs/rtmpose-m_8xb256-420e_coco-256x192.py",
            checkpoint_path="checkpoints/rtmpose-m_coco.pth",
            device="cuda"
        )
        
        image = cv2.imread("person.jpg")
        heatmap = adapter.predict(image)
        # heatmap is now in StandardizedHeatmap format
        ```
    
    Supported Models:
        ✓ RTMPose (all variants: s, m, l, x)
        ✓ ViTPose (base, large, huge)
        ✓ HRNet (w32, w48)
        ✗ SimCC-based models (requires different handling)
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        backend: str = "pytorch"
    ) -> None:
        """
        Initialize MMPose adapter with config and checkpoint.
        
        Args:
            config_path: Path to .py config file defining model architecture.
            checkpoint_path: Path to .pth checkpoint file with trained weights.
            device: Torch device string ("cpu", "cuda", "cuda:0").
            backend: Inference backend - "pytorch" or "onnxruntime" for speed.
        
        Raises:
            FileNotFoundError: If config or checkpoint doesn't exist.
            RuntimeError: If model architecture mismatches checkpoint.
        
        Initialization Steps:
            1. Import mmpose and mmcv (lazy import for optional dependency)
            2. Load config using mmcv.Config.fromfile()
            3. Build model from config.model specification
            4. Load checkpoint state_dict into model
            5. Set model to eval mode and move to device
            6. Extract metadata (num_keypoints, input_size, keypoint_names)
        """
        ...
    
    def predict(
        self,
        image: npt.NDArray[np.uint8],
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> StandardizedHeatmap:
        """
        Run pose estimation on an image using MMPose.
        
        Pipeline:
        --------
        1. Preprocess: Resize and pad image to model input size
        2. To Tensor: Convert to PyTorch tensor (C, H, W)
        3. Inference: Forward pass through model
        4. Extract: Get heatmaps from PixelData output
        5. Normalize: Ensure heatmap values in [0, 1]
        6. Package: Wrap in StandardizedHeatmap with metadata
        
        Args:
            image: RGB image (H, W, 3), uint8 format.
            bbox: Optional person bounding box (x1, y1, x2, y2).
                 If provided, crops image before inference.
        
        Returns:
            StandardizedHeatmap with:
                - data: (num_keypoints, h, w) heatmap tensor
                - original_size: (H, W) from input image
                - scale_factor: Resize ratio applied
                - offset: Padding offset (dx, dy)
        
        MMPose Output Format:
            result.pred_instances contains:
                - keypoints: (N, K, 2) - x, y coordinates
                - keypoint_scores: (N, K) - confidence per keypoint
                - heatmaps: (K, H, W) - probability maps (if available)
        
        Note:
            Some MMPose models (e.g., SimCC) don't output heatmaps directly.
            In this case, we reconstruct Gaussian heatmaps from keypoint coords.
        """
        ...
    
    def warmup(self, iterations: int = 3) -> None:
        """
        Warm up MMPose model with dummy forward passes.
        
        CUDA kernel compilation happens on first call, causing slowdown.
        This method runs dummy inference to trigger compilation.
        
        Args:
            iterations: Number of warmup passes (default: 3).
        
        Procedure:
            1. Create dummy input matching self.input_size
            2. Run forward passes without gradients
            3. Synchronize CUDA to ensure completion
        """
        ...
    
    @property
    def keypoint_names(self) -> List[str]:
        """
        Extract keypoint names from MMPose config.
        
        MMPose configs specify dataset metadata in:
            cfg.dataset.keypoint_info
        
        Returns:
            List of keypoint names in order (e.g., ["nose", "left_eye", ...]).
        """
        ...
    
    @property
    def flip_pairs(self) -> List[Tuple[int, int]]:
        """
        Extract left/right flip pairs from MMPose config.
        
        Located in: cfg.dataset.flip_pairs
        
        Returns:
            List of (left_idx, right_idx) tuples.
        """
        ...
    
    def _extract_heatmaps(
        self,
        result: Any
    ) -> npt.NDArray[np.float32]:
        """
        Extract heatmaps from MMPose result object.
        
        Handles different output formats:
        1. Direct heatmap output (TopDown models)
        2. PixelData with heatmaps attribute
        3. SimCC output (reconstruct Gaussians from coords)
        
        Args:
            result: MMPose prediction result (PoseDataSample).
        
        Returns:
            Heatmap array (num_keypoints, h, w), normalized to [0, 1].
        
        Raises:
            ValueError: If heatmaps cannot be extracted (unsupported format).
        """
        ...
    
    def _reconstruct_gaussian_heatmap(
        self,
        keypoints: npt.NDArray[np.float32],
        scores: npt.NDArray[np.float32],
        heatmap_size: Tuple[int, int],
        sigma: float = 2.0
    ) -> npt.NDArray[np.float32]:
        """
        Reconstruct heatmaps as Gaussians centered at keypoint coordinates.
        
        Used for SimCC models that output coordinates directly.
        
        Mathematical Form:
            H_k(x, y) = s_k · exp(-[(x - x_k)² + (y - y_k)²] / (2σ²))
        
        Args:
            keypoints: Coordinates array (num_keypoints, 2).
            scores: Confidence scores (num_keypoints,).
            heatmap_size: (height, width) of output heatmap.
            sigma: Gaussian std dev in pixels.
        
        Returns:
            Reconstructed heatmap (num_keypoints, h, w).
        """
        ...


class Detectron2Adapter(BasePoseModel):
    """
    Adapter for Facebook's Detectron2 pose estimation.
    
    Detectron2 provides:
    - Keypoint R-CNN (detection + pose)
    - DensePose (dense correspondences)
    - Panoptic segmentation with keypoints
    
    This adapter wraps Detectron2's DefaultPredictor API for pose estimation.
    
    Key Differences from MMPose:
        - Outputs Instances object (not PixelData)
        - Integrated person detection (no separate bbox needed)
        - Heatmaps not exposed directly (must reconstruct or extract from RoI heads)
    
    Args:
        config_file: Path to Detectron2 YAML config.
        checkpoint_url: URL or path to model weights.
        device: Device string.
        score_threshold: Minimum detection confidence (default: 0.7).
    
    Example:
        ```python
        adapter = Detectron2Adapter(
            config_file="COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
            checkpoint_url="detectron2://COCO-Keypoints/...",
            device="cuda"
        )
        ```
    """
    
    def __init__(
        self,
        config_file: str,
        checkpoint_url: str,
        device: str = "cuda",
        score_threshold: float = 0.7
    ) -> None:
        """Initialize Detectron2 adapter."""
        ...
    
    def predict(
        self,
        image: npt.NDArray[np.uint8],
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> StandardizedHeatmap:
        """Run Detectron2 prediction and extract heatmaps."""
        ...
    
    def warmup(self, iterations: int = 3) -> None:
        """Warm up Detectron2 model."""
        ...
    
    @property
    def keypoint_names(self) -> List[str]:
        """COCO keypoint names (Detectron2 uses COCO format)."""
        ...
    
    @property
    def flip_pairs(self) -> List[Tuple[int, int]]:
        """COCO flip pairs."""
        ...


class MediaPipeAdapter(BasePoseModel):
    """
    Adapter for Google's MediaPipe Pose.
    
    MediaPipe provides lightweight, real-time pose estimation optimized for:
    - Mobile deployment (TensorFlow Lite)
    - Video streaming (temporal smoothing)
    - Full-body pose (33 landmarks including hands, face)
    
    Key Features:
        - Extremely fast (>30 FPS on CPU)
        - No heatmaps (direct coordinate regression)
        - Built-in temporal smoothing for video
    
    Limitations:
        - No native heatmap output (we reconstruct)
        - Fixed model (no custom training)
        - Lower accuracy than state-of-the-art
    
    Args:
        model_complexity: 0 (lite), 1 (full), or 2 (heavy).
        static_image_mode: If True, treats each image independently.
        device: "cpu" (MediaPipe doesn't support GPU via Python API).
    
    Example:
        ```python
        adapter = MediaPipeAdapter(model_complexity=1)
        heatmap = adapter.predict(image)
        ```
    """
    
    def __init__(
        self,
        model_complexity: int = 1,
        static_image_mode: bool = True,
        device: str = "cpu"
    ) -> None:
        """Initialize MediaPipe adapter."""
        ...
    
    def predict(
        self,
        image: npt.NDArray[np.uint8],
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> StandardizedHeatmap:
        """Run MediaPipe prediction and reconstruct heatmaps."""
        ...
    
    def warmup(self, iterations: int = 3) -> None:
        """Warm up MediaPipe (minimal effect on CPU)."""
        ...
    
    @property
    def keypoint_names(self) -> List[str]:
        """MediaPipe keypoint names (33 landmarks)."""
        ...
    
    @property
    def flip_pairs(self) -> List[Tuple[int, int]]:
        """MediaPipe flip pairs for symmetric landmarks."""
        ...

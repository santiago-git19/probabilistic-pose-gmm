"""
Concrete Model Adapters for Popular Pose Estimation Frameworks.

This module implements adapters for:
1. MMPose (OpenMMLab's pose estimation library)

Each adapter inherits from BasePoseModel and implements framework-specific
initialization, inference, and output conversion logic.

Implementation Strategy:
-----------------------
- Lazy imports (import mmpose only when MMPoseAdapter is instantiated)
- Exception wrapping (convert framework errors to our custom exceptions)
- Config validation (ensure checkpoint matches expected architecture)
- Path handling (solve MMPose's relative path issues)
"""

from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import os
import logging

import numpy as np
import numpy.typing as npt

from .base import BasePoseModel, COCO_KEYPOINT_NAMES, COCO_FLIP_PAIRS
from ..utils.types import StandardizedHeatmap

# Configure logging
logger = logging.getLogger(__name__)


class ModelPathResolver:
    """
    Centralized path resolution for model configs and checkpoints.
    
    This class solves the common problem of MMPose requiring specific
    working directory context for relative path resolution in configs.
    
    Design Pattern: Strategy for path resolution
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize path resolver.
        
        Args:
            project_root: Root of the project. If None, auto-detected.
        """
        if project_root is None:
            # Auto-detect: look for pyproject.toml or models/mmpose
            current = Path(__file__).resolve()
            for parent in current.parents:
                if (parent / "pyproject.toml").exists():
                    project_root = parent
                    break
            if project_root is None:
                project_root = Path.cwd()
        
        self.project_root = Path(project_root)
        self.mmpose_root = self.project_root / "models" / "mmpose"
        self.weights_dir = self.project_root / "models" / "weights"
        self.configs_dir = self.mmpose_root / "configs"
    
    def resolve_config(self, config_path: str) -> Path:
        """
        Resolve config path to absolute path.
        
        Args:
            config_path: Relative path from configs dir, or absolute path.
        
        Returns:
            Absolute path to config file.
        
        Raises:
            FileNotFoundError: If config doesn't exist.
        """
        path = Path(config_path)
        
        # If already absolute and exists, return it
        if path.is_absolute() and path.exists():
            return path
        
        # Try relative to configs dir
        full_path = self.configs_dir / config_path
        if full_path.exists():
            return full_path
        
        # Try relative to mmpose root
        full_path = self.mmpose_root / config_path
        if full_path.exists():
            return full_path
        
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            f"Searched in:\n"
            f"  - {self.configs_dir / config_path}\n"
            f"  - {self.mmpose_root / config_path}"
        )
    
    def resolve_checkpoint(self, checkpoint_path: str) -> Path:
        """
        Resolve checkpoint path to absolute path.
        
        Args:
            checkpoint_path: Filename or path to checkpoint.
        
        Returns:
            Absolute path to checkpoint file.
        
        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
        """
        path = Path(checkpoint_path)
        
        # If already absolute and exists, return it
        if path.is_absolute() and path.exists():
            return path
        
        # Try in weights directory
        full_path = self.weights_dir / checkpoint_path
        if full_path.exists():
            return full_path
        
        # Try as-is (relative to cwd)
        if path.exists():
            return path.resolve()
        
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Searched in:\n"
            f"  - {self.weights_dir / checkpoint_path}\n"
            f"  - {path.resolve()}"
        )


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
    
    Key Challenges Solved:
    ---------------------
    - MMPose uses custom data structures (PixelData, InstanceData)
    - Different models output heatmaps in different formats
    - Working directory context switching for relative path resolution
    
    Args:
        config_path: Path to MMPose config file (.py).
        checkpoint_path: Path to model weights (.pth).
        device: Device for inference ("cpu", "cuda", "cuda:0").
        project_root: Optional project root for path resolution.
    
    Example:
        ```python
        adapter = MMPoseAdapter(
            config_path="body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py",
            checkpoint_path="hrnet_w32.pth",
            device="cuda"
        )
        
        image = cv2.imread("person.jpg")
        heatmap = adapter.predict(image, bbox=(x1, y1, x2, y2))
        ```
    """
    
    # Model registry: maps model names to their config/checkpoint
    MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
        "resnet50": {
            "config": "body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py",
            "checkpoint": "resnet50.pth",
            "input_size": (256, 192)
        },
        "hrnet_w32": {
            "config": "body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py",
            "checkpoint": "hrnet_w32.pth",
            "input_size": (256, 192)
        },
        "vitpose_small": {
            "config": "body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py",
            "checkpoint": "vitpose_small_mmpose.pth",
            "input_size": (256, 192)
        }
    }
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        project_root: Optional[Path] = None
    ) -> None:
        """
        Initialize MMPose adapter with config and checkpoint.
        
        Args:
            config_path: Path to .py config file defining model architecture.
            checkpoint_path: Path to .pth checkpoint file with trained weights.
            device: Torch device string ("cpu", "cuda", "cuda:0").
            project_root: Optional project root for path resolution.
        
        Raises:
            FileNotFoundError: If config or checkpoint doesn't exist.
            RuntimeError: If model architecture mismatches checkpoint.
            ImportError: If MMPose is not installed.
        """
        # Initialize path resolver
        self._path_resolver = ModelPathResolver(project_root)
        
        # Resolve paths
        self._config_path = self._path_resolver.resolve_config(config_path)
        self._checkpoint_path = self._path_resolver.resolve_checkpoint(checkpoint_path)
        
        logger.info(f"Resolved config: {self._config_path}")
        logger.info(f"Resolved checkpoint: {self._checkpoint_path}")
        
        # Lazy import MMPose
        try:
            from mmpose.apis import init_model
            from mmpose.utils import register_all_modules
        except ImportError as e:
            raise ImportError(
                "MMPose is not installed. Please install it with:\n"
                "  pip install mmpose\n"
                "Or via poetry:\n"
                "  poetry install"
            ) from e
        
        # Register all MMPose modules
        register_all_modules()
        
        # Context switching: MMPose configs use relative imports
        original_cwd = os.getcwd()
        try:
            os.chdir(str(self._path_resolver.mmpose_root))
            logger.debug(f"Changed working directory to: {self._path_resolver.mmpose_root}")
            
            # Initialize model
            self._model = init_model(
                str(self._config_path),
                str(self._checkpoint_path),
                device=device
            )
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        finally:
            os.chdir(original_cwd)
            logger.debug(f"Restored working directory to: {original_cwd}")
        
        # Extract model metadata
        self._cfg = self._model.cfg
        input_size = self._extract_input_size()
        num_keypoints = self._extract_num_keypoints()
        model_name = self._extract_model_name()
        
        # Initialize base class
        super().__init__(
            model_name=model_name,
            input_size=input_size,
            num_keypoints=num_keypoints,
            device=device
        )
        
        # Cache for keypoint names
        self._keypoint_names_cache: Optional[List[str]] = None
        self._flip_pairs_cache: Optional[List[Tuple[int, int]]] = None
        
        logger.info(
            f"Initialized {model_name} adapter:\n"
            f"  - Input size: {input_size}\n"
            f"  - Num keypoints: {num_keypoints}\n"
            f"  - Device: {device}"
        )
    
    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        device: str = "cuda",
        project_root: Optional[Path] = None
    ) -> "MMPoseAdapter":
        """
        Factory method to create adapter from registered model name.
        
        This provides a cleaner API than specifying config/checkpoint paths.
        
        Args:
            model_name: One of the registered model names (e.g., "hrnet_w32").
            device: Computation device.
            project_root: Optional project root.
        
        Returns:
            Initialized MMPoseAdapter.
        
        Raises:
            ValueError: If model_name is not registered.
        
        Example:
            ```python
            adapter = MMPoseAdapter.from_model_name("hrnet_w32", device="cuda")
            ```
        """
        if model_name not in cls.MODEL_REGISTRY:
            available = ", ".join(cls.MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model: {model_name}. Available models: {available}"
            )
        
        model_info = cls.MODEL_REGISTRY[model_name]
        return cls(
            config_path=model_info["config"],
            checkpoint_path=model_info["checkpoint"],
            device=device,
            project_root=project_root
        )
    
    def predict(
        self,
        image: npt.NDArray[np.uint8],
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> StandardizedHeatmap:
        """
        Run pose estimation on an image using MMPose.
        
        Pipeline:
        --------
        1. Convert bbox format if provided
        2. Run direct forward pass to get heatmaps from model head
        3. Extract heatmaps before decoding
        4. Package in StandardizedHeatmap
        
        Args:
            image: RGB image (H, W, 3), uint8 format.
            bbox: Optional person bounding box (x, y, w, h) in COCO format.
                 If provided, crops image before inference.
        
        Returns:
            StandardizedHeatmap with:
                - data: (num_keypoints, h, w) heatmap tensor
                - original_size: (H, W) from input image
                - scale_factor: Resize ratio applied
                - offset: Padding offset (dx, dy)
        
        Note:
            This method uses direct model forward pass to access heatmaps
            before they are decoded into keypoint coordinates.
        """
        original_size = (image.shape[0], image.shape[1])
        
        # Forward pass with heatmap extraction
        try:
            heatmap, metadata = self._forward_with_heatmaps(image, bbox)
            
            return StandardizedHeatmap(
                data=heatmap,
                original_size=original_size,
                scale_factor=metadata.get("scale_factor", 1.0),
                offset=metadata.get("offset", (0.0, 0.0)),
                confidence_map=metadata.get("confidence_map", None),
                reconstructed=False  # Heatmap was obtained directly from the model
            )
        
        except Exception as e:
            logger.warning(f"Failed to extract heatmaps directly: {e}")
            logger.info("Falling back to coordinate-based reconstruction")
            
            # Fallback: Use inference_topdown and reconstruct from coordinates
            from mmpose.apis import inference_topdown
            
            # Convert bbox format
            if bbox is not None:
                x, y, w, h = bbox
                bbox_xyxy = np.array([[x, y, x + w, y + h]])
            else:
                img_h, img_w = image.shape[:2]
                bbox_xyxy = np.array([[0, 0, img_w, img_h]])
            
            # Run inference
            results = inference_topdown(self._model, image, bboxes=bbox_xyxy)
            
            if len(results) == 0:
                logger.warning("No pose detected in image")
                heatmap_size = self._get_heatmap_size()
                empty_heatmap = np.zeros(
                    (self.num_keypoints, heatmap_size[0], heatmap_size[1]),
                    dtype=np.float32
                )
                return StandardizedHeatmap(
                    data=empty_heatmap,
                    original_size=original_size,
                    scale_factor=1.0,
                    offset=(0.0, 0.0)
                )
            
            # Extract/reconstruct heatmaps
            result = results[0]
            heatmap, metadata = self._extract_heatmaps(result, original_size, bbox)
            
            return StandardizedHeatmap(
                data=heatmap,
                original_size=original_size,
                scale_factor=metadata.get("scale_factor", 1.0),
                offset=metadata.get("offset", (0.0, 0.0)),
                confidence_map=metadata.get("confidence_map", None),
                reconstructed=True  # Heatmap was reconstructed
            )
    
    def predict_keypoints(
        self,
        image: npt.NDArray[np.uint8],
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Get keypoint coordinates directly (without heatmaps).
        
        Useful for quick inference when uncertainty quantification is not needed.
        
        Args:
            image: RGB image (H, W, 3), uint8.
            bbox: Optional bounding box (x, y, w, h) in COCO format.
        
        Returns:
            Tuple of:
                - keypoints: (num_keypoints, 2) array of (x, y) coordinates
                - scores: (num_keypoints,) confidence scores
        """
        from mmpose.apis import inference_topdown
        
        # Convert bbox format
        if bbox is not None:
            x, y, w, h = bbox
            bbox_xyxy = np.array([[x, y, x + w, y + h]])
        else:
            img_h, img_w = image.shape[:2]
            bbox_xyxy = np.array([[0, 0, img_w, img_h]])
        
        results = inference_topdown(self._model, image, bboxes=bbox_xyxy)
        
        if len(results) == 0:
            return (
                np.zeros((self.num_keypoints, 2), dtype=np.float32),
                np.zeros(self.num_keypoints, dtype=np.float32)
            )
        
        pred_instances = results[0].pred_instances
        keypoints = pred_instances.keypoints[0]  # (K, 2)
        scores = pred_instances.keypoint_scores[0]  # (K,)
        
        return keypoints.astype(np.float32), scores.astype(np.float32)
    
    def warmup(self, iterations: int = 3) -> None:
        """
        Warm up MMPose model with dummy forward passes.
        
        CUDA kernel compilation happens on first call, causing slowdown.
        This method runs dummy inference to trigger compilation.
        
        Args:
            iterations: Number of warmup passes (default: 3).
        """
        import torch
        
        logger.info(f"Warming up model with {iterations} iterations...")
        
        # Create dummy image
        h, w = self.input_size
        dummy_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        
        with torch.no_grad():
            for i in range(iterations):
                _ = self.predict(dummy_image)
        
        # Synchronize CUDA
        if "cuda" in self.device:
            torch.cuda.synchronize()
        
        logger.info("Warmup complete")
    
    @property
    def keypoint_names(self) -> List[str]:
        """
        Extract keypoint names from MMPose config.
        
        Returns:
            List of keypoint names in order (e.g., ["nose", "left_eye", ...]).
        """
        if self._keypoint_names_cache is not None:
            return self._keypoint_names_cache
        
        # Try to extract from config
        try:
            dataset_info = self._cfg.get("dataset_info", {})
            keypoint_info = dataset_info.get("keypoint_info", {})
            
            if keypoint_info:
                # Sort by keypoint ID and extract names
                names = [None] * len(keypoint_info)
                for name, info in keypoint_info.items():
                    idx = info.get("id", -1)
                    if 0 <= idx < len(names):
                        names[idx] = name
                
                if all(n is not None for n in names):
                    self._keypoint_names_cache = names
                    return names
        except Exception as e:
            logger.debug(f"Could not extract keypoint names from config: {e}")
        
        # Default to COCO keypoints
        self._keypoint_names_cache = COCO_KEYPOINT_NAMES
        return COCO_KEYPOINT_NAMES
    
    @property
    def flip_pairs(self) -> List[Tuple[int, int]]:
        """
        Extract left/right flip pairs from MMPose config.
        
        Returns:
            List of (left_idx, right_idx) tuples.
        """
        if self._flip_pairs_cache is not None:
            return self._flip_pairs_cache
        
        # Try to extract from config
        try:
            dataset_info = self._cfg.get("dataset_info", {})
            skeleton_info = dataset_info.get("skeleton_info", {})
            
            # Look for flip pairs in dataset info
            flip_pairs = dataset_info.get("flip_pairs", None)
            if flip_pairs:
                self._flip_pairs_cache = [tuple(pair) for pair in flip_pairs]
                return self._flip_pairs_cache
        except Exception as e:
            logger.debug(f"Could not extract flip pairs from config: {e}")
        
        # Default to COCO flip pairs
        self._flip_pairs_cache = COCO_FLIP_PAIRS
        return COCO_FLIP_PAIRS
    
    def _extract_input_size(self) -> Tuple[int, int]:
        """Extract input size from MMPose config."""
        try:
            # Try different config locations
            if hasattr(self._cfg, "data_preprocessor"):
                dp = self._cfg.data_preprocessor
                if hasattr(dp, "input_size"):
                    return tuple(dp.input_size)
            
            if hasattr(self._cfg, "codec"):
                codec = self._cfg.codec
                if hasattr(codec, "input_size"):
                    return tuple(codec.input_size)
        except Exception as e:
            logger.debug(f"Could not extract input size from config: {e}")
        
        # Default
        return (256, 192)
    
    def _extract_num_keypoints(self) -> int:
        """Extract number of keypoints from MMPose config."""
        try:
            if hasattr(self._cfg, "data_preprocessor"):
                dp = self._cfg.data_preprocessor
                if hasattr(dp, "num_keypoints"):
                    return dp.num_keypoints
            
            # Try from dataset info
            dataset_info = self._cfg.get("dataset_info", {})
            keypoint_info = dataset_info.get("keypoint_info", {})
            if keypoint_info:
                return len(keypoint_info)
        except Exception as e:
            logger.debug(f"Could not extract num_keypoints from config: {e}")
        
        # Default for COCO
        return 17
    
    def _extract_model_name(self) -> str:
        """Extract model name from config path."""
        config_name = self._config_path.stem
        # Remove common prefixes/suffixes
        name = config_name.replace("td-hm_", "").replace("_8xb64-210e_coco-256x192", "")
        return name
    
    def _get_heatmap_size(self) -> Tuple[int, int]:
        """Get expected heatmap output size."""
        # Typically heatmap is 1/4 of input size
        h, w = self.input_size
        return (h // 4, w // 4)
    
    def _forward_with_heatmaps(
        self,
        image: npt.NDArray[np.uint8],
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> Tuple[npt.NDArray[np.float32], Dict[str, Any]]:
        """
        Perform forward pass and extract heatmaps directly from model head.
        
        This method bypasses the normal inference API to access intermediate
        heatmap outputs before they are decoded into keypoint coordinates.
        
        Args:
            image: RGB image (H, W, 3), uint8.
            bbox: Optional bounding box (x, y, w, h).
        
        Returns:
            Tuple of (heatmaps array, metadata dict).
        """
        import torch
        import cv2
        from mmpose.structures import PoseDataSample
        
        # Prepare image crop
        if bbox is not None:
            x, y, w, h = bbox
            crop = image[int(y):int(y+h), int(x):int(x+w)]
        else:
            crop = image
        
        # Resize to model input size
        resized = cv2.resize(crop, (self.input_size[1], self.input_size[0]))  # (W, H)
        
        # Prepare input tensor
        # MMPose expects BGR format
        img_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        img_tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).float()  # (C, H, W)
        
        # Normalize (MMPose uses ImageNet stats by default)
        mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)
        
        # Move to correct device
        device = next(self._model.parameters()).device
        img_tensor = img_tensor.to(device)
        
        # Forward through model
        with torch.no_grad():
            # Extract features
            feats = self._model.extract_feat(img_tensor)
            
            # Forward through head to get heatmaps
            head_output = self._model.head.forward(feats)
            
            # Extract heatmaps tensor
            heatmaps_tensor = None
            
            if isinstance(head_output, torch.Tensor):
                # Direct tensor output
                heatmaps_tensor = head_output
            elif isinstance(head_output, tuple) and len(head_output) > 0:
                # Tuple output, first element is usually heatmaps
                heatmaps_tensor = head_output[0]
            elif isinstance(head_output, dict) and "heatmaps" in head_output:
                heatmaps_tensor = head_output["heatmaps"]
            
            if heatmaps_tensor is not None:
                # Convert to numpy
                heatmaps = heatmaps_tensor.squeeze(0).cpu().numpy()  # Remove batch dim, shape: (K, H, W)
                
                # Ensure correct shape
                if heatmaps.ndim == 3:
                    # Apply sigmoid if needed (heatmaps might be logits)
                    if heatmaps.min() < 0 or heatmaps.max() > 1:
                        heatmaps = 1.0 / (1.0 + np.exp(-heatmaps))  # Sigmoid
                    
                    # Normalize to [0, 1]
                    heatmaps = self._normalize_heatmaps(heatmaps)
                    heatmaps = np.clip(heatmaps, 0.0, 1.0)
                    
                    # Also get keypoint predictions for confidence scores
                    # Create proper data sample for prediction
                    data_sample = PoseDataSample()
                    try:
                        predictions = self._model.head.predict([feats], [data_sample], test_cfg={})
                        if len(predictions) > 0 and hasattr(predictions[0].pred_instances, "keypoint_scores"):
                            confidence_map = predictions[0].pred_instances.keypoint_scores[0].cpu().numpy()
                            confidence_map = np.clip(confidence_map, 0.0, 1.0).astype(np.float32)
                        else:
                            confidence_map = None
                    except Exception as e:
                        logger.debug(f"Could not get confidence scores: {e}")
                        confidence_map = None
                    
                    metadata = {
                        "scale_factor": heatmaps.shape[1] / self.input_size[0],
                        "offset": (0.0, 0.0),
                        "confidence_map": confidence_map
                    }
                    
                    return heatmaps.astype(np.float32), metadata
        
        # If we reach here, extraction failed
        raise RuntimeError("Failed to extract heatmaps from model head")
    
    def _extract_heatmaps(
        self,
        result: Any,
        original_size: Tuple[int, int],
        bbox: Optional[Tuple[float, float, float, float]]
    ) -> Tuple[npt.NDArray[np.float32], Dict[str, Any]]:
        """
        Extract heatmaps from MMPose result object.
        
        Handles different output formats:
        1. Direct heatmap output (TopDown models)
        2. SimCC output (reconstruct Gaussians from coords)
        
        Args:
            result: MMPose prediction result (PoseDataSample).
            original_size: Original image (H, W).
            bbox: Bounding box used for inference.
        
        Returns:
            Tuple of (heatmap array, metadata dict).
        """
        pred_instances = result.pred_instances
        
        # Try to get direct heatmaps
        if hasattr(result, "pred_fields") and hasattr(result.pred_fields, "heatmaps"):
            heatmaps = result.pred_fields.heatmaps
            if hasattr(heatmaps, "numpy"):
                heatmaps = heatmaps.numpy()
            elif hasattr(heatmaps, "cpu"):
                heatmaps = heatmaps.cpu().numpy()
            
            # Normalize to [0, 1] and clip to ensure valid range
            heatmaps = self._normalize_heatmaps(heatmaps)
            heatmaps = np.clip(heatmaps, 0.0, 1.0)
            
            metadata = {
                "scale_factor": 1.0,
                "offset": (0.0, 0.0),
                "confidence_map": np.clip(pred_instances.keypoint_scores[0], 0.0, 1.0).astype(np.float32)
            }
            
            return heatmaps.astype(np.float32), metadata
        
        # Reconstruct from keypoint coordinates
        keypoints = pred_instances.keypoints[0]  # (K, 2)
        scores = pred_instances.keypoint_scores[0]  # (K,)
        
        heatmap_size = self._get_heatmap_size()
        heatmaps = self._reconstruct_gaussian_heatmap(
            keypoints=keypoints.astype(np.float32),
            scores=scores.astype(np.float32),
            heatmap_size=heatmap_size,
            input_size=self.input_size,
            sigma=2.0
        )
        
        # Ensure heatmaps are in [0, 1] range
        heatmaps = np.clip(heatmaps, 0.0, 1.0)
        
        metadata = {
            "scale_factor": heatmap_size[0] / self.input_size[0],
            "offset": (0.0, 0.0),
            "confidence_map": np.clip(scores, 0.0, 1.0).astype(np.float32),
            "reconstructed": True
        }
        
        return heatmaps, metadata
    
    def _normalize_heatmaps(
        self,
        heatmaps: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Normalize heatmaps to [0, 1] range per keypoint."""
        normalized = np.zeros_like(heatmaps)
        for k in range(heatmaps.shape[0]):
            hm = heatmaps[k]
            hm_min, hm_max = hm.min(), hm.max()
            if hm_max > hm_min:
                normalized[k] = (hm - hm_min) / (hm_max - hm_min)
            else:
                normalized[k] = hm
        return normalized
    
    def _reconstruct_gaussian_heatmap(
        self,
        keypoints: npt.NDArray[np.float32],
        scores: npt.NDArray[np.float32],
        heatmap_size: Tuple[int, int],
        input_size: Tuple[int, int],
        sigma: float = 2.0
    ) -> npt.NDArray[np.float32]:
        """
        Reconstruct heatmaps as Gaussians centered at keypoint coordinates.
        
        Used for models that output coordinates directly (e.g., SimCC).
        
        Mathematical Form:
            H_k(x, y) = s_k · exp(-[(x - x_k)² + (y - y_k)²] / (2σ²))
        
        Args:
            keypoints: Coordinates array (num_keypoints, 2) in input space.
            scores: Confidence scores (num_keypoints,).
            heatmap_size: (height, width) of output heatmap.
            input_size: (height, width) of model input.
            sigma: Gaussian std dev in heatmap pixels.
        
        Returns:
            Reconstructed heatmap (num_keypoints, h, w).
        """
        num_keypoints = keypoints.shape[0]
        h, w = heatmap_size
        
        # Scale factor from input to heatmap space
        scale_y = h / input_size[0]
        scale_x = w / input_size[1]
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        heatmaps = np.zeros((num_keypoints, h, w), dtype=np.float32)
        
        for k in range(num_keypoints):
            # Transform keypoint to heatmap space
            kp_x = keypoints[k, 0] * scale_x
            kp_y = keypoints[k, 1] * scale_y
            
            # Generate Gaussian
            gaussian = np.exp(
                -((x_coords - kp_x)**2 + (y_coords - kp_y)**2) / (2 * sigma**2)
            )
            
            # Weight by confidence score
            heatmaps[k] = scores[k] * gaussian
        
        return heatmaps


def create_model_adapter(
    model_name: str,
    device: str = "cuda",
    project_root: Optional[Path] = None
) -> BasePoseModel:
    """
    Factory function to create the appropriate model adapter.
    
    This is the recommended way to instantiate models, as it provides
    a clean API and handles all path resolution internally.
    
    Args:
        model_name: Name of the model to load. Options:
            - "resnet50": ResNet-50 baseline
            - "hrnet_w32": HRNet-W32 (high-resolution)
            - "vitpose_small": ViTPose-Small (Vision Transformer)
        device: Computation device ("cpu", "cuda", "cuda:0").
        project_root: Optional project root for path resolution.
    
    Returns:
        Initialized model adapter implementing BasePoseModel interface.
    
    Raises:
        ValueError: If model_name is not recognized.
    
    Example:
        ```python
        model = create_model_adapter("hrnet_w32", device="cuda")
        heatmap = model.predict(image, bbox=(x, y, w, h))
        ```
    """
    model_name_lower = model_name.lower().replace("-", "_")
    
    if model_name_lower in MMPoseAdapter.MODEL_REGISTRY:
        return MMPoseAdapter.from_model_name(
            model_name_lower,
            device=device,
            project_root=project_root
        )
    
    available = list(MMPoseAdapter.MODEL_REGISTRY.keys())
    raise ValueError(
        f"Unknown model: {model_name}\n"
        f"Available models: {available}"
    )


def list_available_models() -> List[str]:
    """
    List all available pre-configured models.
    
    Returns:
        List of model names that can be passed to create_model_adapter().
    """
    return list(MMPoseAdapter.MODEL_REGISTRY.keys())

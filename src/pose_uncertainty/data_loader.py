import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional
from pycocotools.coco import COCO
from .utils.types import ImageSample, Keypoint


def _resize_image_and_annotations(
    img_array: np.ndarray,
    bbox: tuple,
    keypoints: List[Keypoint],
    resize_scale: float,
) -> tuple:
    """Resize image and scale annotations by the same factor."""
    if resize_scale is None or abs(resize_scale - 1.0) < 1e-8:
        return img_array, bbox, keypoints

    if resize_scale <= 0:
        raise ValueError(f"resize_scale must be > 0, got {resize_scale}")

    height, width = img_array.shape[:2]
    new_width = max(1, int(round(width * resize_scale)))
    new_height = max(1, int(round(height * resize_scale)))
    interpolation = cv2.INTER_AREA if resize_scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img_array, (new_width, new_height), interpolation=interpolation)

    x, y, w, h = bbox
    scaled_bbox = (
        float(x * resize_scale),
        float(y * resize_scale),
        float(w * resize_scale),
        float(h * resize_scale),
    )

    scaled_keypoints = [
        Keypoint(
            id=kp.id,
            x=float(kp.x * resize_scale),
            y=float(kp.y * resize_scale),
            confidence=kp.confidence,
            name=kp.name,
        )
        for kp in keypoints
    ]

    return resized, scaled_bbox, scaled_keypoints


def _normalize_kernel_size(kernel_size: Optional[int]) -> int:
    if kernel_size is None:
        return 0

    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return 0

    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def _apply_image_quality_augmentations(
    img_array: np.ndarray,
    image_id: int,
    noise_mean: float,
    noise_sigma: float,
    noise_seed: Optional[int],
    contrast_factor: float,
    blur_kernel_size: Optional[int],
    blur_sigma: float,
    smooth_kernel_size: Optional[int],
) -> np.ndarray:
    """Apply optional quality degradations after resizing.

    The transforms are deterministic per image when ``noise_seed`` is set,
    which keeps benchmark runs reproducible across repeated passes.
    """
    result = img_array.astype(np.float32, copy=False)

    if abs(contrast_factor - 1.0) > 1e-8:
        result *= float(contrast_factor)

    if noise_sigma and noise_sigma > 0:
        seed_base = 0 if noise_seed is None else int(noise_seed)
        local_seed = (seed_base + int(image_id)) % (2**32)
        rng = np.random.default_rng(local_seed)
        noise = rng.normal(
            loc=float(noise_mean),
            scale=float(noise_sigma),
            size=result.shape,
        ).astype(np.float32)
        result += noise

    result = np.clip(result, 0.0, 255.0)

    blur_kernel = _normalize_kernel_size(blur_kernel_size)
    if blur_kernel > 1:
        result = cv2.GaussianBlur(result, (blur_kernel, blur_kernel), float(blur_sigma))

    smooth_kernel = _normalize_kernel_size(smooth_kernel_size)
    if smooth_kernel > 1:
        result = cv2.blur(result, (smooth_kernel, smooth_kernel))

    return np.clip(result, 0.0, 255.0).astype(np.uint8)

class PoseDatasetAdapter(ABC):
    """
    Clase abstracta (Interface).
    Obliga a cualquier dataset nuevo a comportarse igual.
    """
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[ImageSample]:
        pass

class COCOLoader(PoseDatasetAdapter):
    def __init__(
        self,
        data_root: str,
        ann_file: str,
        image_dir: str,
        resize_scale: float = 1.0,
        noise_mean: float = 0.0,
        noise_sigma: float = 0.0,
        noise_seed: Optional[int] = None,
        contrast_factor: float = 1.0,
        blur_kernel_size: Optional[int] = 0,
        blur_sigma: float = 0.0,
        smooth_kernel_size: Optional[int] = 0,
    ):
        """
        :param data_root: Base dataset directory (e.g., ./data/coco)
        :param ann_file: Annotation file name (e.g., annotations/person_keypoints_val2017.json)
        :param image_dir: Image subdirectory name (e.g., val2017)
        :param resize_scale: Scaling factor to adjust input image resolution.
        """
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, image_dir)
        self.ann_path = os.path.join(data_root, ann_file)
        self.resize_scale = resize_scale
        self.noise_mean = noise_mean
        self.noise_sigma = noise_sigma
        self.noise_seed = noise_seed
        self.contrast_factor = contrast_factor
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.smooth_kernel_size = smooth_kernel_size
        
        # Initialize COCO API
        print(f"Loading annotations from {self.ann_path}...")
        self.coco = COCO(self.ann_path)
        
        # Filter person category images only
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        self.img_ids = self.coco.getImgIds(catIds=self.cat_ids)
        print(f"Dataset loaded: {len(self.img_ids)} images discovered.")

    def __len__(self) -> int:
        return len(self.img_ids)

    def _parse_keypoints(self, raw_kps: List[float], num_keypoints: int) -> List[Keypoint]:
        """
        Convert COCO's flat keypoint format to List[Keypoint].
        
        COCO Format: [x1, y1, v1, x2, y2, v2, ..., x17, y17, v17]
        
        Args:
            raw_kps: Flat list of keypoint data (length 51 for COCO: 17 * 3).
            num_keypoints: Number of labeled keypoints (for validation).
        
        Returns:
            List of Keypoint objects with proper naming and confidence.
        
        Confidence Mapping:
            v=0 → confidence=0.0 (not labeled)
            v=1 → confidence=0.5 (occluded, exists but not visible)
            v=2 → confidence=1.0 (visible)
        """
        COCO_KEYPOINT_NAMES = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        keypoints = []
        
        # COCO has 17 keypoints, stored as triplets (x, y, visibility)
        for i in range(17):
            idx = i * 3
            x = raw_kps[idx]
            y = raw_kps[idx + 1]
            v = int(raw_kps[idx + 2])
            
            # Map visibility to confidence
            confidence_map = {0: 0.0, 1: 0.5, 2: 1.0}
            confidence = confidence_map.get(v, 0.0)
            
            keypoints.append(Keypoint(
                id=i,
                x=float(x),
                y=float(y),
                confidence=confidence,
                name=COCO_KEYPOINT_NAMES[i]
            ))
        
        return keypoints

    def __iter__(self) -> Iterator[ImageSample]:
        for img_id in self.img_ids:
            # Image metadata
            img_info = self.coco.loadImgs(img_id)[0]
            path = os.path.join(self.image_dir, img_info['file_name'])
            
            # Load annotations (Bbox and Keypoints)
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            
            # Single-person top-down evaluation: select person instance with maximum area
            if not anns: continue
            
            main_person = max(anns, key=lambda x: x['area'])
            bbox = main_person['bbox'] # [x, y, w, h]
            
            # Lazy loading to avoid RAM saturation
            img_array = cv2.imread(path)
            if img_array is None:
                continue # Skip corrupt image
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            keypoints = self._parse_keypoints(main_person['keypoints'], num_keypoints=17)
            img_array, bbox, keypoints = _resize_image_and_annotations(
                img_array,
                tuple(bbox),
                keypoints,
                self.resize_scale,
            )
            img_array = _apply_image_quality_augmentations(
                img_array,
                image_id=img_id,
                noise_mean=self.noise_mean,
                noise_sigma=self.noise_sigma,
                noise_seed=self.noise_seed,
                contrast_factor=self.contrast_factor,
                blur_kernel_size=self.blur_kernel_size,
                blur_sigma=self.blur_sigma,
                smooth_kernel_size=self.smooth_kernel_size,
            )

            yield ImageSample(
                image_id=img_id,
                image_path=path,
                image_array=img_array,
                bbox=bbox,
                ground_truth_keypoints=keypoints,
                dataset_source="coco"
            )

class CrowdPoseLoader(PoseDatasetAdapter):
    def __init__(
        self,
        data_root: str,
        ann_file: str,
        image_dir: str,
        resize_scale: float = 1.0,
        noise_mean: float = 0.0,
        noise_sigma: float = 0.0,
        noise_seed: Optional[int] = None,
        contrast_factor: float = 1.0,
        blur_kernel_size: Optional[int] = 0,
        blur_sigma: float = 0.0,
        smooth_kernel_size: Optional[int] = 0,
    ):
        """
        Loader for CrowdPose dataset.
        
        :param data_root: Base path (e.g., ./data/crowdpose)
        :param ann_file: Annotation JSON file (e.g., json/crowdpose_val.json)
        :param image_dir: Image directory (e.g., images)
        :param resize_scale: Scaling factor to adjust input image resolution.
        """
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, image_dir)
        self.ann_path = os.path.join(data_root, ann_file)
        self.resize_scale = resize_scale
        self.noise_mean = noise_mean
        self.noise_sigma = noise_sigma
        self.noise_seed = noise_seed
        self.contrast_factor = contrast_factor
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.smooth_kernel_size = smooth_kernel_size
        
        # Load CrowdPose annotations (COCO format compatible)
        print(f"Loading annotations from {self.ann_path}...")
        self.coco = COCO(self.ann_path)
        
        # Filter only 'person' category images
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        self.img_ids = self.coco.getImgIds(catIds=self.cat_ids)
        print(f"Dataset loaded: {len(self.img_ids)} images discovered.")

    def __len__(self) -> int:
        return len(self.img_ids)

    def _parse_keypoints(self, raw_kps: List[float], num_keypoints: int) -> List[Keypoint]:
        """
        Convert CrowdPose's flat keypoint format to List[Keypoint].
        
        CrowdPose Format: [x1, y1, v1, x2, y2, v2, ..., x14, y14, v14]
        
        Args:
            raw_kps: Flat list of keypoint data (length 42 for CrowdPose: 14 * 3).
            num_keypoints: Number of labeled keypoints (for validation).
        
        Returns:
            List of Keypoint objects with proper naming and confidence.
        
        Confidence Mapping:
            v=0 → confidence=0.0 (not labeled)
            v=1 → confidence=0.5 (occluded, exists but not visible)
            v=2 → confidence=1.0 (visible)
        """
        CROWDPOSE_KEYPOINT_NAMES = [
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
            "head", "neck"
        ]
        
        keypoints = []
        
        # CrowdPose has 14 keypoints, stored as triplets (x, y, visibility)
        for i in range(14):
            idx = i * 3
            x = raw_kps[idx]
            y = raw_kps[idx + 1]
            v = int(raw_kps[idx + 2])
            
            # Map visibility to confidence
            confidence_map = {0: 0.0, 1: 0.5, 2: 1.0}
            confidence = confidence_map.get(v, 0.0)
            
            keypoints.append(Keypoint(
                id=i,
                x=float(x),
                y=float(y),
                confidence=confidence,
                name=CROWDPOSE_KEYPOINT_NAMES[i]
            ))
        
        return keypoints

    def __iter__(self) -> Iterator[ImageSample]:
        for img_id in self.img_ids:
            # Image metadata
            img_info = self.coco.loadImgs(img_id)[0]
            path = os.path.join(self.image_dir, img_info['file_name'])
            
            # Load Annotations (Bbox and Keypoints)
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            
            # CrowdPose has multiple people per image.
            # For single-person pipeline, select the largest person (by bbox area).
            if not anns: 
                continue
            
            # Calculate area from bbox [x, y, w, h] since 'area' field doesn't exist
            main_person = max(anns, key=lambda x: x['bbox'][2] * x['bbox'][3])
            bbox = main_person['bbox']  # [x, y, w, h]
            
            # Lazy loading: Load image only when needed
            img_array = cv2.imread(path)
            if img_array is None:
                continue  # Skip if image is corrupted
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            keypoints = self._parse_keypoints(main_person['keypoints'], num_keypoints=14)
            img_array, bbox, keypoints = _resize_image_and_annotations(
                img_array,
                tuple(bbox),
                keypoints,
                self.resize_scale,
            )
            img_array = _apply_image_quality_augmentations(
                img_array,
                image_id=img_id,
                noise_mean=self.noise_mean,
                noise_sigma=self.noise_sigma,
                noise_seed=self.noise_seed,
                contrast_factor=self.contrast_factor,
                blur_kernel_size=self.blur_kernel_size,
                blur_sigma=self.blur_sigma,
                smooth_kernel_size=self.smooth_kernel_size,
            )

            yield ImageSample(
                image_id=img_id,
                image_path=path,
                image_array=img_array,
                bbox=bbox,
                ground_truth_keypoints=keypoints,
                dataset_source="crowdpose"
            )

class OCHumanLoader(PoseDatasetAdapter):
    def __init__(
        self,
        data_root: str,
        ann_file: str,
        image_dir: str,
        resize_scale: float = 1.0,
        noise_mean: float = 0.0,
        noise_sigma: float = 0.0,
        noise_seed: Optional[int] = None,
        contrast_factor: float = 1.0,
        blur_kernel_size: Optional[int] = 0,
        blur_sigma: float = 0.0,
        smooth_kernel_size: Optional[int] = 0,
    ):
        """
        Loader for OCHuman dataset.
        
        :param data_root: Base path (e.g., ./data/ochuman)
        :param ann_file: Annotation JSON file (e.g., annotations/ochuman_coco_format_val_range_0.00_1.00.json)
        :param image_dir: Image directory (e.g., images)
        :param resize_scale: Scaling factor to adjust input image resolution.
        """
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, image_dir)
        self.ann_path = os.path.join(data_root, ann_file)
        self.resize_scale = resize_scale
        self.noise_mean = noise_mean
        self.noise_sigma = noise_sigma
        self.noise_seed = noise_seed
        self.contrast_factor = contrast_factor
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.smooth_kernel_size = smooth_kernel_size
        
        # Load OCHuman annotations (COCO format compatible)
        print(f"Loading annotations from {self.ann_path}...")
        self.coco = COCO(self.ann_path)
        
        # Filter only 'person' category images
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        self.img_ids = self.coco.getImgIds(catIds=self.cat_ids)
        print(f"Dataset loaded: {len(self.img_ids)} images discovered.")

    def __len__(self) -> int:
        return len(self.img_ids)

    def _parse_keypoints(self, raw_kps: List[float], num_keypoints: int) -> List[Keypoint]:
        """
        Convert OCHuman's flat keypoint format to List[Keypoint].
        
        OCHuman Format: [x1, y1, v1, x2, y2, v2, ..., x17, y17, v17]
        (Same as COCO: 17 keypoints)
        
        Args:
            raw_kps: Flat list of keypoint data (length 51 for OCHuman: 17 * 3).
            num_keypoints: Number of labeled keypoints (for validation).
        
        Returns:
            List of Keypoint objects with proper naming and confidence.
        
        Confidence Mapping:
            v=0 → confidence=0.0 (not labeled)
            v=1 → confidence=0.5 (occluded, exists but not visible)
            v=2 → confidence=1.0 (visible)
        """
        OCHUMAN_KEYPOINT_NAMES = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        keypoints = []
        
        # OCHuman has 17 keypoints (same as COCO), stored as triplets (x, y, visibility)
        for i in range(17):
            idx = i * 3
            x = raw_kps[idx]
            y = raw_kps[idx + 1]
            v = int(raw_kps[idx + 2])
            
            # Map visibility to confidence
            confidence_map = {0: 0.0, 1: 0.5, 2: 1.0}
            confidence = confidence_map.get(v, 0.0)
            
            keypoints.append(Keypoint(
                id=i,
                x=float(x),
                y=float(y),
                confidence=confidence,
                name=OCHUMAN_KEYPOINT_NAMES[i]
            ))
        
        return keypoints

    def __iter__(self) -> Iterator[ImageSample]:
        for img_id in self.img_ids:
            # Image metadata
            img_info = self.coco.loadImgs(img_id)[0]
            path = os.path.join(self.image_dir, img_info['file_name'])
            
            # Load Annotations (Bbox and Keypoints)
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            
            # OCHuman has multiple people per image with heavy occlusion.
            # For single-person pipeline, select the largest person (by area).
            if not anns: 
                continue
            
            # OCHuman has 'area' field like COCO
            main_person = max(anns, key=lambda x: x['area'])
            bbox = main_person['bbox']  # [x, y, w, h]
            
            # Lazy loading: Load image only when needed
            img_array = cv2.imread(path)
            if img_array is None:
                continue  # Skip if image is corrupted
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            keypoints = self._parse_keypoints(main_person['keypoints'], num_keypoints=17)
            img_array, bbox, keypoints = _resize_image_and_annotations(
                img_array,
                tuple(bbox),
                keypoints,
                self.resize_scale,
            )
            img_array = _apply_image_quality_augmentations(
                img_array,
                image_id=img_id,
                noise_mean=self.noise_mean,
                noise_sigma=self.noise_sigma,
                noise_seed=self.noise_seed,
                contrast_factor=self.contrast_factor,
                blur_kernel_size=self.blur_kernel_size,
                blur_sigma=self.blur_sigma,
                smooth_kernel_size=self.smooth_kernel_size,
            )

            yield ImageSample(
                image_id=img_id,
                image_path=path,
                image_array=img_array,
                bbox=bbox,
                ground_truth_keypoints=keypoints,
                dataset_source="ochuman"
            )
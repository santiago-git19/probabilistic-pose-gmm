"""Verification and Unit Tests for Batch Processing Implementation.

Validates that batch processing methods (predict_batch and predict_keypoints_batch)
produce identical results to sequential processing, while maintaining metadata consistency.

Usage:
    # Run via pytest:
    pytest src/tests/test_batch_processing.py

    # Run via CLI:
    python src/tests/test_batch_processing.py --model mock --device cpu
    python src/tests/test_batch_processing.py --model hrnet_w32 --device cuda
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Tuple
import logging

import pytest
import numpy as np
import numpy.typing as npt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pose_uncertainty.models.base import BasePoseModel, MockPoseModel
from src.pose_uncertainty.models.adapters import MMPoseAdapter
from src.pose_uncertainty.utils.types import StandardizedHeatmap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_dummy_images(
    num_images: int = 8,
    image_size: Tuple[int, int] = (480, 640),
    seed: int = 42
) -> List[npt.NDArray[np.uint8]]:
    """Generate dummy RGB images for testing."""
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(num_images):
        img = rng.integers(0, 256, (*image_size, 3), dtype=np.uint8)
        images.append(img)
    return images


def verify_heatmaps_identical(
    heatmaps_seq: List[StandardizedHeatmap],
    heatmaps_batch: List[StandardizedHeatmap],
    atol: float = 1e-5,
    rtol: float = 1e-5
) -> bool:
    """Verify that sequential and batch heatmaps are numerically identical."""
    if len(heatmaps_seq) != len(heatmaps_batch):
        logger.error(
            f"Length mismatch: sequential={len(heatmaps_seq)}, "
            f"batch={len(heatmaps_batch)}"
        )
        return False
    
    all_passed = True
    for i, (hm_seq, hm_batch) in enumerate(zip(heatmaps_seq, heatmaps_batch)):
        if hm_seq.data.shape != hm_batch.data.shape:
            logger.error(
                f"Image {i}: Shape mismatch - "
                f"seq={hm_seq.data.shape}, batch={hm_batch.data.shape}"
            )
            all_passed = False
            continue
        
        if not np.allclose(hm_seq.data, hm_batch.data, atol=atol, rtol=rtol):
            max_diff = np.abs(hm_seq.data - hm_batch.data).max()
            logger.error(
                f"Image {i}: Heatmap data mismatch - max_diff={max_diff:.6e}"
            )
            all_passed = False
        
        if hm_seq.original_size != hm_batch.original_size:
            logger.error(
                f"Image {i}: original_size mismatch - "
                f"seq={hm_seq.original_size}, batch={hm_batch.original_size}"
            )
            all_passed = False
        
        if hm_seq.reconstructed != hm_batch.reconstructed:
            logger.error(
                f"Image {i}: reconstructed flag mismatch - "
                f"seq={hm_seq.reconstructed}, batch={hm_batch.reconstructed}"
            )
            all_passed = False
        
        if hm_seq.confidence_map is not None and hm_batch.confidence_map is not None:
            if not np.allclose(
                hm_seq.confidence_map, hm_batch.confidence_map, atol=atol, rtol=rtol
            ):
                max_diff = np.abs(hm_seq.confidence_map - hm_batch.confidence_map).max()
                logger.error(
                    f"Image {i}: Confidence map mismatch - max_diff={max_diff:.6e}"
                )
                all_passed = False
    
    return all_passed


def verify_keypoints_identical(
    keypoints_seq: List[npt.NDArray[np.float32]],
    scores_seq: List[npt.NDArray[np.float32]],
    keypoints_batch: List[npt.NDArray[np.float32]],
    scores_batch: List[npt.NDArray[np.float32]],
    atol: float = 1e-3,
    rtol: float = 1e-3
) -> bool:
    """Verify that sequential and batch keypoint predictions are identical."""
    if len(keypoints_seq) != len(keypoints_batch):
        logger.error(
            f"Length mismatch: sequential={len(keypoints_seq)}, "
            f"batch={len(keypoints_batch)}"
        )
        return False
    
    all_passed = True
    for i, (kpts_seq, scores_s, kpts_batch, scores_b) in enumerate(
        zip(keypoints_seq, scores_seq, keypoints_batch, scores_batch)
    ):
        if not np.allclose(kpts_seq, kpts_batch, atol=atol, rtol=rtol):
            max_diff = np.abs(kpts_seq - kpts_batch).max()
            logger.error(
                f"Image {i}: Keypoints mismatch - max_diff={max_diff:.6e}"
            )
            all_passed = False
        
        if not np.allclose(scores_s, scores_b, atol=atol, rtol=rtol):
            max_diff = np.abs(scores_s - scores_b).max()
            logger.error(
                f"Image {i}: Scores mismatch - max_diff={max_diff:.6e}"
            )
            all_passed = False
    
    return all_passed


def benchmark_predict(
    model: BasePoseModel,
    images: List[npt.NDArray[np.uint8]],
    bboxes: List[Tuple[float, float, float, float]] = None
) -> Tuple[List[StandardizedHeatmap], float]:
    """Benchmark sequential predict() calls."""
    if bboxes is None:
        bboxes = [None] * len(images)
    
    start_time = time.perf_counter()
    results = [model.predict(img, bbox) for img, bbox in zip(images, bboxes)]
    elapsed = time.perf_counter() - start_time
    return results, elapsed


def benchmark_predict_batch(
    model: BasePoseModel,
    images: List[npt.NDArray[np.uint8]],
    bboxes: List[Tuple[float, float, float, float]] = None
) -> Tuple[List[StandardizedHeatmap], float]:
    """Benchmark batch predict_batch() call."""
    start_time = time.perf_counter()
    results = model.predict_batch(images, bboxes)
    elapsed = time.perf_counter() - start_time
    return results, elapsed


def benchmark_predict_keypoints(
    model: BasePoseModel,
    images: List[npt.NDArray[np.uint8]],
    bboxes: List[Tuple[float, float, float, float]] = None
) -> Tuple[Tuple[List, List], float]:
    """Benchmark sequential predict_keypoints() calls."""
    if bboxes is None:
        bboxes = [None] * len(images)
    
    keypoints_list = []
    scores_list = []
    start_time = time.perf_counter()
    for img, bbox in zip(images, bboxes):
        kpts, scores = model.predict_keypoints(img, bbox)
        keypoints_list.append(kpts)
        scores_list.append(scores)
    elapsed = time.perf_counter() - start_time
    return (keypoints_list, scores_list), elapsed


def benchmark_predict_keypoints_batch(
    model: BasePoseModel,
    images: List[npt.NDArray[np.uint8]],
    bboxes: List[Tuple[float, float, float, float]] = None
) -> Tuple[Tuple[List, List], float]:
    """Benchmark batch predict_keypoints_batch() call."""
    start_time = time.perf_counter()
    keypoints_list, scores_list = model.predict_keypoints_batch(images, bboxes)
    elapsed = time.perf_counter() - start_time
    return (keypoints_list, scores_list), elapsed


# =============================================================================
# PyTest Test Cases
# =============================================================================

def test_mock_batch_predict_heatmaps_identity():
    """Verify MockPoseModel batch predict vs sequential predict."""
    model = MockPoseModel(mode="unimodal", input_size=(64, 48))
    images = generate_dummy_images(num_images=4, image_size=(128, 128))
    
    heatmaps_seq, _ = benchmark_predict(model, images)
    heatmaps_batch, _ = benchmark_predict_batch(model, images)
    
    assert verify_heatmaps_identical(heatmaps_seq, heatmaps_batch)


def test_mock_batch_predict_keypoints_identity():
    """Verify MockPoseModel batch keypoints vs sequential keypoints."""
    model = MockPoseModel(mode="unimodal", input_size=(64, 48))
    images = generate_dummy_images(num_images=4, image_size=(128, 128))
    
    (kpts_seq, scores_seq), _ = benchmark_predict_keypoints(model, images)
    (kpts_batch, scores_batch), _ = benchmark_predict_keypoints_batch(model, images)
    
    assert verify_keypoints_identical(kpts_seq, scores_seq, kpts_batch, scores_batch)


def test_mock_batch_with_bboxes():
    """Verify batch processing with explicit bounding boxes."""
    model = MockPoseModel(mode="unimodal", input_size=(64, 48))
    images = generate_dummy_images(num_images=3, image_size=(200, 200))
    bboxes = [(10, 10, 100, 100), (20, 20, 150, 150), (0, 0, 200, 200)]
    
    heatmaps_seq, _ = benchmark_predict(model, images, bboxes)
    heatmaps_batch, _ = benchmark_predict_batch(model, images, bboxes)
    
    assert verify_heatmaps_identical(heatmaps_seq, heatmaps_batch)


# =============================================================================
# CLI Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify batch processing implementation and benchmark performance"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mock",
        choices=["mock", "resnet50", "hrnet_w32", "vitpose_small"],
        help="Model to use for testing (default: mock)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (cpu, cuda, cuda:0, etc.)"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=8,
        help="Number of test images (default: 8)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[480, 640],
        metavar=("HEIGHT", "WIDTH"),
        help="Image size as HEIGHT WIDTH (default: 480 640)"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for numerical comparison (default: 1e-5)"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for numerical comparison (default: 1e-5)"
    )
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("BATCH PROCESSING VERIFICATION SCRIPT")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Num images: {args.num_images}")
    logger.info(f"Image size: {args.image_size}")
    logger.info(f"Tolerance: atol={args.atol}, rtol={args.rtol}")
    logger.info("=" * 80)
    
    if args.model == "mock":
        model = MockPoseModel(mode="unimodal", input_size=(64, 48))
        logger.info("✓ MockPoseModel created")
    else:
        try:
            model = MMPoseAdapter.from_model_name(args.model, device=args.device)
            logger.info(f"✓ MMPoseAdapter ({args.model}) created")
            model.warmup(iterations=3)
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            return 1
            
    images = generate_dummy_images(num_images=args.num_images, image_size=tuple(args.image_size))
    
    heatmaps_seq, time_seq = benchmark_predict(model, images)
    heatmaps_batch, time_batch = benchmark_predict_batch(model, images)
    speedup = time_seq / time_batch if time_batch > 0 else float("inf")
    logger.info(f"Speedup heatmaps: {speedup:.2f}x (sequential: {time_seq:.4f}s, batch: {time_batch:.4f}s)")
    
    if not verify_heatmaps_identical(heatmaps_seq, heatmaps_batch, atol=args.atol, rtol=args.rtol):
        logger.error("✗ Heatmap verification FAILED")
        return 1
        
    (kpts_seq, scores_seq), time_kpts_seq = benchmark_predict_keypoints(model, images)
    (kpts_batch, scores_batch), time_kpts_batch = benchmark_predict_keypoints_batch(model, images)
    speedup_kpts = time_kpts_seq / time_kpts_batch if time_kpts_batch > 0 else float("inf")
    logger.info(f"Speedup keypoints: {speedup_kpts:.2f}x (sequential: {time_kpts_seq:.4f}s, batch: {time_kpts_batch:.4f}s)")
    
    if not verify_keypoints_identical(kpts_seq, scores_seq, kpts_batch, scores_batch, atol=args.atol, rtol=args.rtol):
        logger.error("✗ Keypoint verification FAILED")
        return 1
        
    logger.info("=" * 80)
    logger.info("VERIFICATION SUMMARY: ALL TESTS PASSED ✓")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())

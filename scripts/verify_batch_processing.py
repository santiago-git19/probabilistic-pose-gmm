"""
Verification Script for Batch Processing Implementation.

This script validates that the batch processing methods (predict_batch and 
predict_keypoints_batch) produce identical results to sequential processing,
while measuring the performance improvement.

Requirements:
------------
1. Numerical Identity: batch results == sequential results (np.allclose)
2. Metadata Consistency: Each result has correct original_size, metadata
3. Performance: Batch processing should be faster for batch_size >= 4

Usage:
------
    python scripts/verify_batch_processing.py
    
    # With real MMPose model (requires GPU):
    python scripts/verify_batch_processing.py --model hrnet_w32 --device cuda
    
    # With mock model (fast, CPU-only):
    python scripts/verify_batch_processing.py --model mock --device cpu
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Tuple
import logging

import numpy as np
import numpy.typing as npt

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from pose_uncertainty.models.base import BasePoseModel, MockPoseModel
from pose_uncertainty.models.adapters import MMPoseAdapter
from pose_uncertainty.utils.types import StandardizedHeatmap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_dummy_images(
    num_images: int = 8,
    image_size: Tuple[int, int] = (480, 640),
    seed: int = 42
) -> List[npt.NDArray[np.uint8]]:
    """
    Generate dummy RGB images for testing.
    
    Args:
        num_images: Number of images to generate.
        image_size: (height, width) of each image.
        seed: Random seed for reproducibility.
    
    Returns:
        List of N images, each (H, W, 3) uint8.
    """
    rng = np.random.default_rng(seed)
    images = []
    
    for i in range(num_images):
        # Generate random RGB image
        img = rng.integers(0, 256, (*image_size, 3), dtype=np.uint8)
        images.append(img)
    
    return images


def verify_heatmaps_identical(
    heatmaps_seq: List[StandardizedHeatmap],
    heatmaps_batch: List[StandardizedHeatmap],
    atol: float = 1e-5,
    rtol: float = 1e-5
) -> bool:
    """
    Verify that sequential and batch heatmaps are numerically identical.
    
    Args:
        heatmaps_seq: Results from sequential predict() calls.
        heatmaps_batch: Results from predict_batch().
        atol: Absolute tolerance for np.allclose.
        rtol: Relative tolerance for np.allclose.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    if len(heatmaps_seq) != len(heatmaps_batch):
        logger.error(
            f"Length mismatch: sequential={len(heatmaps_seq)}, "
            f"batch={len(heatmaps_batch)}"
        )
        return False
    
    all_passed = True
    
    for i, (hm_seq, hm_batch) in enumerate(zip(heatmaps_seq, heatmaps_batch)):
        # Check data shape
        if hm_seq.data.shape != hm_batch.data.shape:
            logger.error(
                f"Image {i}: Shape mismatch - "
                f"seq={hm_seq.data.shape}, batch={hm_batch.data.shape}"
            )
            all_passed = False
            continue
        
        # Check numerical values
        if not np.allclose(hm_seq.data, hm_batch.data, atol=atol, rtol=rtol):
            max_diff = np.abs(hm_seq.data - hm_batch.data).max()
            logger.error(
                f"Image {i}: Heatmap data mismatch - max_diff={max_diff:.6e}"
            )
            all_passed = False
        
        # Check metadata
        if hm_seq.original_size != hm_batch.original_size:
            logger.error(
                f"Image {i}: original_size mismatch - "
                f"seq={hm_seq.original_size}, batch={hm_batch.original_size}"
            )
            all_passed = False
        
        # Check reconstructed flag
        if hm_seq.reconstructed != hm_batch.reconstructed:
            logger.error(
                f"Image {i}: reconstructed flag mismatch - "
                f"seq={hm_seq.reconstructed}, batch={hm_batch.reconstructed}"
            )
            all_passed = False
        
        # Check confidence maps if present
        if hm_seq.confidence_map is not None and hm_batch.confidence_map is not None:
            if not np.allclose(
                hm_seq.confidence_map, hm_batch.confidence_map, atol=atol, rtol=rtol
            ):
                max_diff = np.abs(hm_seq.confidence_map - hm_batch.confidence_map).max()
                logger.error(
                    f"Image {i}: Confidence map mismatch - max_diff={max_diff:.6e}"
                )
                all_passed = False
    
    if all_passed:
        logger.info("✓ All heatmap verifications PASSED")
    
    return all_passed


def verify_keypoints_identical(
    keypoints_seq: List[npt.NDArray[np.float32]],
    scores_seq: List[npt.NDArray[np.float32]],
    keypoints_batch: List[npt.NDArray[np.float32]],
    scores_batch: List[npt.NDArray[np.float32]],
    atol: float = 1e-3,
    rtol: float = 1e-3
) -> bool:
    """
    Verify that sequential and batch keypoint predictions are identical.
    
    Args:
        keypoints_seq: Keypoints from sequential predict_keypoints().
        scores_seq: Scores from sequential predict_keypoints().
        keypoints_batch: Keypoints from predict_keypoints_batch().
        scores_batch: Scores from predict_keypoints_batch().
        atol: Absolute tolerance for np.allclose.
        rtol: Relative tolerance for np.allclose.
    
    Returns:
        True if all checks pass, False otherwise.
    """
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
        # Check keypoints
        if not np.allclose(kpts_seq, kpts_batch, atol=atol, rtol=rtol):
            max_diff = np.abs(kpts_seq - kpts_batch).max()
            logger.error(
                f"Image {i}: Keypoints mismatch - max_diff={max_diff:.6e}"
            )
            all_passed = False
        
        # Check scores
        if not np.allclose(scores_s, scores_b, atol=atol, rtol=rtol):
            max_diff = np.abs(scores_s - scores_b).max()
            logger.error(
                f"Image {i}: Scores mismatch - max_diff={max_diff:.6e}"
            )
            all_passed = False
    
    if all_passed:
        logger.info("✓ All keypoint verifications PASSED")
    
    return all_passed


def benchmark_predict(
    model: BasePoseModel,
    images: List[npt.NDArray[np.uint8]],
    bboxes: List[Tuple[float, float, float, float]] = None
) -> Tuple[List[StandardizedHeatmap], float]:
    """
    Benchmark sequential predict() calls.
    
    Args:
        model: Model adapter instance.
        images: List of input images.
        bboxes: Optional list of bounding boxes.
    
    Returns:
        Tuple of (results, elapsed_time_seconds).
    """
    if bboxes is None:
        bboxes = [None] * len(images)
    
    logger.info(f"Running sequential predict() for {len(images)} images...")
    
    start_time = time.perf_counter()
    results = [model.predict(img, bbox) for img, bbox in zip(images, bboxes)]
    elapsed = time.perf_counter() - start_time
    
    logger.info(f"Sequential predict() took {elapsed:.4f}s ({elapsed/len(images):.4f}s per image)")
    
    return results, elapsed


def benchmark_predict_batch(
    model: BasePoseModel,
    images: List[npt.NDArray[np.uint8]],
    bboxes: List[Tuple[float, float, float, float]] = None
) -> Tuple[List[StandardizedHeatmap], float]:
    """
    Benchmark batch predict_batch() call.
    
    Args:
        model: Model adapter instance.
        images: List of input images.
        bboxes: Optional list of bounding boxes.
    
    Returns:
        Tuple of (results, elapsed_time_seconds).
    """
    logger.info(f"Running predict_batch() for {len(images)} images...")
    
    start_time = time.perf_counter()
    results = model.predict_batch(images, bboxes)
    elapsed = time.perf_counter() - start_time
    
    logger.info(f"Batch predict_batch() took {elapsed:.4f}s ({elapsed/len(images):.4f}s per image)")
    
    return results, elapsed


def benchmark_predict_keypoints(
    model: BasePoseModel,
    images: List[npt.NDArray[np.uint8]],
    bboxes: List[Tuple[float, float, float, float]] = None
) -> Tuple[Tuple[List, List], float]:
    """
    Benchmark sequential predict_keypoints() calls.
    
    Args:
        model: Model adapter instance.
        images: List of input images.
        bboxes: Optional list of bounding boxes.
    
    Returns:
        Tuple of ((keypoints_list, scores_list), elapsed_time_seconds).
    """
    if bboxes is None:
        bboxes = [None] * len(images)
    
    logger.info(f"Running sequential predict_keypoints() for {len(images)} images...")
    
    keypoints_list = []
    scores_list = []
    
    start_time = time.perf_counter()
    for img, bbox in zip(images, bboxes):
        kpts, scores = model.predict_keypoints(img, bbox)
        keypoints_list.append(kpts)
        scores_list.append(scores)
    elapsed = time.perf_counter() - start_time
    
    logger.info(
        f"Sequential predict_keypoints() took {elapsed:.4f}s "
        f"({elapsed/len(images):.4f}s per image)"
    )
    
    return (keypoints_list, scores_list), elapsed


def benchmark_predict_keypoints_batch(
    model: BasePoseModel,
    images: List[npt.NDArray[np.uint8]],
    bboxes: List[Tuple[float, float, float, float]] = None
) -> Tuple[Tuple[List, List], float]:
    """
    Benchmark batch predict_keypoints_batch() call.
    
    Args:
        model: Model adapter instance.
        images: List of input images.
        bboxes: Optional list of bounding boxes.
    
    Returns:
        Tuple of ((keypoints_list, scores_list), elapsed_time_seconds).
    """
    logger.info(f"Running predict_keypoints_batch() for {len(images)} images...")
    
    start_time = time.perf_counter()
    keypoints_list, scores_list = model.predict_keypoints_batch(images, bboxes)
    elapsed = time.perf_counter() - start_time
    
    logger.info(
        f"Batch predict_keypoints_batch() took {elapsed:.4f}s "
        f"({elapsed/len(images):.4f}s per image)"
    )
    
    return (keypoints_list, scores_list), elapsed


def main() -> int:
    """
    Main verification and benchmarking routine.
    
    Returns:
        0 if all tests pass, 1 otherwise.
    """
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
    
    # Create model
    logger.info("\n[1/6] Creating model...")
    if args.model == "mock":
        model = MockPoseModel(mode="unimodal", input_size=(64, 48))
        logger.info("✓ MockPoseModel created")
    else:
        try:
            model = MMPoseAdapter.from_model_name(args.model, device=args.device)
            logger.info(f"✓ MMPoseAdapter ({args.model}) created")
            # Warmup for accurate timing
            logger.info("  Warming up model...")
            model.warmup(iterations=3)
            logger.info("  ✓ Warmup complete")
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            return 1
    
    # Generate test images
    logger.info("\n[2/6] Generating test images...")
    images = generate_dummy_images(
        num_images=args.num_images,
        image_size=tuple(args.image_size)
    )
    logger.info(f"✓ Generated {len(images)} images of shape {images[0].shape}")
    
    # Benchmark predict() vs predict_batch()
    logger.info("\n[3/6] Benchmarking predict() vs predict_batch()...")
    logger.info("-" * 80)
    
    heatmaps_seq, time_seq = benchmark_predict(model, images)
    heatmaps_batch, time_batch = benchmark_predict_batch(model, images)
    
    speedup = time_seq / time_batch if time_batch > 0 else float('inf')
    logger.info(f"\nSpeedup: {speedup:.2f}x (sequential: {time_seq:.4f}s, batch: {time_batch:.4f}s)")
    
    # Verify heatmaps are identical
    logger.info("\n[4/6] Verifying heatmap numerical identity...")
    logger.info("-" * 80)
    
    heatmaps_pass = verify_heatmaps_identical(
        heatmaps_seq, heatmaps_batch, atol=args.atol, rtol=args.rtol
    )
    
    if not heatmaps_pass:
        logger.error("✗ Heatmap verification FAILED")
        return 1
    
    # Benchmark predict_keypoints() vs predict_keypoints_batch()
    logger.info("\n[5/6] Benchmarking predict_keypoints() vs predict_keypoints_batch()...")
    logger.info("-" * 80)
    
    (kpts_seq, scores_seq), time_kpts_seq = benchmark_predict_keypoints(model, images)
    (kpts_batch, scores_batch), time_kpts_batch = benchmark_predict_keypoints_batch(model, images)
    
    speedup_kpts = time_kpts_seq / time_kpts_batch if time_kpts_batch > 0 else float('inf')
    logger.info(
        f"\nSpeedup: {speedup_kpts:.2f}x "
        f"(sequential: {time_kpts_seq:.4f}s, batch: {time_kpts_batch:.4f}s)"
    )
    
    # Verify keypoints are identical
    logger.info("\n[6/6] Verifying keypoint numerical identity...")
    logger.info("-" * 80)
    
    keypoints_pass = verify_keypoints_identical(
        kpts_seq, scores_seq, kpts_batch, scores_batch,
        atol=args.atol, rtol=args.rtol
    )
    
    if not keypoints_pass:
        logger.error("✗ Keypoint verification FAILED")
        return 1
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✓ All tests PASSED")
    logger.info(f"✓ predict_batch() speedup: {speedup:.2f}x")
    logger.info(f"✓ predict_keypoints_batch() speedup: {speedup_kpts:.2f}x")
    logger.info(f"✓ Numerical identity verified (atol={args.atol}, rtol={args.rtol})")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

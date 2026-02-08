"""
Quick Example: Batch Processing with MMPoseAdapter

This is a minimal working example demonstrating the new batch processing APIs.
Can be run directly with: python scripts/example_batch_usage.py
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from pose_uncertainty.models.base import MockPoseModel


def example_basic_usage():
    """Basic usage of batch processing APIs."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Batch Processing")
    print("=" * 80)
    
    # Create model (using MockModel for simplicity - no GPU needed)
    model = MockPoseModel(mode="unimodal", input_size=(64, 48))
    print("✓ Model created")
    
    # Generate dummy images (different sizes to show flexibility)
    images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
        np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8),
    ]
    print(f"✓ Generated {len(images)} test images")
    
    # Optional: specify bounding boxes (None = use full image)
    bboxes = [
        (100, 100, 200, 300),  # (x, y, w, h) in COCO format
        None,                   # Use full image
        (50, 50, 400, 400),
        None,
    ]
    
    # Method 1: Get heatmaps
    print("\n--- Method 1: predict_batch() ---")
    heatmaps = model.predict_batch(images, bboxes)
    
    for i, hm in enumerate(heatmaps):
        print(f"Image {i}:")
        print(f"  Shape: {hm.data.shape}")  # (17, 64, 48)
        print(f"  Original size: {hm.original_size}")
        print(f"  Reconstructed: {hm.reconstructed}")
    
    # Method 2: Get keypoints directly
    print("\n--- Method 2: predict_keypoints_batch() ---")
    keypoints_list, scores_list = model.predict_keypoints_batch(images, bboxes)
    
    for i, (kpts, scores) in enumerate(zip(keypoints_list, scores_list)):
        print(f"Image {i}:")
        print(f"  Keypoints shape: {kpts.shape}")  # (17, 2)
        print(f"  Mean confidence: {scores.mean():.3f}")
        print(f"  First 3 keypoints: {kpts[:3]}")


def example_consistency_check():
    """Verify that batch and sequential methods produce identical results."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Verify Numerical Identity")
    print("=" * 80)
    
    model = MockPoseModel(mode="unimodal", input_size=(64, 48))
    
    # Test images
    images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(4)
    ]
    
    # Sequential processing
    print("Running sequential predict()...")
    heatmaps_seq = [model.predict(img) for img in images]
    
    # Batch processing
    print("Running predict_batch()...")
    heatmaps_batch = model.predict_batch(images)
    
    # Verify identity
    print("\nVerifying results...")
    all_match = True
    for i, (hm_seq, hm_batch) in enumerate(zip(heatmaps_seq, heatmaps_batch)):
        if np.allclose(hm_seq.data, hm_batch.data, atol=1e-5):
            max_diff = np.abs(hm_seq.data - hm_batch.data).max()
            print(f"✓ Image {i}: MATCH (max_diff={max_diff:.2e})")
        else:
            print(f"✗ Image {i}: MISMATCH")
            all_match = False
    
    if all_match:
        print("\n✅ SUCCESS: All results are numerically identical!")
    else:
        print("\n❌ FAILURE: Some results differ!")


def example_performance():
    """Compare performance of sequential vs batch processing."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Performance Comparison")
    print("=" * 80)
    
    import time
    
    model = MockPoseModel(mode="unimodal", input_size=(64, 48))
    
    # Generate more images for realistic benchmark
    num_images = 16
    images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(num_images)
    ]
    
    print(f"Processing {num_images} images...\n")
    
    # Sequential
    start = time.perf_counter()
    heatmaps_seq = [model.predict(img) for img in images]
    time_seq = time.perf_counter() - start
    
    print(f"Sequential predict():")
    print(f"  Total time: {time_seq:.4f}s")
    print(f"  Per image: {time_seq/num_images:.4f}s")
    
    # Batch
    start = time.perf_counter()
    heatmaps_batch = model.predict_batch(images)
    time_batch = time.perf_counter() - start
    
    print(f"\nBatch predict_batch():")
    print(f"  Total time: {time_batch:.4f}s")
    print(f"  Per image: {time_batch/num_images:.4f}s")
    
    # Speedup
    speedup = time_seq / time_batch if time_batch > 0 else float('inf')
    print(f"\n📊 Speedup: {speedup:.2f}x")
    
    # Note about GPU
    print("\n💡 Note: This example uses MockModel (CPU).")
    print("   With a real model on GPU, expect 2-5x speedup for batch_size >= 4")


def example_with_real_model():
    """
    Example with real MMPoseAdapter (requires GPU and dependencies).
    This example will only run if mmpose is installed.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Real Model (Optional - requires mmpose)")
    print("=" * 80)
    
    try:
        from pose_uncertainty.models.adapters import MMPoseAdapter
    except ImportError:
        print("⚠️  MMPose not installed. Skipping real model example.")
        print("   Install with: pip install mmpose")
        return
    
    try:
        # Try to create model
        print("Loading HRNet-W32 model...")
        model = MMPoseAdapter.from_model_name("hrnet_w32", device="cuda")
        print("✓ Model loaded successfully")
        
        # Warmup
        print("Warming up model...")
        model.warmup(iterations=3)
        
        # Test images
        images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(8)
        ]
        
        # Batch inference
        import time
        print(f"\nRunning batch inference on {len(images)} images...")
        start = time.perf_counter()
        heatmaps = model.predict_batch(images)
        elapsed = time.perf_counter() - start
        
        print(f"✓ Completed in {elapsed:.4f}s ({elapsed/len(images):.4f}s per image)")
        print(f"✓ Output: {len(heatmaps)} heatmaps of shape {heatmaps[0].data.shape}")
        
    except FileNotFoundError as e:
        print(f"⚠️  Model files not found: {e}")
        print("   Ensure model weights are in models/weights/")
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_consistency_check()
    example_performance()
    
    # Optional: try with real model
    try:
        example_with_real_model()
    except Exception as e:
        print(f"\n⚠️  Could not run real model example: {e}")
    
    print("\n" + "=" * 80)
    print("✅ All examples completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run full verification: python scripts/verify_batch_processing.py")
    print("  2. Try with real model: python scripts/verify_batch_processing.py --model hrnet_w32 --device cuda")
    print("  3. Read documentation: docs/BATCH_PROCESSING_GUIDE.md")

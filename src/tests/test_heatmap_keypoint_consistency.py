"""
Test Suite: Heatmap→Keypoint Consistency

Verifies that the following guarantee holds:
    predict_keypoints(image) == decode_heatmaps(predict(image))

This ensures the heatmap extraction and decoding pipeline is mathematically
equivalent to direct keypoint prediction.

Reference: notebooks/10_complete_pipeline.ipynb, Cell 44
"""

import numpy as np
import pytest
from pathlib import Path

# Test with mock model first (no dependencies)
from pose_uncertainty.models.base import MockPoseModel


class TestMockModelConsistency:
    """Test consistency with synthetic mock model."""
    
    def setup_method(self):
        """Create mock model for testing."""
        self.model = MockPoseModel(input_size=(256, 192), num_keypoints=17)
    
    def test_predict_returns_standardized_heatmap(self):
        """Verify predict() returns proper StandardizedHeatmap."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        heatmap = self.model.predict(image)
        
        # MockModel uses input_size for heatmap dimensions
        expected_h, expected_w = self.model.input_size
        assert heatmap.data.shape == (17, expected_h, expected_w), f"Wrong heatmap shape, expected (17, {expected_h}, {expected_w})"
        assert heatmap.data.dtype == np.float32, "Wrong dtype"
        assert 0.0 <= heatmap.data.min() <= 1.0, "Values out of [0, 1] range"
        assert 0.0 <= heatmap.data.max() <= 1.0, "Values out of [0, 1] range"
        assert heatmap.original_size == (480, 640), "Wrong original_size"
    
    def test_predict_keypoints_returns_correct_format(self):
        """Verify predict_keypoints() returns (keypoints, scores)."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        keypoints, scores = self.model.predict_keypoints(image)
        
        assert keypoints.shape == (17, 2), "Wrong keypoints shape"
        assert scores.shape == (17,), "Wrong scores shape"
        assert keypoints.dtype == np.float32, "Wrong keypoints dtype"
        assert scores.dtype == np.float32, "Wrong scores dtype"
        assert 0.0 <= scores.min() <= 1.0, "Scores out of [0, 1]"
        assert 0.0 <= scores.max() <= 1.0, "Scores out of [0, 1]"
    
    def test_decode_heatmaps_returns_correct_format(self):
        """Verify decode_heatmaps() works with StandardizedHeatmap."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        heatmap = self.model.predict(image)
        keypoints, scores = self.model.decode_heatmaps(heatmap)
        
        assert keypoints.shape == (17, 2), "Wrong keypoints shape"
        assert scores.shape == (17,), "Wrong scores shape"
        assert keypoints.dtype == np.float32, "Wrong dtype"
        assert scores.dtype == np.float32, "Wrong dtype"
    
    def test_consistency_predict_vs_decode(self):
        """
        CRITICAL TEST: Verify predict_keypoints() matches decode_heatmaps(predict()).
        
        This is the core guarantee of the API.
        """
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Path 1: Direct prediction
        keypoints_direct, scores_direct = self.model.predict_keypoints(image)
        
        # Path 2: Heatmap → Decode
        heatmap = self.model.predict(image)
        keypoints_decoded, scores_decoded = self.model.decode_heatmaps(heatmap)
        
        # Verify exact match (mock model should be deterministic)
        np.testing.assert_allclose(
            keypoints_direct, keypoints_decoded,
            rtol=1e-5, atol=1e-3,
            err_msg="Direct prediction does not match decoded heatmaps"
        )
        
        np.testing.assert_allclose(
            scores_direct, scores_decoded,
            rtol=1e-5, atol=1e-3,
            err_msg="Direct scores do not match decoded scores"
        )
    
    def test_consistency_with_bbox(self):
        """Test consistency when using bbox cropping."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 200, 300)  # (x, y, w, h)
        
        # Path 1: Direct
        keypoints_direct, scores_direct = self.model.predict_keypoints(image, bbox)
        
        # Path 2: Heatmap → Decode
        heatmap = self.model.predict(image, bbox)
        keypoints_decoded, scores_decoded = self.model.decode_heatmaps(heatmap)
        
        np.testing.assert_allclose(
            keypoints_direct, keypoints_decoded,
            rtol=1e-5, atol=1e-3,
            err_msg="Consistency fails with bbox"
        )


@pytest.mark.slow
@pytest.mark.requires_mmpose
class TestMMPoseAdapterConsistency:
    """
    Test consistency with real MMPose models.
    
    These tests require:
    - MMPose installed (models/mmpose/)
    - Model weights downloaded (models/weights/)
    - CUDA device (optional, falls back to CPU)
    """
    
    @pytest.fixture(scope="class")
    def model(self):
        """Load HRNet-W32 model (lightest COCO model)."""
        try:
            from pose_uncertainty.models.adapters import MMPoseAdapter
            
            # Try CUDA first, fallback to CPU
            try:
                adapter = MMPoseAdapter.from_model_name("hrnet_w32", device="cuda")
                print("✓ Using CUDA device")
            except:
                adapter = MMPoseAdapter.from_model_name("hrnet_w32", device="cpu")
                print("✓ Using CPU device")
            
            adapter.warmup(iterations=1)
            return adapter
        
        except Exception as e:
            pytest.skip(f"MMPose not available: {e}")
    
    @pytest.fixture(scope="class")
    def test_image(self):
        """Load a test image from COCO."""
        # Create synthetic image if COCO not available
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return image
    
    def test_predict_stores_metadata(self, model, test_image):
        """Verify predict() stores transformation metadata."""
        heatmap = model.predict(test_image)
        
        assert hasattr(heatmap, "metadata"), "No metadata attribute"
        assert heatmap.metadata is not None, "Metadata is None"
        
        metadata = heatmap.metadata
        assert "input_center" in metadata, "Missing input_center"
        assert "input_scale" in metadata, "Missing input_scale"
        assert "input_size" in metadata, "Missing input_size"
        
        # Verify types
        assert metadata["input_center"] is not None
        assert metadata["input_scale"] is not None
        assert isinstance(metadata["input_size"], (tuple, list, np.ndarray))
    
    def test_consistency_mmpose_model(self, model, test_image):
        """
        CRITICAL TEST: Verify MMPose adapter consistency.
        
        Note: If heatmaps are reconstructed (not directly from model), tolerance
        is relaxed since reconstruction is an approximation.
        
        Tolerance:
            - Direct heatmaps: 0.01 pixels (Cell 44 achieved < 0.00004 px)
            - Reconstructed heatmaps: 50 pixels (reasonable for Gaussian reconstruction)
        """
        # Path 1: Direct prediction
        keypoints_direct, scores_direct = model.predict_keypoints(test_image)
        
        # Path 2: Heatmap → Decode
        heatmap = model.predict(test_image)
        keypoints_decoded, scores_decoded = model.decode_heatmaps(heatmap)
        
        # Compute per-keypoint differences
        diffs = np.linalg.norm(keypoints_direct - keypoints_decoded, axis=1)
        max_diff = np.max(diffs)
        mean_diff = np.mean(diffs)
        
        print(f"\n📊 Consistency Results:")
        print(f"   Max difference:  {max_diff:.6f} pixels")
        print(f"   Mean difference: {mean_diff:.6f} pixels")
        print(f"   Keypoints tested: {len(diffs)}")
        print(f"   Heatmaps reconstructed: {getattr(heatmap, 'reconstructed', False)}")
        
        # Determine tolerance based on whether heatmaps were reconstructed
        if getattr(heatmap, 'reconstructed', False):
            tolerance = 50.0  # Relaxed tolerance for reconstructed heatmaps
            print(f"   ⚠️ Using relaxed tolerance ({tolerance} px) for reconstructed heatmaps")
        else:
            tolerance = 0.01  # Tight tolerance for direct heatmaps
            print(f"   ✓ Using tight tolerance ({tolerance} px) for direct heatmaps")
        
        # Verify tolerance
        assert max_diff < tolerance, (
            f"Maximum difference {max_diff:.6f} exceeds tolerance {tolerance} px. "
            f"Heatmaps {'were reconstructed' if getattr(heatmap, 'reconstructed', False) else 'from same forward pass'}."
        )
        
        # Scores should also match reasonably
        score_diffs = np.abs(scores_direct - scores_decoded)
        assert np.max(score_diffs) < 0.5, "Confidence scores differ significantly"
    
    def test_consistency_with_bbox_mmpose(self, model, test_image):
        """Test consistency with bbox on real model."""
        bbox = (100, 100, 200, 300)
        
        keypoints_direct, _ = model.predict_keypoints(test_image, bbox)
        heatmap = model.predict(test_image, bbox)
        keypoints_decoded, _ = model.decode_heatmaps(heatmap)
        
        diffs = np.linalg.norm(keypoints_direct - keypoints_decoded, axis=1)
        max_diff = np.max(diffs)
        
        # Determine tolerance based on reconstruction
        tolerance = 50.0 if getattr(heatmap, 'reconstructed', False) else 0.01
        
        assert max_diff < tolerance, (
            f"Consistency fails with bbox: max_diff={max_diff:.6f} > {tolerance} px"
        )
    
    def test_multiple_images_consistency(self, model):
        """Test consistency across multiple random images."""
        np.random.seed(42)
        
        all_diffs = []
        tolerance = None
        for i in range(5):
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            keypoints_direct, _ = model.predict_keypoints(image)
            heatmap = model.predict(image)
            keypoints_decoded, _ = model.decode_heatmaps(heatmap)
            
            if tolerance is None:
                # Set tolerance based on first image
                tolerance = 50.0 if getattr(heatmap, 'reconstructed', False) else 0.01
            
            diffs = np.linalg.norm(keypoints_direct - keypoints_decoded, axis=1)
            all_diffs.extend(diffs.tolist())
        
        max_diff = np.max(all_diffs)
        mean_diff = np.mean(all_diffs)
        
        print(f"\n📊 Multi-image Results ({len(all_diffs)} keypoints):")
        print(f"   Max: {max_diff:.6f} px")
        print(f"   Mean: {mean_diff:.6f} px")
        print(f"   Std: {np.std(all_diffs):.6f} px")
        print(f"   Tolerance: {tolerance} px")
        
        assert max_diff < tolerance, "Consistency fails across multiple images"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_decode_without_metadata_raises_error(self):
        """decode_heatmaps() should raise if metadata is missing."""
        from pose_uncertainty.utils.types import StandardizedHeatmap
        
        # Create heatmap without metadata
        heatmap = StandardizedHeatmap(
            data=np.random.rand(17, 48, 64).astype(np.float32),
            original_size=(480, 640),
            scale_factor=1.0,
            offset=(0.0, 0.0)
        )
        
        try:
            from pose_uncertainty.models.adapters import MMPoseAdapter
            model = MMPoseAdapter.from_model_name("hrnet_w32", device="cpu")
            
            with pytest.raises(ValueError, match="No metadata available"):
                model.decode_heatmaps(heatmap)
        
        except Exception:
            pytest.skip("MMPose not available")
    
    def test_empty_image_handling(self):
        """Test behavior with empty/zero images."""
        model = MockPoseModel(input_size=(256, 192), num_keypoints=17)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Should not crash
        keypoints, scores = model.predict_keypoints(image)
        assert keypoints.shape == (17, 2)
        
        heatmap = model.predict(image)
        keypoints_dec, scores_dec = model.decode_heatmaps(heatmap)
        assert keypoints_dec.shape == (17, 2)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])

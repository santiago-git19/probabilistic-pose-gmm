"""
Round-trip verification for coordinate transformations.

Verifies that the image→heatmap→image round-trip produces
(practically) identical coordinates, validating that
``transform_image_coords_to_heatmap`` is the exact mathematical
inverse of ``transform_heatmap_coords_to_image``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pose_uncertainty.models.adapters import MMPoseAdapter
from pose_uncertainty.utils.types import StandardizedHeatmap


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _make_heatmap_with_metadata(
    input_center, input_scale, input_size=(192, 256),
    hm_shape=(17, 64, 48),
) -> StandardizedHeatmap:
    """Build a dummy StandardizedHeatmap with realistic MMPose metadata."""
    data = np.random.rand(*hm_shape).astype(np.float32)
    # Normalise to [0, 1]
    for k in range(data.shape[0]):
        mn, mx = data[k].min(), data[k].max()
        if mx > mn:
            data[k] = (data[k] - mn) / (mx - mn)
    return StandardizedHeatmap(
        data=data,
        original_size=(480, 640),
        metadata={
            "input_center": list(input_center),
            "input_scale": list(input_scale),
            "input_size": list(input_size),
        },
    )


# --------------------------------------------------------------------------
# Test: Round-trip  image → heatmap → image
# --------------------------------------------------------------------------

class TestCoordinateRoundTrip:
    """Validates image↔heatmap coordinate transformations are exact inverses."""

    @pytest.fixture(params=[
        # (input_center, input_scale) — varied realistic scenarios
        ([320.0, 240.0], [200.0, 300.0]),   # centred bbox
        ([100.0, 80.0],  [150.0, 250.0]),   # top-left bbox
        ([500.0, 400.0], [100.0, 180.0]),   # bottom-right, small bbox
        ([250.5, 310.7], [175.3, 290.1]),   # non-integer values
    ])
    def heatmap(self, request):
        center, scale = request.param
        return _make_heatmap_with_metadata(center, scale)

    def test_single_point_roundtrip(self, heatmap):
        """One point: image → heatmap → image ≈ original."""
        img_pt = np.array([150.0, 200.0], dtype=np.float32)

        hm_pt = MMPoseAdapter.transform_image_coords_to_heatmap(img_pt, heatmap)
        recovered = MMPoseAdapter.transform_heatmap_coords_to_image(hm_pt, heatmap)

        np.testing.assert_allclose(recovered, img_pt, atol=1e-3,
                                   err_msg="Single-point round trip failed")

    def test_batch_roundtrip(self, heatmap):
        """N points: image → heatmap → image ≈ original."""
        img_pts = np.array([
            [100.0, 150.0],
            [320.5, 240.3],
            [0.0,   0.0],
            [639.0, 479.0],
        ], dtype=np.float32)

        hm_pts = MMPoseAdapter.transform_image_coords_to_heatmap(img_pts, heatmap)
        recovered = MMPoseAdapter.transform_heatmap_coords_to_image(hm_pts, heatmap)

        np.testing.assert_allclose(recovered, img_pts, atol=1e-3,
                                   err_msg="Batch round trip failed")

    def test_heatmap_then_image_roundtrip(self, heatmap):
        """Start from heatmap space: hm → image → hm ≈ original."""
        hm_pts = np.array([
            [24.0, 32.0],
            [10.5, 50.2],
            [0.0,   0.0],
            [47.0, 63.0],
        ], dtype=np.float32)

        img_pts = MMPoseAdapter.transform_heatmap_coords_to_image(hm_pts, heatmap)
        recovered = MMPoseAdapter.transform_image_coords_to_heatmap(img_pts, heatmap)

        np.testing.assert_allclose(recovered, hm_pts, atol=1e-3,
                                   err_msg="Heatmap→Image→Heatmap round trip failed")

    def test_known_corners(self):
        """Heatmap origin (0,0) and corner should map consistently."""
        hm = _make_heatmap_with_metadata(
            input_center=[320.0, 240.0],
            input_scale=[200.0, 300.0],
        )
        # Heatmap (0,0) → image
        origin_img = MMPoseAdapter.transform_heatmap_coords_to_image(
            np.array([0.0, 0.0], dtype=np.float32), hm
        )
        # Round-trip
        origin_back = MMPoseAdapter.transform_image_coords_to_heatmap(origin_img, hm)
        np.testing.assert_allclose(origin_back, [0.0, 0.0], atol=1e-4)

        # Heatmap corner (47, 63) → image
        corner_img = MMPoseAdapter.transform_heatmap_coords_to_image(
            np.array([47.0, 63.0], dtype=np.float32), hm
        )
        corner_back = MMPoseAdapter.transform_image_coords_to_heatmap(corner_img, hm)
        np.testing.assert_allclose(corner_back, [47.0, 63.0], atol=1e-4)

    def test_gt_realistic_keypoints(self):
        """Simulate 17 COCO GT keypoints through the round-trip."""
        rng = np.random.default_rng(42)
        gt_img = rng.uniform(50, 600, size=(17, 2)).astype(np.float32)

        hm = _make_heatmap_with_metadata(
            input_center=[300.0, 250.0],
            input_scale=[250.0, 350.0],
        )

        gt_hm = MMPoseAdapter.transform_image_coords_to_heatmap(gt_img, hm)
        gt_recovered = MMPoseAdapter.transform_heatmap_coords_to_image(gt_hm, hm)

        np.testing.assert_allclose(gt_recovered, gt_img, atol=1e-3,
                                   err_msg="17-keypoint GT round trip failed")

    def test_affine_consistency(self):
        """Verify the static method _heatmap_to_image_affine and the
        inverse transform are algebraically consistent."""
        hm = _make_heatmap_with_metadata(
            input_center=[320.0, 240.0],
            input_scale=[200.0, 300.0],
        )
        A, b = MMPoseAdapter._heatmap_to_image_affine(hm)

        # Manual inverse: A_inv = diag(1/A_ii), b_inv = -A_inv @ b
        A_inv = np.diag(1.0 / np.diag(A))

        pts_hm = np.array([[10.0, 20.0], [30.0, 50.0]], dtype=np.float32)

        # Forward via affine
        pts_img_affine = (pts_hm @ A.T + b).astype(np.float32)
        # Forward via method
        pts_img_method = MMPoseAdapter.transform_heatmap_coords_to_image(pts_hm, hm)
        np.testing.assert_allclose(pts_img_method, pts_img_affine, atol=1e-4)

        # Inverse via method
        pts_hm_back = MMPoseAdapter.transform_image_coords_to_heatmap(pts_img_method, hm)
        np.testing.assert_allclose(pts_hm_back, pts_hm, atol=1e-3)

    def test_missing_metadata_raises(self):
        """Both methods must raise ValueError when metadata is None."""
        bad_hm = StandardizedHeatmap(
            data=np.zeros((17, 64, 48), dtype=np.float32),
            original_size=(480, 640),
            metadata=None,
        )
        pt = np.array([10.0, 20.0], dtype=np.float32)

        with pytest.raises(ValueError, match="metadata is None"):
            MMPoseAdapter.transform_image_coords_to_heatmap(pt, bad_hm)
        with pytest.raises(ValueError, match="metadata is None"):
            MMPoseAdapter.transform_heatmap_coords_to_image(pt, bad_hm)

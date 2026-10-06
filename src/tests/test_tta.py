"""
Exhaustive test suite for the TTA (Test-Time Augmentation) engine.

Covers:
    1. Configuration & combinatorics (batch sizes for every config combo).
    2. Data integrity (dtype, shape preservation).
    3. Inverse-flip mathematics (spatial flip + channel swap + pixel shift).
    4. Edge cases & determinism.

Run with::

    pytest src/tests/test_tta.py -v
"""

from __future__ import annotations

import copy
from typing import List, Tuple

import cv2
import numpy as np
import pytest

# ── Module under test ──────────────────────────────────────────────────
from pose_uncertainty.pipeline.tta import TTAEngine, TTAMetadata


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def dummy_image() -> np.ndarray:
    """128×96 synthetic RGB image (uint8) with a gradient pattern."""
    h, w = 128, 96
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.arange(w, dtype=np.uint8)[None, :]  # R channel gradient
    img[:, :, 1] = np.arange(h, dtype=np.uint8)[:, None]  # G channel gradient
    img[:, :, 2] = 128  # constant B
    return img


@pytest.fixture
def small_image() -> np.ndarray:
    """10×10 tiny image for quick pixel-level checks."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(10, 10, 3), dtype=np.uint8)


def _make_config(**overrides) -> dict:
    """Return a minimal config dict, merging *overrides* into defaults."""
    cfg: dict = {
        "flip": {"enabled": False},
        "photometric": {
            "brightness": {"enabled": False, "delta": 30.0},
            "contrast": {"enabled": False, "range": [0.8, 1.2]},
            "noise": {"enabled": False, "sigma": 10.0},
            "blur": {"enabled": False, "kernel_size": 3},
        },
        "seed": 42,
    }
    for key, val in overrides.items():
        # Allow flat keys like "flip.enabled"
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return cfg


# =====================================================================
# 1. Configuration & Combinatorics
# =====================================================================

class TestConfigurationCombinatorics:
    """Verify that prepare_batch produces the correct number of variants."""

    def test_empty_config_returns_only_original(self, dummy_image: np.ndarray):
        """All augmentations disabled → single 'original' image."""
        engine = TTAEngine(_make_config())
        imgs, metas = engine.prepare_batch(dummy_image)
        assert len(imgs) == 1
        assert len(metas) == 1
        assert metas[0].is_flipped is False
        assert metas[0].transform_type == "original"

    def test_flip_only(self, dummy_image: np.ndarray):
        """Flip enabled, no photometric → 2 images."""
        cfg = _make_config(**{"flip.enabled": True})
        engine = TTAEngine(cfg)
        imgs, metas = engine.prepare_batch(dummy_image)

        assert len(imgs) == 2
        types = [m.transform_type for m in metas]
        assert "original" in types
        assert "flip" in types
        assert metas[0].is_flipped is False
        assert metas[1].is_flipped is True

    def test_noise_only(self, dummy_image: np.ndarray):
        """No flip, noise enabled → 2 images (original + noise variant)."""
        cfg = _make_config(**{"photometric.noise.enabled": True})
        engine = TTAEngine(cfg)
        imgs, metas = engine.prepare_batch(dummy_image)

        assert len(imgs) == 2
        types = [m.transform_type for m in metas]
        assert "original" in types
        assert "original+noise" in types

    def test_flip_plus_noise_cartesian(self, dummy_image: np.ndarray):
        """Flip + noise → 4 images (2 geo × 2 photo)."""
        cfg = _make_config(**{
            "flip.enabled": True,
            "photometric.noise.enabled": True,
        })
        engine = TTAEngine(cfg)
        imgs, metas = engine.prepare_batch(dummy_image)

        assert len(imgs) == 4
        expected_types = {"original", "original+noise", "flip", "flip+noise"}
        actual_types = {m.transform_type for m in metas}
        assert actual_types == expected_types

    def test_flip_plus_two_photometric(self, dummy_image: np.ndarray):
        """Flip + brightness + noise → 6 images (2 geo × 3 photo)."""
        cfg = _make_config(**{
            "flip.enabled": True,
            "photometric.brightness.enabled": True,
            "photometric.noise.enabled": True,
        })
        engine = TTAEngine(cfg)
        imgs, metas = engine.prepare_batch(dummy_image)

        assert len(imgs) == 6
        assert sum(m.is_flipped for m in metas) == 3  # half flipped

    def test_all_photometric_enabled(self, dummy_image: np.ndarray):
        """All 4 photometric + flip → 10 images (2 geo × 5 photo)."""
        cfg = _make_config(**{
            "flip.enabled": True,
            "photometric.brightness.enabled": True,
            "photometric.contrast.enabled": True,
            "photometric.noise.enabled": True,
            "photometric.blur.enabled": True,
        })
        engine = TTAEngine(cfg)
        imgs, metas = engine.prepare_batch(dummy_image)
        assert len(imgs) == 10

    def test_photometric_params_recorded_in_metadata(self, dummy_image: np.ndarray):
        """Photometric params must appear in TTAMetadata for traceability."""
        cfg = _make_config(**{
            "photometric.noise.enabled": True,
            "photometric.noise.sigma": 15.0,
        })
        engine = TTAEngine(cfg)
        _, metas = engine.prepare_batch(dummy_image)

        noise_meta = [m for m in metas if "noise" in m.transform_type]
        assert len(noise_meta) == 1
        assert noise_meta[0].photometric_params is not None
        assert noise_meta[0].photometric_params["sigma"] == 15.0


# =====================================================================
# 2. Data Integrity
# =====================================================================

class TestDataIntegrity:
    """Ensure dtype, shape, and value range are preserved."""

    def test_dtype_preserved_uint8(self, dummy_image: np.ndarray):
        cfg = _make_config(**{
            "flip.enabled": True,
            "photometric.brightness.enabled": True,
            "photometric.noise.enabled": True,
            "photometric.blur.enabled": True,
        })
        engine = TTAEngine(cfg)
        imgs, _ = engine.prepare_batch(dummy_image)
        for img in imgs:
            assert img.dtype == np.uint8, f"Expected uint8, got {img.dtype}"

    def test_shape_preserved(self, dummy_image: np.ndarray):
        h, w, c = dummy_image.shape
        cfg = _make_config(**{
            "flip.enabled": True,
            "photometric.noise.enabled": True,
            "photometric.blur.enabled": True,
            "photometric.contrast.enabled": True,
            "photometric.brightness.enabled": True,
        })
        engine = TTAEngine(cfg)
        imgs, _ = engine.prepare_batch(dummy_image)
        for img in imgs:
            assert img.shape == (h, w, c)

    def test_values_in_range(self, dummy_image: np.ndarray):
        cfg = _make_config(**{
            "photometric.brightness.enabled": True,
            "photometric.brightness.delta": 200.0,  # extreme
            "photometric.noise.enabled": True,
            "photometric.noise.sigma": 100.0,  # extreme
        })
        engine = TTAEngine(cfg)
        imgs, _ = engine.prepare_batch(dummy_image)
        for img in imgs:
            assert img.min() >= 0 and img.max() <= 255

    def test_original_image_unchanged(self, dummy_image: np.ndarray):
        """Augmentation must not mutate the caller's array."""
        original_copy = dummy_image.copy()
        cfg = _make_config(**{
            "flip.enabled": True,
            "photometric.noise.enabled": True,
        })
        engine = TTAEngine(cfg)
        engine.prepare_batch(dummy_image)
        np.testing.assert_array_equal(dummy_image, original_copy)


# =====================================================================
# 3. Flip correctness (image-level)
# =====================================================================

class TestFlipCorrectness:
    """Verify pixel-level correctness of the horizontal flip."""

    def test_flip_matches_opencv(self, small_image: np.ndarray):
        cfg = _make_config(**{"flip.enabled": True})
        engine = TTAEngine(cfg)
        imgs, metas = engine.prepare_batch(small_image)

        flipped_imgs = [img for img, m in zip(imgs, metas) if m.is_flipped]
        assert len(flipped_imgs) == 1
        expected = cv2.flip(small_image, 1)
        np.testing.assert_array_equal(flipped_imgs[0], expected)

    def test_double_flip_roundtrip(self, dummy_image: np.ndarray):
        """Flipping twice recovers the original image."""
        flipped_once = TTAEngine._apply_flip(dummy_image)
        flipped_twice = TTAEngine._apply_flip(flipped_once)
        np.testing.assert_array_equal(flipped_twice, dummy_image)


# =====================================================================
# 4. Inverse flip – Heatmap mathematics  (CRITICAL)
# =====================================================================

class TestInverseFlipHeatmap:
    """Verify ``inverse_flip_heatmap`` replicates MMPose behaviour."""

    @staticmethod
    def _synthetic_heatmap(
        n_keypoints: int, height: int, width: int
    ) -> np.ndarray:
        """All-zeros heatmap; caller places peaks manually."""
        return np.zeros((n_keypoints, height, width), dtype=np.float32)

    # ---- basic spatial flip ---- #

    def test_spatial_flip_without_shift(self):
        """Value at (k=0, y=0, x=2) moves to (k=0, y=0, W-1-2) after flip."""
        K, H, W = 2, 8, 10
        hm = self._synthetic_heatmap(K, H, W)
        hm[0, 0, 2] = 1.0  # peak at x=2

        result = TTAEngine.inverse_flip_heatmap(
            hm, flip_pairs=[], shift_heatmap=False
        )
        # After spatial flip: x_new = W-1-2 = 7
        assert result[0, 0, 7] == 1.0

    # ---- channel swap ---- #

    def test_channel_swap(self):
        """flip_pairs=[(0,1)] should swap channels 0 and 1."""
        K, H, W = 2, 8, 10
        hm = self._synthetic_heatmap(K, H, W)
        hm[0, 3, 5] = 1.0  # peak only in channel 0

        result = TTAEngine.inverse_flip_heatmap(
            hm, flip_pairs=[(0, 1)], shift_heatmap=False,
        )
        # spatial flip moves x=5 → x=4 (W-1-5 = 4)
        # channel swap moves channel 0 → channel 1
        assert result[1, 3, 4] == 1.0
        assert result[0, 3, 4] == 0.0  # channel 0 must be empty

    # ---- 1-pixel alignment shift ---- #

    def test_pixel_shift_applied(self):
        """Shift must move content 1 pixel to the right; col-0 becomes 0."""
        K, H, W = 1, 4, 10
        hm = self._synthetic_heatmap(K, H, W)
        hm[0, 0, 2] = 1.0  # peak at x=2

        result = TTAEngine.inverse_flip_heatmap(
            hm, flip_pairs=[], shift_heatmap=True,
        )
        # spatial flip: x=2 → x=7
        # then shift right by 1: x=7 → x=8,  col-0 = 0
        assert result[0, 0, 8] == 1.0
        assert result[0, 0, 0] == 0.0

    # ---- combined: swap + shift ---- #

    def test_full_inverse_flip_logic(self):
        """Complete test: spatial flip + channel swap + shift.

        Setup
        -----
        heatmap shape (2, 8, 10):
          channel 0, row 0, col 5 = 1.0  (all other zeros)

        flip_pairs = [(0, 1)]

        Expected steps:
        1. Spatial flip (W=10): col 5 → col 4
        2. Channel swap: chan 0 → chan 1
        3. Shift right: col 4 → col 5, col 0 zeroed

        Final: channel 1, row 0, col 5 = 1.0
        """
        K, H, W = 2, 8, 10
        hm = self._synthetic_heatmap(K, H, W)
        hm[0, 0, 5] = 1.0

        result = TTAEngine.inverse_flip_heatmap(
            hm, flip_pairs=[(0, 1)], shift_heatmap=True,
        )
        assert result[1, 0, 5] == 1.0
        # original channel should be zero at that position
        assert result[0, 0, 5] == 0.0
        # leftmost column is zeroed
        assert np.all(result[..., 0] == 0.0)

    # ---- COCO-style multi-pair ---- #

    def test_coco_flip_pairs(self):
        """Verify with realistic COCO-style flip_pairs (8 pairs)."""
        coco_pairs = [
            (1, 2), (3, 4), (5, 6), (7, 8),
            (9, 10), (11, 12), (13, 14), (15, 16),
        ]
        K, H, W = 17, 64, 48
        hm = self._synthetic_heatmap(K, H, W)
        hm[1, 10, 20] = 1.0   # left_eye
        hm[5, 30, 10] = 1.0   # left_shoulder

        result = TTAEngine.inverse_flip_heatmap(
            hm, flip_pairs=coco_pairs, shift_heatmap=True,
        )

        # Channel 1 (left_eye) should move to channel 2 (right_eye)
        # Spatial flip x=20 → 47-20=27, shift → 28
        assert result[2, 10, 28] == 1.0
        assert result[1, 10, 28] == 0.0

        # Channel 5 (left_shoulder) → channel 6 (right_shoulder)
        # Spatial flip x=10 → 37, shift → 38
        assert result[6, 30, 38] == 1.0

    # ---- no-flip-pairs path ---- #

    def test_empty_flip_pairs(self):
        """Works correctly when no pairs are provided (symmetric kps only)."""
        K, H, W = 1, 4, 10
        hm = self._synthetic_heatmap(K, H, W)
        hm[0, 2, 3] = 1.0
        result = TTAEngine.inverse_flip_heatmap(hm, flip_pairs=[], shift_heatmap=True)
        # flip: x=3 → 6, shift → 7
        assert result[0, 2, 7] == 1.0

    # ---- energy conservation ---- #

    def test_heatmap_sum_preserved_no_shift(self):
        """Total 'energy' is preserved without the shift operation."""
        K, H, W = 3, 16, 16
        rng = np.random.default_rng(7)
        hm = rng.random((K, H, W)).astype(np.float32)
        total_before = hm.sum()

        result = TTAEngine.inverse_flip_heatmap(
            hm, flip_pairs=[(0, 1)], shift_heatmap=False,
        )
        np.testing.assert_allclose(result.sum(), total_before, rtol=1e-5)

    # ---- dtype preservation ---- #

    def test_output_dtype_float32(self):
        K, H, W = 2, 8, 10
        hm = self._synthetic_heatmap(K, H, W)
        result = TTAEngine.inverse_flip_heatmap(hm, flip_pairs=[(0, 1)])
        assert result.dtype == np.float32

    def test_input_not_mutated(self):
        """inverse_flip_heatmap must not modify the input array."""
        K, H, W = 2, 8, 10
        hm = self._synthetic_heatmap(K, H, W)
        hm[0, 0, 5] = 1.0
        hm_copy = hm.copy()

        TTAEngine.inverse_flip_heatmap(hm, flip_pairs=[(0, 1)])
        np.testing.assert_array_equal(hm, hm_copy)


# =====================================================================
# 5. Determinism & Reproducibility
# =====================================================================

class TestDeterminism:
    """With a fixed seed, results must be bit-identical across runs."""

    def test_seeded_noise_deterministic(self, dummy_image: np.ndarray):
        cfg = _make_config(**{
            "photometric.noise.enabled": True,
            "seed": 123,
        })
        imgs_a, _ = TTAEngine(cfg).prepare_batch(dummy_image)
        imgs_b, _ = TTAEngine(cfg).prepare_batch(dummy_image)

        for a, b in zip(imgs_a, imgs_b):
            np.testing.assert_array_equal(a, b)

    def test_seeded_brightness_deterministic(self, dummy_image: np.ndarray):
        cfg = _make_config(**{
            "photometric.brightness.enabled": True,
            "seed": 99,
        })
        imgs_a, _ = TTAEngine(cfg).prepare_batch(dummy_image)
        imgs_b, _ = TTAEngine(cfg).prepare_batch(dummy_image)
        for a, b in zip(imgs_a, imgs_b):
            np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self, dummy_image: np.ndarray):
        cfg_a = _make_config(**{"photometric.noise.enabled": True, "seed": 1})
        cfg_b = _make_config(**{"photometric.noise.enabled": True, "seed": 2})
        imgs_a, _ = TTAEngine(cfg_a).prepare_batch(dummy_image)
        imgs_b, _ = TTAEngine(cfg_b).prepare_batch(dummy_image)
        # at least the noise image should differ
        assert not np.array_equal(imgs_a[1], imgs_b[1])


# =====================================================================
# 6. TTAMetadata dataclass
# =====================================================================

class TestTTAMetadata:
    """Frozen dataclass properties."""

    def test_immutable(self):
        m = TTAMetadata(is_flipped=False, transform_type="original")
        with pytest.raises(AttributeError):
            m.is_flipped = True  # type: ignore[misc]

    def test_equality(self):
        a = TTAMetadata(is_flipped=True, transform_type="flip")
        b = TTAMetadata(is_flipped=True, transform_type="flip")
        assert a == b

    def test_fields(self):
        m = TTAMetadata(is_flipped=True, transform_type="flip+noise",
                        photometric_params={"sigma": 10.0})
        assert m.is_flipped is True
        assert m.transform_type == "flip+noise"
        assert m.photometric_params == {"sigma": 10.0}


# =====================================================================
# 8. Edge cases
# =====================================================================

class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_single_pixel_image(self):
        """1×1×3 image should not crash."""
        img = np.array([[[128, 64, 32]]], dtype=np.uint8)
        engine = TTAEngine(_make_config(**{"flip.enabled": True}))
        imgs, metas = engine.prepare_batch(img)
        assert len(imgs) == 2
        for i in imgs:
            assert i.shape == (1, 1, 3)

    def test_blur_even_kernel_corrected(self):
        """Even kernel size in config should be bumped to odd."""
        cfg = _make_config(**{
            "photometric.blur.enabled": True,
            "photometric.blur.kernel_size": 4,
        })
        engine = TTAEngine(cfg)
        assert engine._blur_ksize == 5  # corrected to 5

    def test_empty_photometric_config(self, dummy_image: np.ndarray):
        """Config with empty photometric section works fine."""
        cfg = {"flip": {"enabled": True}, "photometric": {}}
        engine = TTAEngine(cfg)
        imgs, _ = engine.prepare_batch(dummy_image)
        assert len(imgs) == 2  # original + flip

    def test_missing_photometric_key(self, dummy_image: np.ndarray):
        """Config with no photometric key at all works fine."""
        cfg = {"flip": {"enabled": False}}
        engine = TTAEngine(cfg)
        imgs, _ = engine.prepare_batch(dummy_image)
        assert len(imgs) == 1

    def test_large_batch_performance(self):
        """All augmentations enabled produces correct count quickly."""
        img = np.random.default_rng(0).integers(
            0, 255, (256, 192, 3), dtype=np.uint8
        )
        cfg = _make_config(**{
            "flip.enabled": True,
            "photometric.brightness.enabled": True,
            "photometric.contrast.enabled": True,
            "photometric.noise.enabled": True,
            "photometric.blur.enabled": True,
        })
        engine = TTAEngine(cfg)
        imgs, metas = engine.prepare_batch(img)
        assert len(imgs) == 10
        assert all(i.dtype == np.uint8 for i in imgs)

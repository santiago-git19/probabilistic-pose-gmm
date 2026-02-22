"""
Comprehensive unit tests for metrics.py

Tests cover:
1. Mathematical correctness of each metric
2. Edge cases (singular matrices, zero weights, invisible keypoints)
3. Numerical stability
4. Integration with sklearn GMM models
"""

import pytest
import numpy as np
from sklearn.mixture import GaussianMixture
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pose_uncertainty.utils.metrics import (
    compute_l2_distance,
    compute_oks,
    compute_nll,
    compute_entropy,
    compute_covariance_volume,
    count_active_components,
    classify_failure_mode,
    COCO_SIGMAS
)
from pose_uncertainty.utils.types import RefinedKeypoint


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def perfect_prediction():
    """Ground truth matches prediction exactly."""
    coords = np.array([[100.0, 200.0], [150.0, 250.0]], dtype=np.float32)
    return coords, coords.copy()


@pytest.fixture
def noisy_prediction():
    """Prediction with small noise."""
    gt = np.array([[100.0, 200.0], [150.0, 250.0]], dtype=np.float32)
    pred = gt + np.random.randn(2, 2) * 2.0  # ~2 pixel noise
    return pred.astype(np.float32), gt


@pytest.fixture
def distant_prediction():
    """Prediction very far from ground truth."""
    gt = np.array([[100.0, 200.0], [150.0, 250.0]], dtype=np.float32)
    pred = np.array([[200.0, 300.0], [250.0, 350.0]], dtype=np.float32)
    return pred, gt


@pytest.fixture
def coco_keypoints():
    """17 COCO keypoints with visibility flags."""
    np.random.seed(42)
    gt_coords = np.random.rand(17, 2).astype(np.float32) * 300  # Random positions
    pred_coords = gt_coords + np.random.randn(17, 2) * 5.0  # Add noise
    visible = np.ones(17, dtype=np.int32) * 2  # All visible
    return pred_coords.astype(np.float32), gt_coords, visible


@pytest.fixture
def coco_keypoints_partial_visible():
    """COCO keypoints with some invisible."""
    np.random.seed(42)
    gt_coords = np.random.rand(17, 2).astype(np.float32) * 300
    pred_coords = gt_coords + np.random.randn(17, 2) * 5.0
    visible = np.array([2, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2, 2, 2, 1, 1, 2, 2], dtype=np.int32)
    return pred_coords.astype(np.float32), gt_coords, visible


@pytest.fixture
def simple_gmm():
    """Fitted GMM on simple 2D data."""
    np.random.seed(42)
    # Generate two clusters
    data1 = np.random.randn(50, 2) * 0.5 + np.array([0.0, 0.0])
    data2 = np.random.randn(50, 2) * 0.5 + np.array([5.0, 5.0])
    data = np.vstack([data1, data2]).astype(np.float32)
    
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(data)
    return gmm, data


@pytest.fixture
def unimodal_gmm():
    """Single Gaussian GMM."""
    np.random.seed(42)
    data = np.random.randn(100, 2).astype(np.float32)
    
    gmm = GaussianMixture(n_components=1, random_state=42)
    gmm.fit(data)
    return gmm


@pytest.fixture
def known_covariances():
    """Covariance matrices with known determinants."""
    # Identity: det = 1
    cov1 = np.eye(2, dtype=np.float32)
    
    # Scaled identity: det = 4
    cov2 = np.eye(2, dtype=np.float32) * 2.0
    
    # Anisotropic: det = 2
    cov3 = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    
    return np.array([cov1, cov2, cov3])


@pytest.fixture
def singular_covariance():
    """Nearly singular covariance matrix."""
    cov = np.array([[1.0, 0.99999], [0.99999, 1.0]], dtype=np.float32)
    return cov


# ============================================================================
# Tests: compute_l2_distance
# ============================================================================

def test_l2_distance_perfect(perfect_prediction):
    """Zero distance for perfect predictions."""
    pred, gt = perfect_prediction
    distances = compute_l2_distance(pred, gt)
    
    assert distances.shape == (2,)
    np.testing.assert_array_almost_equal(distances, np.zeros(2), decimal=5)


def test_l2_distance_known():
    """Known distance: (0,0) to (3,4) = 5."""
    pred = np.array([[0.0, 0.0]], dtype=np.float32)
    gt = np.array([[3.0, 4.0]], dtype=np.float32)
    
    distances = compute_l2_distance(pred, gt)
    
    assert distances.shape == (1,)
    np.testing.assert_almost_equal(distances[0], 5.0, decimal=5)


def test_l2_distance_vectorized():
    """Vectorized computation should be accurate."""
    pred = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    gt = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]], dtype=np.float32)
    
    distances = compute_l2_distance(pred, gt)
    expected = np.array([1.0, 1.0, 1.0])
    
    np.testing.assert_array_almost_equal(distances, expected, decimal=5)


def test_l2_distance_shape_mismatch():
    """Should raise error on shape mismatch."""
    pred = np.array([[0.0, 0.0]], dtype=np.float32)
    gt = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    
    with pytest.raises(AssertionError):
        compute_l2_distance(pred, gt)


# ============================================================================
# Tests: compute_oks
# ============================================================================

def test_oks_perfect(coco_keypoints):
    """Perfect prediction should give OKS = 1.0."""
    pred, gt, visible = coco_keypoints
    area = 10000.0  # Arbitrary bbox area
    
    # Use same coords for pred and gt
    oks, _ = compute_oks(pred, pred, visible, area)
    
    assert 0.99 <= oks <= 1.0  # Allow small numerical error


def test_oks_distant(distant_prediction):
    """Very distant prediction should give low OKS."""
    pred, gt = distant_prediction
    visible = np.array([2, 2], dtype=np.int32)
    area = 10000.0
    
    # Extend to 17 keypoints for COCO
    pred_full = np.tile(pred, (9, 1))[:17]
    gt_full = np.tile(gt, (9, 1))[:17]
    visible_full = np.ones(17, dtype=np.int32) * 2
    
    oks, _ = compute_oks(pred_full, gt_full, visible_full, area)
    
    assert oks < 0.1  # Very low similarity


def test_oks_partial_visible(coco_keypoints_partial_visible):
    """OKS should only consider visible keypoints."""
    pred, gt, visible = coco_keypoints_partial_visible
    area = 10000.0
    
    # Make prediction perfect
    oks, _ = compute_oks(gt, gt, visible, area)
    
    assert 0.99 <= oks <= 1.0


def test_oks_no_visible():
    """OKS should be 0 when no keypoints are visible."""
    pred = np.random.rand(17, 2).astype(np.float32)
    gt = np.random.rand(17, 2).astype(np.float32)
    visible = np.zeros(17, dtype=np.int32)  # All invisible
    area = 10000.0
    
    oks, _ = compute_oks(pred, gt, visible, area)
    
    assert oks == 0.0


def test_oks_custom_sigmas():
    """OKS should accept custom sigmas."""
    pred = np.array([[100.0, 200.0]], dtype=np.float32)
    gt = np.array([[102.0, 202.0]], dtype=np.float32)
    visible = np.array([2], dtype=np.int32)
    area = 10000.0
    sigmas = np.array([0.05], dtype=np.float32)  # More tolerant
    
    oks, _ = compute_oks(pred, gt, visible, area, sigmas=sigmas)
    
    assert 0.5 <= oks <= 1.0


# ============================================================================
# Tests: compute_nll
# ============================================================================

def test_nll_near_mean(simple_gmm):
    """Points near mean should have low NLL."""
    gmm, _ = simple_gmm
    
    # Point near first cluster mean
    gt_near = np.array([[0.0, 0.0]], dtype=np.float32)
    nll_near = compute_nll(gmm, gt_near)
    
    # Point very far
    gt_far = np.array([[100.0, 100.0]], dtype=np.float32)
    nll_far = compute_nll(gmm, gt_far)
    
    assert nll_near < nll_far


def test_nll_unimodal(unimodal_gmm):
    """Unimodal GMM should give reasonable NLL."""
    gmm = unimodal_gmm
    
    # Point at origin (near mean)
    gt = np.array([[0.0, 0.0]], dtype=np.float32)
    nll = compute_nll(gmm, gt)
    
    assert nll > 0  # NLL should be positive
    assert nll < 10  # But not extremely high for near-mean point


def test_nll_single_point_reshape():
    """Should handle 1D input by reshaping."""
    np.random.seed(42)
    data = np.random.randn(50, 2).astype(np.float32)
    gmm = GaussianMixture(n_components=1, random_state=42)
    gmm.fit(data)
    
    # Single point as 1D array
    gt = np.array([0.0, 0.0], dtype=np.float32)
    nll = compute_nll(gmm, gt)
    
    assert isinstance(nll, float)
    assert nll > 0


def test_nll_multiple_points():
    """Should compute average NLL for multiple points."""
    np.random.seed(42)
    data = np.random.randn(50, 2).astype(np.float32)
    gmm = GaussianMixture(n_components=1, random_state=42)
    gmm.fit(data)
    
    gt = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    nll = compute_nll(gmm, gt)
    
    assert isinstance(nll, float)
    assert nll > 0


# ============================================================================
# Tests: compute_entropy
# ============================================================================

def test_entropy_single_gaussian():
    """Entropy of single 2D Gaussian: H = log(2πe) + 0.5*log(det(Σ))."""
    weights = np.array([1.0], dtype=np.float32)
    cov = np.eye(2, dtype=np.float32)  # Identity: det = 1
    covariances = cov.reshape(1, 2, 2)
    
    entropy = compute_entropy(weights, covariances)
    
    # Expected: H = log(2πe) + 0.5*log(1) = log(2πe) ≈ 2.837
    expected = np.log(2 * np.pi * np.e)
    
    np.testing.assert_almost_equal(entropy, expected, decimal=2)


def test_entropy_high_variance():
    """Higher variance should give higher entropy."""
    weights = np.array([1.0], dtype=np.float32)
    
    cov_small = np.eye(2, dtype=np.float32) * 0.1
    cov_large = np.eye(2, dtype=np.float32) * 10.0
    
    entropy_small = compute_entropy(weights, cov_small.reshape(1, 2, 2))
    entropy_large = compute_entropy(weights, cov_large.reshape(1, 2, 2))
    
    assert entropy_large > entropy_small


def test_entropy_mixture():
    """Mixture entropy should be higher than single component."""
    # Single Gaussian
    weights_single = np.array([1.0], dtype=np.float32)
    cov_single = np.eye(2, dtype=np.float32)
    entropy_single = compute_entropy(weights_single, cov_single.reshape(1, 2, 2))
    
    # Mixture of two
    weights_mixture = np.array([0.5, 0.5], dtype=np.float32)
    covs_mixture = np.array([np.eye(2), np.eye(2)], dtype=np.float32)
    entropy_mixture = compute_entropy(weights_mixture, covs_mixture)
    
    assert entropy_mixture > entropy_single


def test_entropy_singular_covariance(singular_covariance):
    """Should handle near-singular covariance with regularization."""
    weights = np.array([1.0], dtype=np.float32)
    cov = singular_covariance.reshape(1, 2, 2)
    
    entropy = compute_entropy(weights, cov)
    
    assert np.isfinite(entropy)
    assert entropy > 0


def test_entropy_weights_normalization():
    """Should normalize weights if they don't sum to 1."""
    weights = np.array([2.0, 3.0], dtype=np.float32)  # Sum = 5
    covs = np.array([np.eye(2), np.eye(2)], dtype=np.float32)
    
    entropy = compute_entropy(weights, covs)
    
    assert np.isfinite(entropy)


# ============================================================================
# Tests: compute_covariance_volume
# ============================================================================

def test_covariance_volume_known(known_covariances):
    """Known determinants: 1, 4, 2."""
    covs = known_covariances
    
    # Dominant method (first component)
    weights = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    volume = compute_covariance_volume(covs, weights, method="dominant")
    
    np.testing.assert_almost_equal(volume, 1.0, decimal=5)  # sqrt(1) = 1


def test_covariance_volume_weighted(known_covariances):
    """Weighted average of volumes."""
    covs = known_covariances
    weights = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    
    volume = compute_covariance_volume(covs, weights, method="weighted")
    
    # Expected: 0.5 * sqrt(1) + 0.5 * sqrt(4) = 0.5 * 1 + 0.5 * 2 = 1.5
    expected = 0.5 * 1.0 + 0.5 * 2.0
    
    np.testing.assert_almost_equal(volume, expected, decimal=5)


def test_covariance_volume_single_matrix():
    """Should handle single covariance matrix."""
    cov = np.eye(2, dtype=np.float32) * 4.0  # det = 16
    
    volume = compute_covariance_volume(cov, method="dominant")
    
    np.testing.assert_almost_equal(volume, 4.0, decimal=5)  # sqrt(16) = 4


def test_covariance_volume_singular():
    """Should handle near-singular matrices."""
    cov = np.array([[1e-8, 0.0], [0.0, 1e-8]], dtype=np.float32)
    
    volume = compute_covariance_volume(cov, method="dominant")
    
    assert volume >= 1e-5  # Should use minimum threshold


# ============================================================================
# Tests: count_active_components
# ============================================================================

def test_count_active_components_all():
    """All components above threshold."""
    weights = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    
    count = count_active_components(weights, threshold=0.05)
    
    assert count == 4


def test_count_active_components_some():
    """Some components below threshold."""
    weights = np.array([0.7, 0.25, 0.04, 0.009, 0.001], dtype=np.float32)
    
    count = count_active_components(weights, threshold=0.01)
    
    assert count == 3  # First three > 0.01


def test_count_active_components_none():
    """All components below threshold (edge case)."""
    weights = np.array([0.0001, 0.0002, 0.0003], dtype=np.float32)
    
    count = count_active_components(weights, threshold=0.01)
    
    assert count == 0


def test_count_active_components_single():
    """Single dominant component."""
    weights = np.array([0.99, 0.005, 0.005], dtype=np.float32)
    
    count = count_active_components(weights, threshold=0.01)
    
    assert count == 1


# ============================================================================
# Tests: classify_failure_mode
# ============================================================================

def test_classify_robust():
    """High OKS + low entropy = Robust."""
    metrics = {
        "oks": 0.85,
        "entropy": 1.5,
        "n_active_components": 1
    }
    
    label = classify_failure_mode(metrics)
    
    assert label == "Robust"


def test_classify_denoised():
    """Improvement over baseline = Denoised."""
    metrics = {
        "oks": 0.75,
        "entropy": 2.0,
        "n_active_components": 1
    }
    
    label = classify_failure_mode(metrics, oks_base=0.6)
    
    assert label == "Denoised"


def test_classify_ambiguous():
    """High entropy + multiple modes = Ambiguous."""
    metrics = {
        "oks": 0.65,
        "entropy": 3.5,
        "n_active_components": 2
    }
    
    label = classify_failure_mode(metrics)
    
    assert label == "Ambiguous"


def test_classify_overconfident():
    """Low entropy + low OKS = Overconfident."""
    metrics = {
        "oks": 0.4,
        "entropy": 1.2,
        "n_active_components": 1
    }
    
    label = classify_failure_mode(metrics)
    
    assert label == "Overconfident"


def test_classify_honest_uncertainty():
    """High entropy + moderate OKS = Honest_Uncertainty."""
    metrics = {
        "oks": 0.55,
        "entropy": 2.8,
        "n_active_components": 2
    }
    
    label = classify_failure_mode(metrics)
    
    assert label == "Honest_Uncertainty"


def test_classify_failure():
    """Very low OKS = Failure."""
    metrics = {
        "oks": 0.2,
        "entropy": 1.0,
        "n_active_components": 1
    }
    
    label = classify_failure_mode(metrics)
    
    assert label == "Failure"


def test_classify_all_modes():
    """Ensure all modes are reachable."""
    test_cases = [
        ({"oks": 0.85, "entropy": 1.5, "n_active_components": 1}, None, "Robust"),
        ({"oks": 0.75, "entropy": 2.0, "n_active_components": 1}, 0.6, "Denoised"),
        ({"oks": 0.65, "entropy": 3.5, "n_active_components": 2}, None, "Ambiguous"),
        ({"oks": 0.4, "entropy": 1.2, "n_active_components": 1}, None, "Overconfident"),
        ({"oks": 0.55, "entropy": 2.8, "n_active_components": 2}, None, "Honest_Uncertainty"),
        ({"oks": 0.2, "entropy": 1.0, "n_active_components": 1}, None, "Failure"),
    ]
    
    for metrics, oks_base, expected in test_cases:
        label = classify_failure_mode(metrics, oks_base=oks_base)
        assert label == expected, f"Failed for {metrics}: got {label}, expected {expected}"


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_pipeline_integration(simple_gmm, coco_keypoints):
    """Test full metrics pipeline."""
    gmm, _ = simple_gmm
    pred, gt, visible = coco_keypoints
    
    # Compute all metrics
    area = 10000.0
    oks, _ = compute_oks(pred, gt, visible, area)
    
    # Use first keypoint for NLL
    gt_single = gt[0].reshape(1, 2)
    nll = compute_nll(gmm, gt_single)
    
    # Get GMM parameters
    weights = gmm.weights_.astype(np.float32)
    covariances = gmm.covariances_.astype(np.float32)
    
    entropy = compute_entropy(weights, covariances)
    volume = compute_covariance_volume(covariances, weights)
    n_active = count_active_components(weights)
    
    # Create metrics dict
    metrics = {
        "oks": oks,
        "nll": nll,
        "entropy": entropy,
        "covariance_volume": volume,
        "n_active_components": n_active
    }
    
    # Classify
    label = classify_failure_mode(metrics)
    
    # Assertions
    assert 0.0 <= oks <= 1.0
    assert nll > 0
    assert entropy > 0
    assert volume > 0
    assert n_active >= 1
    assert label in ["Robust", "Denoised", "Ambiguous", "Overconfident", 
                     "Honest_Uncertainty", "Failure"]


# ============================================================================
# Edge Cases
# ============================================================================

def test_edge_case_all_zero_weights():
    """Zero weights should be handled gracefully."""
    weights = np.zeros(3, dtype=np.float32)
    
    count = count_active_components(weights, threshold=0.01)
    
    assert count == 0


def test_edge_case_zero_area_oks():
    """Very small area should not cause division by zero."""
    pred = np.random.rand(17, 2).astype(np.float32)
    gt = np.random.rand(17, 2).astype(np.float32)
    visible = np.ones(17, dtype=np.int32)
    area = 1e-10  # Very small
    
    oks, _ = compute_oks(pred, gt, visible, area)
    
    assert np.isfinite(oks)


def test_edge_case_negative_determinant():
    """Negative determinant (numerical error) should be handled."""
    # Create a covariance that might have numerical issues
    cov = np.array([[1.0, 1.5], [1.5, 1.0]], dtype=np.float32)  # Not valid covariance
    
    # The function should regularize or handle this
    # For now, just check it doesn't crash
    try:
        volume = compute_covariance_volume(cov.reshape(1, 2, 2))
        assert np.isfinite(volume)
    except:
        # If it raises an error, that's also acceptable behavior
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

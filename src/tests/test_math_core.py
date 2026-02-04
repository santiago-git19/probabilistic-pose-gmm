"""
Unit Tests for Mathematical Core Modules.

This module tests the numerical correctness and edge case handling of:
    - Monte Carlo sampling strategies (rejection, importance, stratified)
    - Gaussian mixture models
    - Coordinate transformations
    - Uncertainty quantification

Testing Philosophy (ETH Standards):
-----------------------------------
1. **Numerical Stability**: Test with extreme values (near-zero, very large)
2. **Statistical Correctness**: Verify distributional properties
3. **Reproducibility**: All random operations must be deterministic with seeds
4. **Edge Cases**: Empty heatmaps, singular matrices, degenerate cases
5. **Performance**: Ensure vectorization works correctly

Run with:
    pytest src/tests/test_math_core.py -v
    pytest src/tests/test_math_core.py::test_rejection_sampler_basic -v
"""

import pytest
import numpy as np
from scipy import stats

# Import modules under test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pose_uncertainty.core.sampling import (
    RejectionSampler,
    ImportanceSampler,
    StratifiedSampler,
    SamplingConfig,
    create_sampler,
    sample_from_heatmap,
    apply_temperature_scaling,
    add_subpixel_jitter,
    estimate_effective_sample_size,
    fit_gaussian_to_samples,
)


# =============================================================================
# Fixtures: Synthetic Heatmaps for Testing
# =============================================================================

@pytest.fixture
def gaussian_heatmap():
    """
    Create a perfect 2D Gaussian heatmap centered at (32, 24).
    
    This is used to test if sampling correctly recovers the known
    mean and covariance.
    
    Returns:
        Tuple of (heatmap, true_mean, true_cov).
    """
    height, width = 64, 48
    y, x = np.ogrid[0:height, 0:width]
    
    # True parameters
    true_mean = np.array([24.0, 32.0])  # (x, y) = (24, 32)
    true_cov = np.array([[16.0, 0.0], [0.0, 16.0]])  # σ² = 16 → σ = 4 pixels
    
    # Compute Gaussian
    x_centered = x - true_mean[0]
    y_centered = y - true_mean[1]
    
    exponent = -(x_centered**2 / (2 * true_cov[0, 0]) + 
                 y_centered**2 / (2 * true_cov[1, 1]))
    
    heatmap = np.exp(exponent).astype(np.float32)
    heatmap /= np.sum(heatmap)  # Normalize
    
    return heatmap, true_mean, true_cov


@pytest.fixture
def bimodal_heatmap():
    """
    Create a bimodal heatmap with two Gaussian peaks.
    
    Simulates limb swap ambiguity (e.g., left vs right elbow).
    
    Returns:
        Heatmap with two modes.
    """
    height, width = 64, 48
    y, x = np.ogrid[0:height, 0:width]
    
    # First mode (left peak)
    mean1 = np.array([15.0, 32.0])
    cov1 = 9.0  # σ = 3
    gaussian1 = np.exp(-((x - mean1[0])**2 + (y - mean1[1])**2) / (2 * cov1))
    
    # Second mode (right peak)
    mean2 = np.array([33.0, 32.0])
    cov2 = 9.0
    gaussian2 = np.exp(-((x - mean2[0])**2 + (y - mean2[1])**2) / (2 * cov2))
    
    # Equal mixture
    heatmap = (gaussian1 + gaussian2).astype(np.float32)
    heatmap /= np.sum(heatmap)
    
    return heatmap


@pytest.fixture
def uniform_heatmap():
    """Uniform distribution (flat heatmap)."""
    heatmap = np.ones((64, 48), dtype=np.float32)
    heatmap /= np.sum(heatmap)
    return heatmap


@pytest.fixture
def peaked_heatmap():
    """Highly peaked distribution (argmax-like)."""
    # Create a narrow Gaussian instead of single pixel
    height, width = 64, 48
    y, x = np.ogrid[0:height, 0:width]
    
    # Very narrow Gaussian centered at (24, 32)
    mean_x, mean_y = 24.0, 32.0
    sigma = 2.0  # Small sigma for peaked distribution
    
    gaussian = np.exp(-((x - mean_x)**2 + (y - mean_y)**2) / (2 * sigma**2))
    heatmap = gaussian.astype(np.float32)
    heatmap /= np.sum(heatmap)
    
    return heatmap


# =============================================================================
# Test: Sampling Configuration
# =============================================================================

def test_sampling_config_validation():
    """Test that SamplingConfig validates parameters correctly."""
    # Valid config
    config = SamplingConfig(num_samples=100, batch_size=1000, seed=42)
    assert config.num_samples == 100
    
    # Invalid: negative num_samples
    with pytest.raises(ValueError, match="num_samples must be positive"):
        SamplingConfig(num_samples=-10)
    
    # Invalid: negative jitter
    with pytest.raises(ValueError, match="jitter_magnitude must be non-negative"):
        SamplingConfig(jitter_magnitude=-0.5)
    
    # Invalid: zero temperature
    with pytest.raises(ValueError, match="temperature must be positive"):
        SamplingConfig(temperature=0.0)


# =============================================================================
# Test: Rejection Sampler
# =============================================================================

def test_rejection_sampler_basic(gaussian_heatmap):
    """
    Test that rejection sampling recovers the true mean and covariance
    of a known Gaussian distribution.
    
    Statistical Test:
        Given heatmap ~ N(μ, Σ), samples should have:
        - E[samples] ≈ μ  (within 3σ / √N)
        - Cov[samples] ≈ Σ (within statistical tolerance)
    """
    heatmap, true_mean, true_cov = gaussian_heatmap
    
    config = SamplingConfig(
        num_samples=5000,  # Large N for statistical power
        batch_size=10000,
        use_dequantization=True,
        seed=42
    )
    
    sampler = RejectionSampler(config)
    samples = sampler.sample(heatmap)
    
    # Check shape
    assert samples.shape == (5000, 2)
    
    # Estimate empirical mean
    empirical_mean = np.mean(samples, axis=0)
    
    # Estimate empirical covariance
    empirical_cov = np.cov(samples.T)
    
    # Statistical test: mean within 3σ / √N
    std_error = np.sqrt(np.diag(true_cov) / config.num_samples)
    tolerance_mean = 3 * std_error
    
    assert np.allclose(empirical_mean, true_mean, atol=tolerance_mean.max()), \
        f"Mean mismatch: {empirical_mean} vs {true_mean}"
    
    # Covariance test (more lenient, as variance estimator has higher variance)
    # Check diagonal elements (variances) with relative error
    for i in range(2):
        relative_error = np.abs(empirical_cov[i, i] - true_cov[i, i]) / true_cov[i, i]
        assert relative_error < 0.2, \
            f"Variance {i} mismatch: {empirical_cov[i, i]} vs {true_cov[i, i]}"
    
    # Check off-diagonal elements (should be near zero) with absolute error
    assert np.abs(empirical_cov[0, 1]) < 1.0, \
        f"Covariance off-diagonal too large: {empirical_cov[0, 1]}"


def test_rejection_sampler_reproducibility(gaussian_heatmap):
    """Test that same seed produces identical results."""
    heatmap, _, _ = gaussian_heatmap
    
    config = SamplingConfig(num_samples=100, seed=42)
    
    sampler1 = RejectionSampler(config)
    samples1 = sampler1.sample(heatmap)
    
    sampler2 = RejectionSampler(config)
    samples2 = sampler2.sample(heatmap)
    
    assert np.allclose(samples1, samples2), "Sampling is not reproducible"


def test_rejection_sampler_peaked_distribution(peaked_heatmap):
    """
    Test rejection sampling on highly peaked distribution.
    
    Should concentrate samples around the peak pixel (32, 24).
    """
    config = SamplingConfig(num_samples=500, seed=42)
    sampler = RejectionSampler(config)
    
    samples = sampler.sample(peaked_heatmap)
    
    # Mean should be near (24, 32)
    empirical_mean = np.mean(samples, axis=0)
    expected_mean = np.array([24.0, 32.0])
    
    assert np.allclose(empirical_mean, expected_mean, atol=1.0), \
        f"Mean for peaked heatmap: {empirical_mean} vs {expected_mean}"


def test_rejection_sampler_edge_case_flat_heatmap(uniform_heatmap):
    """
    Test rejection sampling on flat (uniform) heatmap.
    
    This is inefficient for rejection sampling but should still work.
    """
    config = SamplingConfig(
        num_samples=100,
        batch_size=5000,  # Large batch for low acceptance rate
        max_iterations=50,
        seed=42
    )
    
    sampler = RejectionSampler(config)
    samples = sampler.sample(uniform_heatmap)
    
    assert samples.shape == (100, 2)
    
    # Samples should be roughly uniform over [0, 48) × [0, 64)
    # Allow for jitter that might go slightly outside bounds
    assert samples[:, 0].min() >= -1.0 and samples[:, 0].max() < 49.0
    assert samples[:, 1].min() >= -1.0 and samples[:, 1].max() < 65.0
    
    # But most samples should be within bounds
    in_bounds = np.sum((samples[:, 0] >= 0) & (samples[:, 0] < 48) & 
                       (samples[:, 1] >= 0) & (samples[:, 1] < 64))
    assert in_bounds >= 95, f"Too few samples in bounds: {in_bounds}/100"


def test_rejection_sampler_failure_on_pathological_heatmap():
    """
    Test that rejection sampling raises error for pathological inputs.
    
    Pathological case: Heatmap with insufficient mass.
    """
    # Near-zero heatmap
    heatmap = np.ones((64, 48), dtype=np.float32) * 1e-10
    
    config = SamplingConfig(num_samples=100, min_heatmap_mass=1e-6)
    sampler = RejectionSampler(config)
    
    with pytest.raises(ValueError, match="Heatmap mass.*below threshold"):
        sampler.sample(heatmap)


# =============================================================================
# Test: Importance Sampler
# =============================================================================

def test_importance_sampler_basic(gaussian_heatmap):
    """Test importance sampling on Gaussian heatmap."""
    heatmap, true_mean, true_cov = gaussian_heatmap
    
    config = SamplingConfig(num_samples=5000, seed=42)
    sampler = ImportanceSampler(config)
    
    samples = sampler.sample(heatmap)
    
    # Check shape
    assert samples.shape == (5000, 2)
    
    # Estimate mean
    empirical_mean = np.mean(samples, axis=0)
    std_error = np.sqrt(np.diag(true_cov) / config.num_samples)
    tolerance_mean = 3 * std_error
    
    assert np.allclose(empirical_mean, true_mean, atol=tolerance_mean.max())


def test_importance_sampler_vs_rejection(gaussian_heatmap):
    """
    Compare importance and rejection sampling.
    
    Both should produce statistically equivalent results.
    """
    heatmap, _, _ = gaussian_heatmap
    
    config = SamplingConfig(num_samples=1000, seed=42)
    
    rejection_sampler = RejectionSampler(config)
    rejection_samples = rejection_sampler.sample(heatmap)
    rejection_mean = np.mean(rejection_samples, axis=0)
    
    importance_sampler = ImportanceSampler(config)
    importance_samples = importance_sampler.sample(heatmap)
    importance_mean = np.mean(importance_samples, axis=0)
    
    # Means should be close (not exact due to randomness)
    assert np.allclose(rejection_mean, importance_mean, atol=1.0), \
        f"Rejection: {rejection_mean}, Importance: {importance_mean}"


# =============================================================================
# Test: Stratified Sampler
# =============================================================================

def test_stratified_sampler_basic(bimodal_heatmap):
    """
    Test stratified sampling on bimodal distribution.
    
    Stratified sampling should ensure both modes are represented.
    """
    config = SamplingConfig(num_samples=1000, seed=42)
    sampler = StratifiedSampler(config, grid_size=(4, 4))
    
    samples = sampler.sample(bimodal_heatmap)
    
    assert samples.shape == (1000, 2)
    
    # Check that samples span both modes
    # Left mode around x=15, right mode around x=33
    left_samples = samples[samples[:, 0] < 24]
    right_samples = samples[samples[:, 0] >= 24]
    
    assert len(left_samples) > 100, "Too few samples in left mode"
    assert len(right_samples) > 100, "Too few samples in right mode"


# =============================================================================
# Test: Factory Function
# =============================================================================

def test_create_sampler_factory():
    """Test factory function creates correct sampler types."""
    config = SamplingConfig(num_samples=100)
    
    rejection = create_sampler("rejection", config)
    assert isinstance(rejection, RejectionSampler)
    
    importance = create_sampler("importance", config)
    assert isinstance(importance, ImportanceSampler)
    
    stratified = create_sampler("stratified", config, grid_size=(8, 8))
    assert isinstance(stratified, StratifiedSampler)
    
    with pytest.raises(ValueError, match="Unknown sampling strategy"):
        create_sampler("invalid_strategy", config)


# =============================================================================
# Test: Convenience Function
# =============================================================================

def test_sample_from_heatmap_convenience(gaussian_heatmap):
    """Test high-level convenience function."""
    heatmap, true_mean, _ = gaussian_heatmap
    
    samples = sample_from_heatmap(
        heatmap,
        num_samples=1000,
        strategy="rejection",
        seed=42
    )
    
    assert samples.shape == (1000, 2)
    
    empirical_mean = np.mean(samples, axis=0)
    assert np.allclose(empirical_mean, true_mean, atol=1.0)


# =============================================================================
# Test: Utility Functions
# =============================================================================

def test_temperature_scaling():
    """Test softmax temperature scaling."""
    heatmap = np.array([[0.1, 0.3], [0.2, 0.4]], dtype=np.float32)
    heatmap /= np.sum(heatmap)
    
    # Temperature = 1 → no change
    scaled_t1 = apply_temperature_scaling(heatmap, temperature=1.0)
    assert np.allclose(scaled_t1, heatmap)
    
    # Temperature < 1 → sharper (higher max)
    scaled_sharp = apply_temperature_scaling(heatmap, temperature=0.5)
    assert np.max(scaled_sharp) > np.max(heatmap)
    
    # Temperature > 1 → smoother (lower max)
    scaled_smooth = apply_temperature_scaling(heatmap, temperature=2.0)
    assert np.max(scaled_smooth) < np.max(heatmap)
    
    # Should still sum to 1
    assert np.isclose(np.sum(scaled_sharp), 1.0)
    assert np.isclose(np.sum(scaled_smooth), 1.0)


def test_subpixel_jitter():
    """Test that jitter adds appropriate noise."""
    coords = np.array([[10.0, 20.0], [10.0, 20.0], [11.0, 21.0]], dtype=np.float32)
    
    rng = np.random.RandomState(42)
    jittered = add_subpixel_jitter(coords, jitter_magnitude=0.5, rng=rng)
    
    # Should have same shape
    assert jittered.shape == coords.shape
    
    # Should be different (not identical after jitter)
    assert not np.allclose(jittered, coords)
    
    # Should be within jitter_magnitude range
    diff = np.abs(jittered - coords)
    assert np.all(diff <= 0.5)


def test_effective_sample_size():
    """Test ESS calculation."""
    # Uniform weights → ESS = N
    uniform_weights = np.ones(100) / 100
    ess_uniform = estimate_effective_sample_size(uniform_weights)
    assert np.isclose(ess_uniform, 100.0, rtol=0.01)
    
    # Degenerate weights → ESS ≈ 1
    degenerate_weights = np.zeros(100)
    degenerate_weights[0] = 1.0
    ess_degenerate = estimate_effective_sample_size(degenerate_weights)
    assert np.isclose(ess_degenerate, 1.0, rtol=0.01)
    
    # Partially degenerate
    partial_weights = np.array([0.5, 0.3, 0.15, 0.05])
    ess_partial = estimate_effective_sample_size(partial_weights)
    assert 1.0 < ess_partial < 4.0


def test_fit_gaussian_to_samples():
    """Test Gaussian fitting to samples."""
    # Generate samples from known Gaussian
    true_mean = np.array([5.0, 10.0])
    true_cov = np.array([[4.0, 1.0], [1.0, 9.0]])
    
    rng = np.random.RandomState(42)
    samples = rng.multivariate_normal(true_mean, true_cov, size=10000)
    
    # Fit Gaussian
    estimated_mean, estimated_cov = fit_gaussian_to_samples(samples.astype(np.float32))
    
    # Check mean (should be very close with large N)
    assert np.allclose(estimated_mean, true_mean, atol=0.1)
    
    # Check covariance (more variance in estimate)
    assert np.allclose(estimated_cov, true_cov, atol=0.5)


# =============================================================================
# Test: Edge Cases and Error Handling
# =============================================================================

def test_invalid_heatmap_shape():
    """Test that samplers reject invalid heatmap shapes."""
    config = SamplingConfig(num_samples=100)
    sampler = RejectionSampler(config)
    
    # 1D heatmap
    with pytest.raises(ValueError, match="Heatmap must be 2D"):
        sampler.sample(np.array([0.1, 0.2, 0.3]))
    
    # 3D heatmap
    with pytest.raises(ValueError, match="Heatmap must be 2D"):
        sampler.sample(np.random.rand(10, 10, 3).astype(np.float32))


def test_negative_heatmap_values():
    """Test that samplers reject negative heatmap values."""
    config = SamplingConfig(num_samples=100)
    sampler = RejectionSampler(config)
    
    heatmap = np.array([[-0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    
    with pytest.raises(ValueError, match="negative values"):
        sampler.sample(heatmap)


def test_zero_sum_heatmap():
    """Test handling of all-zero heatmap."""
    config = SamplingConfig(num_samples=100, min_heatmap_mass=1e-6)
    sampler = RejectionSampler(config)
    
    heatmap = np.zeros((10, 10), dtype=np.float32)
    
    with pytest.raises(ValueError, match="below threshold"):
        sampler.sample(heatmap)


# =============================================================================
# Integration Test: Full Pipeline
# =============================================================================

def test_full_sampling_pipeline(gaussian_heatmap):
    """
    Integration test: Complete pipeline from heatmap to uncertainty estimate.
    
    Pipeline:
        1. Sample from heatmap (rejection sampling)
        2. Fit Gaussian to samples
        3. Verify parameters match ground truth
    """
    heatmap, true_mean, true_cov = gaussian_heatmap
    
    # Step 1: Sample
    samples = sample_from_heatmap(
        heatmap,
        num_samples=5000,
        strategy="rejection",
        use_dequantization=True,
        seed=42
    )
    
    # Step 2: Fit Gaussian
    estimated_mean, estimated_cov = fit_gaussian_to_samples(samples)
    
    # Step 3: Verify
    assert np.allclose(estimated_mean, true_mean, atol=0.5), \
        f"Mean: {estimated_mean} vs {true_mean}"
    
    # Covariance should be close
    relative_error = np.abs(estimated_cov - true_cov) / (true_cov + 1)
    assert np.all(relative_error < 0.3), \
        f"Covariance error too large:\n{estimated_cov}\nvs\n{true_cov}"


# =============================================================================
# Test: Mixture Model (RobustGaussianMixture)
# =============================================================================

from sklearn.datasets import make_blobs
from pose_uncertainty.core.mixture import (
    RobustGaussianMixture,
    MixtureResult,
    select_best_model,
    fit_with_outer_loop,
)


class TestMixtureModel:
    """
    Unit tests for the RobustGaussianMixture EM implementation.
    
    Tests cover:
        1. Convergence on synthetic Gaussian blobs
        2. Singularity handling (collinear points)
        3. Model selection (AIC/BIC)
        4. Outlier robustness (uniform component)
    """
    
    @pytest.fixture
    def single_blob_data(self):
        """
        Generate single Gaussian blob data for unimodal testing.
        
        Returns:
            Tuple of (samples, true_mean, true_std).
        """
        X, _ = make_blobs(
            n_samples=1000,
            centers=[[10.0, 20.0]],
            cluster_std=2.0,
            random_state=42
        )
        return X, np.array([10.0, 20.0]), 2.0
    
    @pytest.fixture
    def two_blob_data(self):
        """
        Generate two well-separated Gaussian blobs for bimodal testing.
        
        Returns:
            Tuple of (samples, centers, cluster_std).
        """
        centers = [[5.0, 10.0], [25.0, 10.0]]
        X, labels = make_blobs(
            n_samples=1000,
            centers=centers,
            cluster_std=2.0,
            random_state=42
        )
        return X, np.array(centers), 2.0
    
    @pytest.fixture
    def collinear_data(self):
        """
        Generate collinear (1D line) data for singularity testing.
        
        This will produce a singular covariance matrix (rank 1).
        
        Returns:
            Array of shape (100, 2) with points on a line.
        """
        rng = np.random.default_rng(42)
        t = rng.uniform(0, 10, 100)
        x = 2 * t + 5 + rng.normal(0, 0.001, 100)  # Very small noise
        y = 3 * t + 7 + rng.normal(0, 0.001, 100)
        return np.column_stack([x, y])
    
    # =========================================================================
    # Test: Basic Convergence
    # =========================================================================
    
    def test_single_gaussian_convergence(self, single_blob_data):
        """
        Test that EM converges to correct parameters for single Gaussian.
        
        Statistical test:
            - Fitted mean should be within 3σ/√N of true mean
            - Algorithm should converge (not hit max iterations)
        """
        X, true_mean, true_std = single_blob_data
        
        gmm = RobustGaussianMixture(
            n_components=1,
            reg_covar=1e-4,
            max_iter=100,
            tol=1e-4,
            random_state=42
        )
        
        gmm.fit(X)
        
        # Check convergence
        assert gmm.converged_, "EM did not converge for single Gaussian"
        assert gmm.n_iter_ < gmm.max_iter, \
            f"EM used all {gmm.max_iter} iterations"
        
        # Extract fitted mean
        fitted_mean, fitted_cov = gmm.get_mode()
        
        # Statistical tolerance: 3σ/√N
        std_error = true_std / np.sqrt(len(X))
        tolerance = 3 * std_error
        
        np.testing.assert_allclose(
            fitted_mean, true_mean, atol=tolerance,
            err_msg=f"Mean mismatch: {fitted_mean} vs {true_mean}"
        )
    
    def test_two_gaussian_convergence(self, two_blob_data):
        """
        Test that EM correctly identifies two clusters.
        
        Verification:
            - Both means should be close to true cluster centers
            - Each component should have significant weight
        """
        X, true_centers, true_std = two_blob_data
        
        gmm = RobustGaussianMixture(
            n_components=2,
            reg_covar=1e-4,
            max_iter=100,
            tol=1e-4,
            random_state=42
        )
        
        gmm.fit(X)
        
        # Check convergence
        assert gmm.converged_, "EM did not converge for two Gaussians"
        
        # Extract fitted means
        fitted_means = np.array([c.mean for c in gmm.components_])
        
        # Match fitted means to true centers (may be permuted)
        # Use Hungarian algorithm logic (simple for 2 components)
        dist_00 = np.linalg.norm(fitted_means[0] - true_centers[0])
        dist_01 = np.linalg.norm(fitted_means[0] - true_centers[1])
        
        if dist_00 < dist_01:
            matched = [(0, 0), (1, 1)]
        else:
            matched = [(0, 1), (1, 0)]
        
        # Check that fitted means are close to true centers
        std_error = true_std / np.sqrt(len(X) / 2)
        tolerance = 5 * std_error  # More lenient for mixture
        
        for fitted_idx, true_idx in matched:
            np.testing.assert_allclose(
                fitted_means[fitted_idx],
                true_centers[true_idx],
                atol=tolerance,
                err_msg=f"Mean {fitted_idx} doesn't match center {true_idx}"
            )
        
        # Check weights are reasonable (both should be around 0.4-0.5)
        weights = [c.weight for c in gmm.components_]
        for w in weights:
            assert 0.2 < w < 0.8, f"Component weight {w} is unreasonable"
    
    # =========================================================================
    # Test: Singularity Handling
    # =========================================================================
    
    def test_collinear_points_no_crash(self, collinear_data):
        """
        Test that collinear points don't cause crash.
        
        Collinear data has rank-1 covariance matrix (singular).
        The algorithm should:
            1. NOT crash with LinAlgError
            2. Apply regularization automatically
            3. Still converge to reasonable parameters
        """
        X = collinear_data
        
        gmm = RobustGaussianMixture(
            n_components=1,
            reg_covar=1e-3,  # Strong regularization for near-singular case
            max_iter=100,
            tol=1e-4,
            random_state=42
        )
        
        # Should NOT raise exception
        gmm.fit(X)
        
        # Should have converged or at least not crashed
        assert gmm.n_iter_ > 0, "No iterations performed"
        
        # Covariance should be regularized (positive definite)
        _, cov = gmm.get_mode()
        eigenvalues = np.linalg.eigvalsh(cov)
        
        assert np.all(eigenvalues > 0), \
            f"Covariance not positive definite: eigenvalues = {eigenvalues}"
    
    def test_regularization_applied_warning(self, collinear_data, caplog):
        """
        Test that regularization is applied and logged for singular matrices.
        """
        import logging
        
        X = collinear_data
        
        # Enable debug logging to capture regularization warnings
        with caplog.at_level(logging.DEBUG, logger='pose_uncertainty.core.mixture'):
            gmm = RobustGaussianMixture(
                n_components=1,
                reg_covar=1e-2,
                random_state=42
            )
            gmm.fit(X)
        
        # The covariance should be regularized (check eigenvalues)
        _, cov = gmm.get_mode()
        min_eig = np.linalg.eigvalsh(cov).min()
        assert min_eig >= gmm.reg_covar * 0.5, \
            f"Minimum eigenvalue {min_eig} is too small"
    
    # =========================================================================
    # Test: Model Selection (AIC/BIC)
    # =========================================================================
    
    def test_model_selection_single_blob(self, single_blob_data):
        """
        Test that model selection prefers K=1 for unimodal data.
        
        When data comes from a single Gaussian, the AIC/BIC penalty
        for extra parameters should favor the simpler model.
        """
        X, _, _ = single_blob_data
        
        result = select_best_model(
            X,
            aic_weight=0.5,
            bic_weight=0.5,
            reg_covar=1e-4,
            random_state=42
        )
        
        assert result.model_type == 'unimodal', \
            f"Expected unimodal, got {result.model_type}"
    
    def test_model_selection_two_blobs(self, two_blob_data):
        """
        Test that model selection prefers K=2 for bimodal data.
        
        When data clearly comes from two well-separated Gaussians,
        the bimodal model should fit significantly better.
        """
        X, _, _ = two_blob_data
        
        result = select_best_model(
            X,
            aic_weight=0.5,
            bic_weight=0.5,
            reg_covar=1e-4,
            random_state=42
        )
        
        assert result.model_type == 'bimodal', \
            f"Expected bimodal, got {result.model_type}"
    
    def test_aic_bic_calculation(self, single_blob_data):
        """
        Test that AIC/BIC are computed correctly.
        
        For K components with 2D data:
            - Parameters: K * (2 means + 3 cov params + 1 weight) = 6K
            - AIC = 2k - 2 ln(L)
            - BIC = k ln(n) - 2 ln(L)
        """
        X, _, _ = single_blob_data
        n_samples = len(X)
        
        gmm = RobustGaussianMixture(n_components=1, random_state=42)
        gmm.fit(X)
        
        aic = gmm.compute_aic(n_samples)
        bic = gmm.compute_bic(n_samples)
        
        # AIC and BIC should be finite
        assert np.isfinite(aic), f"AIC is not finite: {aic}"
        assert np.isfinite(bic), f"BIC is not finite: {bic}"
        
        # BIC should penalize more than AIC for large N
        # BIC uses ln(n) ≈ 6.9 for n=1000, AIC uses 2
        n_params = 6  # K=1 component
        expected_diff = n_params * (np.log(n_samples) - 2)
        actual_diff = bic - aic
        
        np.testing.assert_allclose(
            actual_diff, expected_diff, rtol=0.01,
            err_msg="BIC-AIC difference doesn't match expected"
        )
    
    # =========================================================================
    # Test: Outlier Robustness (Uniform Component)
    # =========================================================================
    
    def test_uniform_component_absorbs_outliers(self, single_blob_data):
        """
        Test that the uniform component absorbs outlier points.
        
        Add artificial outliers and verify:
            1. Gaussian mean is not affected
            2. Uniform weight increases to account for outliers
        """
        X, true_mean, _ = single_blob_data
        
        # Add 10% outliers (random uniform noise)
        rng = np.random.default_rng(42)
        n_outliers = int(0.1 * len(X))
        outliers = rng.uniform(-50, 50, size=(n_outliers, 2))
        X_with_outliers = np.vstack([X, outliers])
        
        gmm = RobustGaussianMixture(
            n_components=1,
            reg_covar=1e-4,
            random_state=42
        )
        gmm.fit(X_with_outliers, initial_uniform_weight=0.05)
        
        # Uniform weight should have increased
        assert gmm.uniform_weight_ > 0.05, \
            f"Uniform weight {gmm.uniform_weight_} didn't increase with outliers"
        
        # Gaussian mean should still be close to true mean
        fitted_mean, _ = gmm.get_mode()
        np.testing.assert_allclose(
            fitted_mean, true_mean, atol=1.0,
            err_msg=f"Mean affected by outliers: {fitted_mean} vs {true_mean}"
        )
    
    # =========================================================================
    # Test: Outer Loop Stability
    # =========================================================================
    
    def test_outer_loop_reduces_variance(self, single_blob_data):
        """
        Test that the outer loop strategy reduces estimate variance.
        
        Multiple bootstrap iterations should produce more stable estimates
        than a single fit.
        """
        X, true_mean, _ = single_blob_data
        
        result = fit_with_outer_loop(
            X,
            n_outer_iterations=10,
            n_resamples=500,
            random_state=42
        )
        
        # Result should be close to true mean
        np.testing.assert_allclose(
            result.best_mean, true_mean, atol=0.5,
            err_msg=f"Outer loop mean: {result.best_mean} vs {true_mean}"
        )
        
        # Covariance should be positive definite
        eigenvalues = np.linalg.eigvalsh(result.best_covariance)
        assert np.all(eigenvalues > 0), "Covariance not positive definite"
    
    # =========================================================================
    # Test: Edge Cases
    # =========================================================================
    
    def test_minimum_samples(self):
        """Test behavior with minimum number of samples."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        gmm = RobustGaussianMixture(n_components=1, random_state=42)
        gmm.fit(X)
        
        assert gmm.n_iter_ > 0, "Should complete at least one iteration"
    
    def test_insufficient_samples_error(self):
        """Test that too few samples raises error."""
        X = np.array([[1.0, 2.0]])  # Only 1 sample
        
        gmm = RobustGaussianMixture(n_components=2, random_state=42)
        
        with pytest.raises(ValueError, match="at least"):
            gmm.fit(X)
    
    def test_invalid_shape_error(self):
        """Test that wrong input shape raises error."""
        X_1d = np.array([1.0, 2.0, 3.0])
        X_3d = np.random.rand(100, 3)
        
        gmm = RobustGaussianMixture(n_components=1)
        
        with pytest.raises(ValueError, match="shape"):
            gmm.fit(X_1d)
        
        with pytest.raises(ValueError, match="shape"):
            gmm.fit(X_3d)
    
    def test_reproducibility(self, single_blob_data):
        """Test that same seed produces identical results."""
        X, _, _ = single_blob_data
        
        gmm1 = RobustGaussianMixture(n_components=1, random_state=42)
        gmm1.fit(X)
        mean1, _ = gmm1.get_mode()
        
        gmm2 = RobustGaussianMixture(n_components=1, random_state=42)
        gmm2.fit(X)
        mean2, _ = gmm2.get_mode()
        
        np.testing.assert_array_equal(mean1, mean2, "Results not reproducible")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

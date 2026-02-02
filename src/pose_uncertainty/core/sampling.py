"""
Monte Carlo Sampling Strategies from Heatmap Distributions.

This module provides different methods for drawing samples from pose estimation
heatmaps to construct empirical distributions for uncertainty quantification.

Why Monte Carlo Sampling?
-------------------------
Pose heatmaps represent probability distributions P(x, y | image) over spatial
locations. To estimate statistics (mean, covariance) robustly, we:

1. Draw N samples {(xᵢ, yᵢ)}ᵢ₌₁ᴺ from P
2. Fit statistical model (e.g., Gaussian Mixture) to samples
3. Extract refined estimate and uncertainty

Advantages Over Direct Heatmap Moments:
    - Robust to multi-modal distributions
    - Natural handling of discrete → continuous transition
    - Enables outlier detection via mixture fitting

Sampling Strategies:
-------------------
1. **Rejection Sampling** (Von Neumann): Unbiased, efficient for peaked distributions
2. **Importance Sampling**: Draw from P(x,y) directly (CDF inversion)
3. **Uniform + Reweighting**: Sample uniformly, weight by heatmap values
4. **Stratified Sampling**: Divide heatmap into regions, sample proportionally

References:
    - "Distribution-Aware Coordinate Representation of Keypoints" (DARK)
    - Rubinstein, R. "Simulation and the Monte Carlo Method"
    - von Neumann, J. "Various Techniques Used in Connection With Random Digits" (1951)
"""

from typing import Tuple, Optional, Literal, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


# Type aliases
SamplingStrategy = Literal["rejection", "importance", "uniform", "stratified"]


@dataclass
class SamplingConfig:
    """Configuration for Monte Carlo sampling from heatmaps."""
    
    num_samples: int = 500
    """Number of samples to draw per keypoint."""
    
    batch_size: int = 5000
    """Batch size for vectorized rejection sampling."""
    
    max_iterations: int = 100
    """Maximum rejection sampling iterations (safety mechanism)."""
    
    use_dequantization: bool = True
    """Add uniform jitter U[-0.5, 0.5] to prevent singular covariance."""
    
    jitter_magnitude: float = 0.5
    """Standard deviation of sub-pixel jitter (in pixels)."""
    
    temperature: float = 1.0
    """Softmax temperature for heatmap modulation."""
    
    seed: Optional[int] = None
    """Random seed for reproducibility."""
    
    min_heatmap_mass: float = 1e-6
    """Minimum sum(heatmap) to consider valid."""
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.jitter_magnitude < 0:
            raise ValueError(f"jitter_magnitude must be non-negative, got {self.jitter_magnitude}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")


class HeatmapSampler(ABC):
    """
    Abstract base class for heatmap sampling strategies.
    
    Design Pattern: Strategy Pattern
    --------------------------------
    This allows swapping sampling algorithms without modifying client code.
    
    Subclasses must implement:
        - _sample_internal: Core sampling logic
    """
    
    def __init__(self, config: SamplingConfig):
        """
        Initialize sampler with configuration.
        
        Args:
            config: Sampling configuration parameters.
        """
        self.config = config
        self.rng = np.random.RandomState(config.seed)
    
    @abstractmethod
    def _sample_internal(
        self,
        heatmap: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Internal sampling implementation (subclass-specific).
        
        Args:
            heatmap: Normalized 2D heatmap (height, width) with sum = 1.
        
        Returns:
            Samples array of shape (num_samples, 2) with (x, y) coordinates.
        """
        pass
    
    def sample(
        self,
        heatmap: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Public interface: Sample from heatmap with validation and preprocessing.
        
        Pipeline:
        --------
        1. Validate heatmap (non-negative, 2D, sufficient mass)
        2. Normalize to probability distribution
        3. Apply temperature scaling (optional)
        4. Call subclass-specific sampling method
        5. Apply dequantization jitter (optional)
        
        Args:
            heatmap: 2D array (height, width) with non-negative values.
        
        Returns:
            Array (num_samples, 2) with sub-pixel (x, y) coordinates.
        
        Raises:
            ValueError: If heatmap is invalid (negative values, insufficient mass).
        """
        # Validate heatmap shape
        if heatmap.ndim != 2:
            raise ValueError(f"Heatmap must be 2D, got shape {heatmap.shape}")
        
        # Validate non-negativity
        if np.any(heatmap < 0):
            raise ValueError("Heatmap contains negative values")
        
        # Check minimum mass
        total_mass = np.sum(heatmap)
        if total_mass < self.config.min_heatmap_mass:
            raise ValueError(
                f"Heatmap mass {total_mass:.2e} below threshold "
                f"{self.config.min_heatmap_mass:.2e}"
            )
        
        # Normalize to probability distribution
        normalized_heatmap = heatmap / total_mass
        
        # Apply temperature scaling
        if self.config.temperature != 1.0:
            normalized_heatmap = apply_temperature_scaling(
                normalized_heatmap,
                self.config.temperature
            )
        
        # Core sampling (subclass-specific)
        samples = self._sample_internal(normalized_heatmap)
        
        # Apply dequantization jitter
        if self.config.use_dequantization:
            samples = add_subpixel_jitter(
                samples,
                jitter_magnitude=self.config.jitter_magnitude,
                rng=self.rng
            )
        
        return samples
    
    def reset_seed(self, seed: Optional[int] = None):
        """Reset random number generator with new seed."""
        self.rng = np.random.RandomState(seed)


class RejectionSampler(HeatmapSampler):
    """
    Rejection Sampling (Von Neumann Accept-Reject Method).
    
    Algorithm:
    ---------
    Given normalized heatmap P(x, y):
    
    1. Generate candidate (x, y) ~ Uniform(image_domain)
    2. Generate threshold u ~ Uniform(0, 1)
    3. Accept if u < P(x, y), else reject and repeat
    
    Properties:
        - **Unbiased**: Samples exactly from P(x, y)
        - **Efficient**: For peaked distributions (high max(P))
        - **Vectorizable**: Process batches of candidates simultaneously
    
    Efficiency:
        Expected number of trials per accepted sample = 1 / max(P)
        
        For peaked heatmaps (max(P) ≈ 0.1-0.5), this requires 2-10 trials,
        which is acceptable. For flat heatmaps, consider importance sampling.
    
    Implementation Details:
        - Vectorized over batch_size candidates per iteration
        - Early termination when num_samples reached
        - Safety mechanism: max_iterations to prevent infinite loops
    
    References:
        von Neumann, J. "Various Techniques Used in Connection With Random Digits" (1951)
    """
    
    def _sample_internal(
        self,
        heatmap: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Vectorized rejection sampling implementation.
        
        Args:
            heatmap: Normalized 2D probability distribution (sum = 1).
        
        Returns:
            Samples array (num_samples, 2).
        
        Raises:
            RuntimeError: If max_iterations exceeded without collecting enough samples.
        """
        height, width = heatmap.shape
        samples = []
        total_accepted = 0
        iterations = 0
        
        # Find maximum probability (for efficiency diagnostics)
        max_prob = np.max(heatmap)
        
        while total_accepted < self.config.num_samples and iterations < self.config.max_iterations:
            # Generate batch of candidate coordinates
            # Shape: (batch_size, 2) with x ∈ [0, width), y ∈ [0, height)
            candidates_x = self.rng.randint(0, width, size=self.config.batch_size)
            candidates_y = self.rng.randint(0, height, size=self.config.batch_size)
            
            # Evaluate heatmap at candidate positions
            # P(x, y) for each candidate
            probs = heatmap[candidates_y, candidates_x]
            
            # Generate uniform random thresholds
            thresholds = self.rng.uniform(0, max_prob, size=self.config.batch_size)
            
            # Accept-reject decision (vectorized)
            accept_mask = thresholds < probs
            
            # Extract accepted samples
            accepted_x = candidates_x[accept_mask]
            accepted_y = candidates_y[accept_mask]
            
            # Stack as (x, y) pairs
            accepted_samples = np.column_stack([accepted_x, accepted_y]).astype(np.float32)
            
            if len(accepted_samples) > 0:
                samples.append(accepted_samples)
                total_accepted += len(accepted_samples)
            
            iterations += 1
        
        # Check if we collected enough samples
        if total_accepted < self.config.num_samples:
            raise RuntimeError(
                f"Rejection sampling failed: only collected {total_accepted}/{self.config.num_samples} "
                f"samples after {self.config.max_iterations} iterations. "
                f"Heatmap may be too flat (max_prob={max_prob:.4f}). "
                f"Consider using importance sampling instead."
            )
        
        # Concatenate all batches and trim to exact number
        all_samples = np.vstack(samples)
        return all_samples[:self.config.num_samples]


class ImportanceSampler(HeatmapSampler):
    """
    Importance sampling using inverse CDF method.
    
    Algorithm:
    ---------
    1. Flatten heatmap to 1D probability mass function (PMF)
    2. Use np.random.choice with probabilities ∝ heatmap
    3. Convert flat indices back to 2D (x, y) coordinates
    
    Properties:
        - **Fast**: O(N) where N = num_samples
        - **Exact**: Samples exactly from P(x, y)
        - **Memory-efficient**: Single pass through heatmap
    
    Advantages over Rejection Sampling:
        - No rejection → deterministic runtime
        - Works well for flat distributions
    
    Disadvantages:
        - Less flexible for custom distributions
        - Harder to extend to continuous spaces
    """
    
    def _sample_internal(
        self,
        heatmap: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Importance sampling implementation.
        
        Args:
            heatmap: Normalized 2D probability distribution.
        
        Returns:
            Samples array (num_samples, 2).
        """
        height, width = heatmap.shape
        
        # Flatten heatmap to 1D PMF
        pmf = heatmap.ravel()
        
        # Sample flat indices according to PMF
        flat_indices = self.rng.choice(
            a=len(pmf),
            size=self.config.num_samples,
            replace=True,
            p=pmf
        )
        
        # Convert flat indices to 2D coordinates
        # Note: numpy uses row-major (C) ordering
        y_coords = flat_indices // width
        x_coords = flat_indices % width
        
        # Stack as (x, y) pairs
        samples = np.column_stack([x_coords, y_coords]).astype(np.float32)
        
        return samples


class StratifiedSampler(HeatmapSampler):
    """
    Stratified sampling for variance reduction.
    
    Algorithm:
    ---------
    1. Divide heatmap into K×K grid cells
    2. Compute total probability mass in each cell
    3. Allocate samples proportionally to cell mass
    4. Sample uniformly within each cell
    
    Properties:
        - **Lower Variance**: Better than simple random sampling
        - **Coverage Guarantee**: All regions represented
        - **Handles Multi-modality**: Ensures all peaks sampled
    
    Use Cases:
        - Bimodal heatmaps (left/right limb ambiguity)
        - Ensuring coverage in low-probability regions
        - When importance sampling under-samples secondary modes
    """
    
    def __init__(self, config: SamplingConfig, grid_size: Tuple[int, int] = (8, 8)):
        """
        Initialize stratified sampler.
        
        Args:
            config: Sampling configuration.
            grid_size: (rows, cols) for grid subdivision.
        """
        super().__init__(config)
        self.grid_size = grid_size
    
    def _sample_internal(
        self,
        heatmap: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Stratified sampling implementation.
        
        Args:
            heatmap: Normalized 2D probability distribution.
        
        Returns:
            Samples array (num_samples, 2).
        """
        height, width = heatmap.shape
        grid_rows, grid_cols = self.grid_size
        
        # Compute cell dimensions
        cell_height = height / grid_rows
        cell_width = width / grid_cols
        
        # Compute mass in each cell
        cell_masses = np.zeros((grid_rows, grid_cols))
        for i in range(grid_rows):
            for j in range(grid_cols):
                y_start = int(i * cell_height)
                y_end = int((i + 1) * cell_height)
                x_start = int(j * cell_width)
                x_end = int((j + 1) * cell_width)
                cell_masses[i, j] = np.sum(heatmap[y_start:y_end, x_start:x_end])
        
        # Allocate samples proportionally to cell mass
        cell_probs = cell_masses.ravel() / np.sum(cell_masses)
        cell_samples = np.random.multinomial(self.config.num_samples, cell_probs)
        cell_samples = cell_samples.reshape((grid_rows, grid_cols))
        
        # Sample uniformly within each cell
        samples = []
        for i in range(grid_rows):
            for j in range(grid_cols):
                n_cell_samples = cell_samples[i, j]
                if n_cell_samples == 0:
                    continue
                
                # Cell boundaries
                y_start = i * cell_height
                y_end = (i + 1) * cell_height
                x_start = j * cell_width
                x_end = (j + 1) * cell_width
                
                # Sample uniformly within cell
                x_samples = self.rng.uniform(x_start, x_end, size=n_cell_samples)
                y_samples = self.rng.uniform(y_start, y_end, size=n_cell_samples)
                
                cell_samples_array = np.column_stack([x_samples, y_samples])
                samples.append(cell_samples_array)
        
        # Concatenate all cell samples
        all_samples = np.vstack(samples).astype(np.float32)
        
        return all_samples


# =============================================================================
# Factory Function
# =============================================================================

def create_sampler(
    strategy: SamplingStrategy,
    config: SamplingConfig,
    **kwargs
) -> HeatmapSampler:
    """
    Factory function to create sampler instances.
    
    Args:
        strategy: Sampling method name.
        config: Sampling configuration.
        **kwargs: Additional strategy-specific parameters.
    
    Returns:
        Configured HeatmapSampler instance.
    
    Raises:
        ValueError: If strategy is unknown.
    
    Example:
        >>> config = SamplingConfig(num_samples=1000, seed=42)
        >>> sampler = create_sampler("rejection", config)
        >>> samples = sampler.sample(heatmap)
    """
    if strategy == "rejection":
        return RejectionSampler(config)
    elif strategy == "importance":
        return ImportanceSampler(config)
    elif strategy == "stratified":
        grid_size = kwargs.get("grid_size", (8, 8))
        return StratifiedSampler(config, grid_size=grid_size)
    else:
        raise ValueError(
            f"Unknown sampling strategy: {strategy}. "
            f"Must be one of: rejection, importance, stratified"
        )


# =============================================================================
# Utility Functions
# =============================================================================

def apply_temperature_scaling(
    heatmap: npt.NDArray[np.float32],
    temperature: float
) -> npt.NDArray[np.float32]:
    """
    Apply softmax temperature to modulate heatmap peakedness.
    
    Mathematical Form:
    -----------------
        H_T(x, y) = H(x, y)^(1/T) / Z
    
    Where Z = ΣₓΣᵧ H(x, y)^(1/T) is the normalization constant.
    
    This is equivalent to: softmax(log(H) / T)
    
    Args:
        heatmap: Input heatmap (non-negative, normalized).
        temperature: T > 0.
            - T → 0: Argmax (deterministic)
            - T = 1: No change
            - T → ∞: Uniform distribution
    
    Returns:
        Temperature-scaled heatmap (same shape, normalized).
    
    Numerical Stability:
        Uses log-space computation to prevent overflow:
        log(H_T) = (1/T) log(H) - log(Z)
    
    Example:
        >>> heatmap = np.array([[0.1, 0.3], [0.2, 0.4]])
        >>> sharp = apply_temperature_scaling(heatmap, temperature=0.5)  # Peaky
        >>> smooth = apply_temperature_scaling(heatmap, temperature=2.0)  # Flat
    """
    if temperature == 1.0:
        return heatmap
    
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    heatmap_safe = np.maximum(heatmap, epsilon)
    
    # Apply temperature in log space: log(H^(1/T)) = (1/T) * log(H)
    log_heatmap = np.log(heatmap_safe)
    log_heatmap_scaled = log_heatmap / temperature
    
    # Convert back from log space
    heatmap_scaled = np.exp(log_heatmap_scaled)
    
    # Normalize to sum to 1
    heatmap_normalized = heatmap_scaled / np.sum(heatmap_scaled)
    
    return heatmap_normalized


def add_subpixel_jitter(
    coordinates: npt.NDArray[np.float32],
    jitter_magnitude: float = 0.5,
    rng: Optional[np.random.RandomState] = None
) -> npt.NDArray[np.float32]:
    """
    Add uniform sub-pixel jitter to discrete coordinates.
    
    Why Jitter?
    ----------
    When sampling from discrete heatmaps, multiple samples may map to the
    same pixel. Adding jitter:
    1. Prevents degenerate covariance matrices (rank-deficient)
    2. Models uncertainty within the pixel (spatial discretization)
    3. Smooths the empirical distribution
    
    Mathematical Form:
        x_jittered = x_discrete + U[-magnitude, +magnitude]
    
    Args:
        coordinates: Array of shape (N, 2) with integer or float coordinates.
        jitter_magnitude: Magnitude of uniform jitter (default: 0.5 pixels).
        rng: NumPy RandomState for reproducibility.
    
    Returns:
        Jittered coordinates (N, 2).
    
    Typical Values:
        - 0.5: Uniform over pixel (recommended)
        - 0.25: More concentrated
        - 1.0: Blurry (may over-estimate uncertainty)
    
    Example:
        >>> coords = np.array([[10, 20], [10, 20], [11, 21]], dtype=np.float32)
        >>> jittered = add_subpixel_jitter(coords, jitter_magnitude=0.5)
        >>> # Now all coords are unique, preventing singular covariance
    """
    if rng is None:
        rng = np.random.RandomState()
    
    # Generate uniform noise in [-magnitude, +magnitude]
    noise = rng.uniform(
        -jitter_magnitude,
        +jitter_magnitude,
        size=coordinates.shape
    )
    
    jittered = coordinates + noise
    
    return jittered.astype(np.float32)


def estimate_effective_sample_size(
    weights: npt.NDArray[np.float32]
) -> float:
    """
    Compute effective sample size (ESS) for weighted samples.
    
    Definition:
    ----------
    For importance sampling with weights w_i, the ESS is:
    
        ESS = (Σᵢ wᵢ)² / Σᵢ wᵢ²
    
    This quantifies how many "independent" samples the weighted set represents.
    
    Args:
        weights: Array of sample weights (N,).
    
    Returns:
        ESS ∈ [1, N] where:
            - ESS = N: All weights equal (optimal)
            - ESS = 1: One weight dominates (degeneracy)
    
    Usage:
        If ESS < N/2, consider resampling to avoid weight collapse.
    
    Application in TTA:
        After averaging heatmaps from multiple augmentations, check if
        samples are diverse enough. Low ESS suggests over-confident heatmap.
    
    Example:
        >>> weights = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform
        >>> ess = estimate_effective_sample_size(weights)
        >>> # ess ≈ 4.0 (optimal)
        >>>
        >>> weights = np.array([0.9, 0.05, 0.03, 0.02])  # Degeneracy
        >>> ess = estimate_effective_sample_size(weights)
        >>> # ess ≈ 1.2 (poor)
    """
    weights_sum = np.sum(weights)
    weights_sq_sum = np.sum(weights ** 2)
    
    if weights_sq_sum == 0:
        return 0.0
    
    ess = (weights_sum ** 2) / weights_sq_sum
    
    return ess


def fit_gaussian_to_samples(
    samples: npt.NDArray[np.float32],
    weights: Optional[npt.NDArray[np.float32]] = None
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Fit a single Gaussian to weighted samples (maximum likelihood).
    
    For comparison with mixture model approach. Computes:
    
        μ = E[x] = Σᵢ wᵢ xᵢ / Σᵢ wᵢ
        Σ = E[(x - μ)(x - μ)ᵀ] = Σᵢ wᵢ (xᵢ - μ)(xᵢ - μ)ᵀ / Σᵢ wᵢ
    
    Args:
        samples: Array of shape (N, D).
        weights: Optional weights of shape (N,). If None, assumes equal weights.
    
    Returns:
        Tuple of (mean, covariance):
            - mean: Shape (D,)
            - covariance: Shape (D, D)
    
    When to Use:
        - Unimodal heatmaps (single clear peak)
        - Baseline comparison with mixture model
        - Fast approximation (no EM iterations)
    
    Example:
        >>> samples = np.random.randn(1000, 2)
        >>> mean, cov = fit_gaussian_to_samples(samples)
        >>> # mean ≈ [0, 0], cov ≈ [[1, 0], [0, 1]]
    """
    n_samples, n_dims = samples.shape
    
    if weights is None:
        weights = np.ones(n_samples) / n_samples
    else:
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
    
    # Weighted mean
    mean = np.sum(weights[:, np.newaxis] * samples, axis=0)
    
    # Weighted covariance
    centered = samples - mean
    weighted_centered = weights[:, np.newaxis] * centered
    covariance = centered.T @ weighted_centered
    
    return mean.astype(np.float32), covariance.astype(np.float32)


# =============================================================================
# Convenience Function for Direct Usage
# =============================================================================

def sample_from_heatmap(
    heatmap: npt.NDArray[np.float32],
    num_samples: int = 500,
    strategy: SamplingStrategy = "rejection",
    temperature: float = 1.0,
    use_dequantization: bool = True,
    seed: Optional[int] = None
) -> npt.NDArray[np.float32]:
    """
    Convenience function to sample from a heatmap with minimal configuration.
    
    This is a high-level interface that creates a sampler, configures it,
    and returns samples in one call. For more control, use create_sampler().
    
    Args:
        heatmap: 2D array (height, width) with non-negative values.
        num_samples: Number of samples to draw.
        strategy: Sampling method ("rejection", "importance", "stratified").
        temperature: Softmax temperature for heatmap modulation.
        use_dequantization: Whether to add sub-pixel jitter.
        seed: Random seed for reproducibility.
    
    Returns:
        Array (num_samples, 2) with (x, y) coordinates in pixel space.
        Coordinates are sub-pixel (floating-point).
    
    Example:
        >>> heatmap = model.predict(image)[0]  # Shape: (64, 48)
        >>> samples = sample_from_heatmap(heatmap, num_samples=1000)
        >>> mean = samples.mean(axis=0)  # Empirical mean
        >>> cov = np.cov(samples.T)      # Empirical covariance
    """
    config = SamplingConfig(
        num_samples=num_samples,
        temperature=temperature,
        use_dequantization=use_dequantization,
        seed=seed
    )
    
    sampler = create_sampler(strategy, config)
    samples = sampler.sample(heatmap)
    
    return samples

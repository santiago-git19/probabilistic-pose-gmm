"""
Restricted Gaussian Mixture Model for Bimodal Pose Distributions.

This module implements the core algorithm from the CVIU paper for handling
ambiguous pose estimates, particularly left/right limb swaps in symmetric poses.

Mathematical Foundation:
-----------------------
Standard Gaussian Mixture Models (GMMs) fit K components:

    P(x) = Σₖ πₖ 𝒩(x | μₖ, Σₖ)

However, for pose ambiguities (e.g., distinguishing left/right elbow when person
is facing away), we enforce a **shared covariance constraint**:

    Σ₁ = Σ₂ = Σ (tied covariance)

Why This Constraint?
-------------------
1. **Identifiability**: Without constraints, EM may converge to arbitrary local optima
2. **Physical Interpretation**: Left/right limbs have same localization uncertainty
3. **Regularization**: Reduces parameters from 2(d² + d) to d² + d + 1
4. **Computational Efficiency**: Faster covariance estimation

Algorithm: Online Stochastic EM
-------------------------------
Unlike batch EM, we use a streaming variant suitable for TTA pipelines:
1. Initialize with k-means++ on first batch
2. Update statistics incrementally: μ_new = (1-α)μ_old + α x_new
3. Detect convergence via log-likelihood plateau

References:
    - "Restricted Gaussian Mixture Models for Pose Estimation" (CVIU)
    - Bishop, C. "Pattern Recognition and Machine Learning" (Ch. 9)
"""

from typing import Tuple, Optional, List
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import multivariate_normal


@dataclass
class MixtureComponent:
    """
    Single component of a Gaussian Mixture Model.
    
    Attributes:
        mean: Component mean (μₖ) of shape (D,).
        covariance: Covariance matrix (Σₖ) of shape (D, D).
        weight: Mixing coefficient (πₖ) where Σₖ πₖ = 1.
        n_samples: Number of samples currently assigned to this component.
    """
    mean: npt.NDArray[np.float32]
    covariance: npt.NDArray[np.float32]
    weight: float
    n_samples: int = 0


class RestrictedGaussianMixture:
    """
    Gaussian Mixture Model with shared covariance constraint for bimodal pose detection.
    
    This class implements the core mathematical engine for detecting and resolving
    pose ambiguities. It's designed specifically for the case of K=2 components
    (binary ambiguity) with tied covariance matrices.
    
    Mathematical Model:
    ------------------
        P(x | θ) = π₁ 𝒩(x | μ₁, Σ) + π₂ 𝒩(x | μ₂, Σ)
    
    Where:
        - μ₁, μ₂: Component means (e.g., left vs. right elbow positions)
        - Σ: Shared covariance matrix (uncertainty ellipse)
        - π₁, π₂: Mixing weights with π₁ + π₂ = 1
    
    Parameters:
        n_components: Number of mixture components (default: 2 for binary ambiguity).
        shared_covariance: If True, enforces Σ₁ = Σ₂ (RECOMMENDED for pose).
        covariance_type: "full" (general 2×2), "diag" (diagonal), "spherical" (σ²I).
        reg_covar: Regularization added to diagonal for numerical stability (λI).
        max_iter: Maximum EM iterations.
        tol: Convergence threshold for log-likelihood change.
        random_state: Seed for reproducible initialization.
    
    Attributes:
        components_: List of fitted MixtureComponent objects after calling fit().
        converged_: Boolean flag indicating if EM converged.
        n_iter_: Number of iterations actually performed.
        log_likelihood_: Final log-likelihood of the data.
    
    Example:
        >>> # Fit mixture to Monte Carlo samples from a heatmap
        >>> samples = np.array([[100, 200], [102, 198], [150, 220], ...])
        >>> gmm = RestrictedGaussianMixture(n_components=2, shared_covariance=True)
        >>> gmm.fit_online(samples)
        >>> best_mode = gmm.get_mode()  # Returns (x, y) of dominant component
        >>> sigma = gmm.components_[0].covariance  # Uncertainty ellipse
    
    Implementation Notes:
        - Uses scipy's multivariate_normal for probability computations
        - Adds small regularization (1e-6) to prevent singular matrices
        - Detects mode-swapping via Euclidean distance tracking
    """
    
    def __init__(
        self,
        n_components: int = 2,
        shared_covariance: bool = True,
        covariance_type: str = "full",
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize the Restricted Gaussian Mixture Model.
        
        Args:
            n_components: Number of Gaussian components (typically 2 for pose).
            shared_covariance: Enforce Σ₁ = Σ₂ = ... = Σ (CVIU paper constraint).
            covariance_type: Covariance structure - "full", "tied", "diag", "spherical".
            reg_covar: Regularization λ added to covariance diagonal (Σ + λI).
            max_iter: Maximum EM iterations before forced termination.
            tol: Stop if |log_likelihood_new - log_likelihood_old| < tol.
            random_state: Random seed for k-means++ initialization.
        """
        ...
    
    def fit_online(
        self,
        samples: npt.NDArray[np.float32],
        learning_rate: float = 0.1
    ) -> "RestrictedGaussianMixture":
        """
        Fit the mixture model using online (streaming) EM updates.
        
        Online EM is suitable for TTA pipelines where samples arrive sequentially.
        Instead of batch E/M steps, we update statistics incrementally:
        
        E-step (online):
            γᵢₖ = πₖ 𝒩(xᵢ | μₖ, Σₖ) / Σⱼ πⱼ 𝒩(xᵢ | μⱼ, Σⱼ)
        
        M-step (online):
            μₖ ← (1 - α)μₖ + α (Σᵢ γᵢₖ xᵢ) / (Σᵢ γᵢₖ)
        
        Where α is the learning rate (controls forgetting of old estimates).
        
        Args:
            samples: Array of shape (N, D) with N samples in D dimensions.
            learning_rate: Exponential moving average coefficient α ∈ (0, 1].
                          Higher values = faster adaptation, more noise.
        
        Returns:
            self (for method chaining).
        
        Initialization:
            Uses k-means++ to seed component means, preventing poor local optima.
        
        Convergence:
            Monitors relative change in log-likelihood. Sets self.converged_ = True
            if change < self.tol for 3 consecutive iterations.
        """
        ...
    
    def get_mode(self) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Extract the dominant mode (component with highest mixing weight).
        
        For bimodal distributions representing left/right ambiguity, this returns
        the more probable interpretation based on:
        
            argmax_k πₖ
        
        Returns:
            Tuple of (mean, covariance) for the dominant component.
            - mean: Shape (D,) - most likely keypoint location
            - covariance: Shape (D, D) - uncertainty ellipse
        
        Usage in Pipeline:
            After fitting to Monte Carlo samples, extract the refined keypoint:
            ```python
            (x, y), sigma = gmm.get_mode()
            keypoint = RefinedKeypoint(x=x, y=y, sigma=sigma, ...)
            ```
        
        Edge Case:
            If weights are equal (π₁ = π₂), selects component with higher
            sample count (self.components_[k].n_samples).
        """
        ...
    
    def predict_proba(
        self,
        samples: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Compute posterior probabilities (responsibilities) for new samples.
        
        For each sample xᵢ, compute:
            γᵢₖ = P(k | xᵢ) = πₖ 𝒩(xᵢ | μₖ, Σₖ) / P(xᵢ)
        
        Args:
            samples: Array of shape (N, D).
        
        Returns:
            Array of shape (N, K) with posterior probabilities.
            Each row sums to 1: Σₖ γᵢₖ = 1.
        
        Application:
            Can be used to visualize ambiguity - if max_k γᵢₖ ≈ 0.5, the sample
            is ambiguous between modes.
        """
        ...
    
    def detect_outliers(
        self,
        samples: npt.NDArray[np.float32],
        threshold: float = 3.0
    ) -> npt.NDArray[np.bool_]:
        """
        Identify outlier samples using Mahalanobis distance.
        
        Mathematical Definition:
        -----------------------
        For each sample xᵢ, compute distance to nearest component:
        
            d_M(xᵢ, k) = √[(xᵢ - μₖ)ᵀ Σ⁻¹ (xᵢ - μₖ)]
        
        Mark as outlier if min_k d_M(xᵢ, k) > threshold.
        
        Args:
            samples: Array of shape (N, D).
            threshold: Number of standard deviations for outlier cutoff.
                      Common values: 2.5 (loose), 3.0 (moderate), 3.5 (strict).
        
        Returns:
            Boolean array of shape (N,) where True indicates outlier.
        
        Why Mahalanobis?
            Unlike Euclidean distance, accounts for correlation structure and
            varying scale along different dimensions (x vs. y uncertainty).
        
        Usage:
            Outliers in the Monte Carlo samples may indicate:
            1. Noisy heatmap predictions
            2. True multi-modal distributions (>2 modes)
            3. Systematic errors (e.g., incorrect augmentation inverse)
        """
        ...
    
    def _initialize_kmeans_plusplus(
        self,
        samples: npt.NDArray[np.float32]
    ) -> List[npt.NDArray[np.float32]]:
        """
        Initialize component means using k-means++ algorithm.
        
        K-means++ Algorithm:
            1. Choose first center uniformly at random from samples
            2. For remaining centers, choose with probability ∝ D(x)²
               where D(x) = distance to nearest existing center
            3. Repeat until K centers selected
        
        Why K-means++?
            - Provably O(log K)-competitive with optimal k-means
            - Avoids pathological initializations (all means ≈ same point)
            - Faster convergence than random initialization
        
        Returns:
            List of K initial mean vectors.
        """
        ...
    
    def _m_step_shared_covariance(
        self,
        samples: npt.NDArray[np.float32],
        responsibilities: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        M-step with shared covariance constraint.
        
        Mathematical Derivation:
        -----------------------
        Under the constraint Σ₁ = Σ₂ = Σ, the MLE is:
        
            Σ = (1/N) Σₖ Σᵢ γᵢₖ (xᵢ - μₖ)(xᵢ - μₖ)ᵀ
        
        This is a weighted average of per-component scatter matrices.
        
        Args:
            samples: Data matrix (N, D).
            responsibilities: Posterior probabilities (N, K).
        
        Returns:
            Shared covariance matrix (D, D).
        
        Regularization:
            Adds reg_covar * I to ensure positive definiteness.
        """
        ...

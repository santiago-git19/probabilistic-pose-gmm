"""
Robust Gaussian Mixture Model for Bimodal Pose Distributions.

This module implements the core EM algorithm from scratch for handling
ambiguous pose estimates, particularly left/right limb swaps in symmetric poses.

Mathematical Foundation:
-----------------------
Standard Gaussian Mixture Models (GMMs) fit K components:

    P(x) = Σₖ πₖ 𝒩(x | μₖ, Σₖ)

For robustness, we add a Uniform component to handle outliers:

    P(x) = Σₖ πₖ 𝒩(x | μₖ, Σₖ) + π_uniform * U(x | Area)

Why This Structure?
-------------------
1. **Outlier Robustness**: Uniform component absorbs noise points
2. **Physical Interpretation**: Heatmap samples may include artifacts
3. **Model Selection**: AIC/BIC to choose between unimodal and bimodal

Algorithm: Batch EM with Robustness
-----------------------------------
1. Initialize with k-means or random initialization
2. E-step: Compute responsibilities γₙₖ
3. M-step: Update parameters with regularization
4. Detect convergence via log-likelihood plateau

References:
    - Bishop, C. "Pattern Recognition and Machine Learning" (Ch. 9)
    - McLachlan & Peel "Finite Mixture Models" (2000)
"""

from __future__ import annotations
from typing import Tuple, Optional, List, Literal, Dict, Any
from dataclasses import dataclass, field
import logging
import warnings
from pose_uncertainty.utils.types import MixtureComponent, MixtureResult

import numpy as np
import numpy.typing as npt
from scipy.cluster.vq import kmeans2
from scipy.stats import multivariate_normal
from scipy.linalg import eigvalsh

# Configure logging
logger = logging.getLogger(__name__)

class RobustGaussianMixture:
    """
    Robust Gaussian Mixture Model with Uniform outlier component.
    
    This class implements the core EM algorithm for detecting and resolving
    pose ambiguities. It supports K Gaussians + 1 Uniform component.
    
    Mathematical Model:
    ------------------
        P(x | θ) = Σₖ πₖ 𝒩(x | μₖ, Σₖ) + π_u * U(x | A)
    
    Where:
        - μₖ: Component means
        - Σₖ: Component covariances
        - πₖ: Mixing weights for Gaussians
        - π_u: Weight for uniform (outlier) component
        - A: Bounding area for uniform distribution
    
    Parameters:
        n_components: Number of Gaussian components.
        init_method: Initialization method ('kmeans' or 'random').
        reg_covar: Regularization strength σ² added to covariance diagonal.
        max_iter: Maximum EM iterations.
        tol: Convergence threshold for log-likelihood change.
        random_state: Seed for reproducible initialization.
        min_component_weight: Minimum weight before component is considered dead.
    
    Attributes:
        components_: List of fitted MixtureComponent objects after calling fit().
        uniform_weight_: Weight of the uniform component.
        converged_: Boolean flag indicating if EM converged.
        n_iter_: Number of iterations actually performed.
        log_likelihood_: Final log-likelihood of the data.
        bounding_box_: (x_min, x_max, y_min, y_max) for uniform density.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        init_method: Literal['kmeans', 'random'] = 'kmeans',
        reg_covar: float = 1e-4,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        min_component_weight: float = 1e-3
    ) -> None:
        """
        Initialize the Robust Gaussian Mixture Model.
        
        Args:
            n_components: Number of Gaussian components (typically 1 or 2).
            init_method: 'kmeans' uses scipy.cluster.vq.kmeans2, 'random' samples from data.
            reg_covar: Regularization λ added to covariance diagonal (Σ + λI).
            max_iter: Maximum EM iterations before forced termination.
            tol: Stop if |log_likelihood_new - log_likelihood_old| < tol.
            random_state: Random seed for initialization.
            min_component_weight: Threshold below which a component is considered dead.
        """
        self.n_components = n_components
        self.init_method = init_method
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.min_component_weight = min_component_weight
        
        # Fitted attributes (set after fit())
        self.components_: List[MixtureComponent] = []
        self.uniform_weight_: float = 0.0
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.log_likelihood_: float = -np.inf
        self.bounding_box_: Tuple[float, float, float, float] = (0, 1, 0, 1)
        
        # Internal state
        self._rng = np.random.default_rng(random_state)
    
    def fit(
        self,
        samples: npt.NDArray[np.float64],
        initial_uniform_weight: float = 0.1,
        bounding_box: Optional[Tuple[float, float, float, float]] = None
    ) -> "RobustGaussianMixture":
        """
        Fit the mixture model using the EM algorithm.
        
        Args:
            samples: Array of shape (N, 2) with N 2D samples.
            initial_uniform_weight: Initial weight for the uniform component.
            bounding_box: Optional bounding box (x_min, x_max, y_min, y_max).
                      If not provided, it will be calculated from the samples.
        
        Returns:
            self (for method chaining).
        
        Raises:
            ValueError: If samples has wrong shape or insufficient points.
        """
        # Validate input
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError(f"Expected samples of shape (N, 2), got {samples.shape}")
        
        n_samples = samples.shape[0]
        if n_samples < self.n_components + 1:
            raise ValueError(
                f"Need at least {self.n_components + 1} samples, got {n_samples}"
            )
        
        # Use provided bounding box or calculate it
        if bounding_box is not None:
            x_min, x_max, y_min, y_max = bounding_box
        else:
            # Compute bounding box for uniform density
            margin = 1.0  # Small margin around data
            x_min, y_min = samples.min(axis=0) - margin
            x_max, y_max = samples.max(axis=0) + margin
        self.bounding_box_ = (x_min, x_max, y_min, y_max)
        self._uniform_density = 1.0 / ((x_max - x_min) * (y_max - y_min))
        
        # Initialize parameters
        self._initialize(samples, initial_uniform_weight)
        
        # EM iterations
        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            # E-step: Compute responsibilities
            responsibilities, uniform_resp = self._e_step(samples)
            
            # M-step: Update parameters
            self._m_step(samples, responsibilities, uniform_resp)
            
            # Compute log-likelihood
            current_ll = self._compute_log_likelihood(samples)
            
            # Check for dead components
            self._check_dead_components(responsibilities)
            
            # Check convergence
            ll_change = current_ll - prev_ll
            if abs(ll_change) < self.tol and iteration > 0:
                self.converged_ = True
                self.n_iter_ = iteration + 1
                logger.debug(f"EM converged at iteration {iteration + 1}")
                break
            
            prev_ll = current_ll
        else:
            self.n_iter_ = self.max_iter
            logger.warning(f"EM did not converge after {self.max_iter} iterations")
        
        self.log_likelihood_ = current_ll
        return self
    
    def _initialize(
        self,
        samples: npt.NDArray[np.float64],
        initial_uniform_weight: float
    ) -> None:
        """
        Initialize mixture parameters.
        
        Args:
            samples: Data samples of shape (N, 2).
            initial_uniform_weight: Initial weight for uniform component.
        """
        n_samples, dim = samples.shape
        
        # Initialize uniform weight
        self.uniform_weight_ = initial_uniform_weight
        remaining_weight = 1.0 - initial_uniform_weight
        
        if self.init_method == 'kmeans':
            # Use scipy's kmeans2 for initialization
            try:
                centroids, labels = kmeans2(
                    samples, 
                    self.n_components, 
                    minit='++',  # k-means++ initialization
                    seed=self.random_state
                )
            except Exception as e:
                logger.warning(f"K-means initialization failed: {e}. Using random init.")
                centroids = self._random_init(samples)
                labels = self._assign_labels(samples, centroids)
        else:
            # Random initialization: sample from data
            centroids = self._random_init(samples)
            labels = self._assign_labels(samples, centroids)
        
        # Initialize components
        self.components_ = []
        for k in range(self.n_components):
            mask = labels == k
            n_assigned = mask.sum()
            
            if n_assigned > 0:
                mean = centroids[k]
                # Initial covariance from assigned points
                if n_assigned > 1:
                    cov = np.cov(samples[mask].T)
                    if cov.ndim == 0:  # Single feature edge case
                        cov = np.array([[cov, 0], [0, cov]])
                else:
                    cov = np.eye(dim) * 10.0  # Default spread
                
                # Regularize covariance
                cov = self._regularize_covariance(cov)
                weight = (remaining_weight * n_assigned) / n_samples
            else:
                # No points assigned - initialize with global stats
                mean = samples.mean(axis=0)
                cov = self._regularize_covariance(np.cov(samples.T))
                weight = remaining_weight / self.n_components
            
            self.components_.append(MixtureComponent(
                mean=mean.astype(np.float32),
                covariance=cov.astype(np.float32),
                weight=float(weight),
                n_samples=int(n_assigned)
            ))
        
        # Normalize weights
        self._normalize_weights()
    
    def _random_init(self, samples: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Random initialization by sampling from data."""
        indices = self._rng.choice(len(samples), self.n_components, replace=False)
        return samples[indices].copy()
    
    def _assign_labels(
        self, 
        samples: npt.NDArray[np.float64], 
        centroids: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.int64]:
        """Assign samples to nearest centroid."""
        distances = np.zeros((len(samples), len(centroids)))
        for k, centroid in enumerate(centroids):
            distances[:, k] = np.linalg.norm(samples - centroid, axis=1)
        return np.argmin(distances, axis=1)
    
    def _regularize_covariance(
        self, 
        cov: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Regularize covariance matrix to ensure positive definiteness.
        
        Checks eigenvalues and adds regularization if needed.
        
        Args:
            cov: Covariance matrix of shape (D, D).
        
        Returns:
            Regularized covariance matrix.
        """
        cov = np.atleast_2d(cov)
        if cov.shape == (1, 1):
            cov = np.array([[cov[0, 0], 0], [0, cov[0, 0]]])
        
        # Check eigenvalues
        try:
            eigenvalues = eigvalsh(cov)
            min_eig = eigenvalues.min()
        except np.linalg.LinAlgError:
            min_eig = -1  # Force regularization
        
        if min_eig < self.reg_covar:
            # Apply regularization: Σ̃ = Σ + σ²I
            reg_amount = self.reg_covar - min(0, min_eig)
            cov = cov + reg_amount * np.eye(cov.shape[0])
            logger.debug(
                f"Applied covariance regularization: min_eig={min_eig:.2e}, "
                f"reg={reg_amount:.2e}"
            )
        
        return cov
    
    def _normalize_weights(self) -> None:
        """Ensure all weights sum to 1."""
        total = self.uniform_weight_ + sum(c.weight for c in self.components_)
        if total > 0:
            self.uniform_weight_ /= total
            for c in self.components_:
                c.weight /= total
    
    def _e_step(
        self, 
        samples: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        E-step: Compute responsibilities (posterior probabilities).
        
        For each sample n and component k:
            R_{n,k} = π_k * N(x_n | μ_k, Σ_k) / P(x_n)
        
        Where P(x_n) = Σ_k π_k * N(x_n | μ_k, Σ_k) + π_u * U(x_n)
        
        Args:
            samples: Data samples of shape (N, 2).
        
        Returns:
            Tuple of (gaussian_responsibilities, uniform_responsibilities).
            - gaussian_responsibilities: Shape (N, K)
            - uniform_responsibilities: Shape (N,)
        """
        n_samples = len(samples)
        
        # Compute Gaussian densities
        gaussian_probs = np.zeros((n_samples, self.n_components))
        for k, comp in enumerate(self.components_):
            try:
                #rv = multivariate_normal(mean=comp.mean, cov=comp.covariance, allow_singular=True)
                #gaussian_probs[:, k] = comp.weight * rv.pdf(samples)
                pdf_values = self._gaussian_pdf(samples, comp.mean, comp.covariance)
                gaussian_probs[:, k] = comp.weight * pdf_values
            except Exception as e:
                logger.warning(f"Component {k} density computation failed: {e}")
                gaussian_probs[:, k] = 0.0
        
        # Uniform density contribution
        uniform_probs = self.uniform_weight_ * self._uniform_density * np.ones(n_samples)
        
        # Total probability (normalizer)
        total_probs = gaussian_probs.sum(axis=1) + uniform_probs
        total_probs = np.maximum(total_probs, 1e-300)  # Prevent division by zero
        
        # Responsibilities
        gaussian_resp = gaussian_probs / total_probs[:, np.newaxis]
        uniform_resp = uniform_probs / total_probs
        
        return gaussian_resp, uniform_resp
    
    def _m_step(
        self,
        samples: npt.NDArray[np.float64],
        responsibilities: npt.NDArray[np.float64],
        uniform_resp: npt.NDArray[np.float64]
    ) -> None:
        """
        M-step: Update parameters using current responsibilities.
        
        Update formulas:
            π_k = (1/N) Σ_n R_{n,k}
            μ_k = Σ_n R_{n,k} x_n / Σ_n R_{n,k}
            Σ_k = Σ_n R_{n,k} (x_n - μ_k)(x_n - μ_k)^T / Σ_n R_{n,k}
        
        Args:
            samples: Data samples of shape (N, 2).
            responsibilities: Gaussian responsibilities of shape (N, K).
            uniform_resp: Uniform responsibilities of shape (N,).
        """
        n_samples = len(samples)
        
        for k, comp in enumerate(self.components_):
            r_k = responsibilities[:, k]
            n_k = r_k.sum()
            
            if n_k < 1e-10:
                # Dead component - skip update
                continue
            
            # Update mean: μ_k = Σ_n R_{n,k} x_n / Σ_n R_{n,k}
            new_mean = (r_k[:, np.newaxis] * samples).sum(axis=0) / n_k
            
            # Update covariance: Σ_k = Σ_n R_{n,k} (x_n - μ_k)(x_n - μ_k)^T / Σ_n R_{n,k}
            diff = samples - new_mean
            new_cov = (r_k[:, np.newaxis, np.newaxis] * 
                      (diff[:, :, np.newaxis] @ diff[:, np.newaxis, :])).sum(axis=0) / n_k
            
            # Regularize covariance
            new_cov = self._regularize_covariance(new_cov)
            
            # Update weight: π_k = (1/N) Σ_n R_{n,k}
            new_weight = n_k / n_samples
            
            # Store updates
            comp.mean = new_mean.astype(np.float32)
            comp.covariance = new_cov.astype(np.float32)
            comp.weight = float(new_weight)
            comp.n_samples = int(n_k)
        
        # Update uniform weight
        self.uniform_weight_ = uniform_resp.sum() / n_samples
        
        # Normalize weights
        self._normalize_weights()
    
    def _check_dead_components(self, responsibilities: npt.NDArray[np.float64]) -> None:
        """Check for and handle dead components (π_k → 0)."""
        for k, comp in enumerate(self.components_):
            if comp.weight < self.min_component_weight:
                logger.warning(
                    f"Component {k} is dying (weight={comp.weight:.2e}). "
                    "Consider using fewer components."
                )
    
    def _compute_log_likelihood(self, samples: npt.NDArray[np.float64]) -> float:
        """
        Compute log-likelihood of data under current model.
        
        log L = Σ_n log P(x_n)
              = Σ_n log [Σ_k π_k N(x_n | μ_k, Σ_k) + π_u U(x_n)]
        
        Args:
            samples: Data samples of shape (N, 2).
        
        Returns:
            Log-likelihood value.
        """
        n_samples = len(samples)
        
        # Compute Gaussian densities
        probs = np.zeros(n_samples)
        for comp in self.components_:
            try:
                #rv = multivariate_normal(mean=comp.mean, cov=comp.covariance, allow_singular=True)
                #probs += comp.weight * rv.pdf(samples)
                pdf_values = self._gaussian_pdf(samples, comp.mean, comp.covariance)
                probs += comp.weight * pdf_values
            except Exception:
                pass
        
        # Add uniform component
        probs += self.uniform_weight_ * self._uniform_density
        
        # Log-likelihood
        probs = np.maximum(probs, 1e-300)
        return np.sum(np.log(probs))
    
    def _gaussian_pdf(
        self, 
        samples: npt.NDArray[np.float64], 
        mean: npt.NDArray[np.float64], 
        cov: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Calcula la PDF multivariante vectorizada para N muestras.
        Reemplaza a scipy.stats.multivariate_normal.pdf para máxima eficiencia.
        """
        d = samples.shape[1]
        
        try:
            # Inversa y determinante (muy rápido para 2x2)
            inv_cov = np.linalg.inv(cov)
            det_cov = np.linalg.det(cov)
            
            # Protección contra matrices singulares o mal condicionadas
            if det_cov <= 0:
                return np.zeros(len(samples), dtype=np.float64)
                
        except np.linalg.LinAlgError:
            return np.zeros(len(samples), dtype=np.float64)
            
        # Constante de normalización
        norm_const = 1.0 / np.sqrt(((2 * np.pi) ** d) * det_cov)
        
        # Desviación respecto a la media (N, 2)
        diff = samples - mean 
        
        # Cálculo eficiente de Mahalanobis con Einstein Summation:
        # 'ni' (muestras x dimensiones), 'ij' (inversa covarianza), 'nj' (muestras x dimensiones)
        # El resultado es un array 1D de tamaño N.
        mahalanobis_sq = np.einsum('ni,ij,nj->n', diff, inv_cov, diff)
        
        # Prevenir underflow extremo antes de la exponencial
        mahalanobis_sq = np.clip(mahalanobis_sq, a_min=None, a_max=700)
        
        return norm_const * np.exp(-0.5 * mahalanobis_sq)
    
    def compute_aic(self, n_samples: int) -> float:
        """
        Compute Akaike Information Criterion.
        
        AIC = 2k - 2 ln(L)
        
        Where k = number of parameters:
            - K Gaussians: K * (2 means + 3 covariance params + 1 weight) = 6K
            - 1 Uniform: 1 weight
            - Total: 6K + 1 (but weights sum to 1, so -1)
        
        Args:
            n_samples: Number of data samples.
        
        Returns:
            AIC value (lower is better).
        """
        # Parameters: K * (mean: 2 + cov: 3 unique + weight: 1) - 1 constraint
        n_params = self.n_components * 6  # 2 mean + 3 cov + 1 weight per Gaussian
        # Note: weights sum to 1, so we have (K+1)-1 = K free weight parameters
        
        return 2 * n_params - 2 * self.log_likelihood_
    
    def compute_bic(self, n_samples: int) -> float:
        """
        Compute Bayesian Information Criterion.
        
        BIC = k ln(n) - 2 ln(L)
        
        Args:
            n_samples: Number of data samples.
        
        Returns:
            BIC value (lower is better).
        """
        n_params = self.n_components * 6
        return n_params * np.log(n_samples) - 2 * self.log_likelihood_
    
    def get_mode(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Extract the dominant mode (component with highest mixing weight).
        
        Returns:
            Tuple of (mean, covariance) for the dominant component.
        """
        if not self.components_:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        best_idx = np.argmax([c.weight for c in self.components_])
        best_comp = self.components_[best_idx]
        
        return best_comp.mean.astype(np.float64), best_comp.covariance.astype(np.float64)
    
    def predict_proba(
        self,
        samples: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Compute posterior probabilities for new samples.
        
        Args:
            samples: Array of shape (N, 2).
        
        Returns:
            Array of shape (N, K+1) with posteriors for K Gaussians + 1 Uniform.
        """
        gaussian_resp, uniform_resp = self._e_step(samples)
        return np.column_stack([gaussian_resp, uniform_resp])


def select_best_model(
    samples: npt.NDArray[np.float64],
    aic_weight: float = 0.5,
    bic_weight: float = 0.5,
    reg_covar: float = 1e-4,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    bounding_box: Optional[Tuple[float, float, float, float]] = None
) -> MixtureResult:
    """
    Select best model topology (unimodal vs bimodal) using AIC/BIC.
    
    Compares:
        - Model A (Bimodal): 2 Gaussians + 1 Uniform
        - Model B (Unimodal): 1 Gaussian + 1 Uniform
    
    Selection criterion:
        Score = aic_weight * AIC + bic_weight * BIC
        Lower score wins.
    
    Args:
        samples: Data samples of shape (N, 2).
        aic_weight: Weight for AIC in combined score.
        bic_weight: Weight for BIC in combined score.
        reg_covar: Regularization strength for covariance.
        max_iter: Maximum EM iterations.
        tol: Convergence threshold.
        random_state: Random seed.
    
    Returns:
        MixtureResult with the winning model's parameters.
    """
    samples = np.asarray(samples, dtype=np.float64)
    n_samples = len(samples)
    
    results: Dict[str, Tuple[RobustGaussianMixture, float]] = {}
    
    # Fit Model A (Bimodal): 2 Gaussians + Uniform
    model_a = RobustGaussianMixture(
        n_components=2,
        reg_covar=reg_covar,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )
    try:
        model_a.fit(samples, bounding_box=bounding_box)
        aic_a = model_a.compute_aic(n_samples)
        bic_a = model_a.compute_bic(n_samples)
        score_a = aic_weight * aic_a + bic_weight * bic_a
        results['bimodal'] = (model_a, score_a)
        logger.debug(f"Model A (bimodal): AIC={aic_a:.2f}, BIC={bic_a:.2f}, Score={score_a:.2f}")
    except Exception as e:
        logger.warning(f"Bimodal model fitting failed: {e}")
        results['bimodal'] = (None, np.inf)
    
    # Fit Model B (Unimodal): 1 Gaussian + Uniform
    model_b = RobustGaussianMixture(
        n_components=1,
        reg_covar=reg_covar,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )
    try:
        model_b.fit(samples, bounding_box=bounding_box)
        aic_b = model_b.compute_aic(n_samples)
        bic_b = model_b.compute_bic(n_samples)
        score_b = aic_weight * aic_b + bic_weight * bic_b
        results['unimodal'] = (model_b, score_b)
        logger.debug(f"Model B (unimodal): AIC={aic_b:.2f}, BIC={bic_b:.2f}, Score={score_b:.2f}")
    except Exception as e:
        logger.warning(f"Unimodal model fitting failed: {e}")
        results['unimodal'] = (None, np.inf)
    
    # Select winner
    best_type = min(results, key=lambda x: results[x][1])
    best_model, best_score = results[best_type]
    
    if best_model is None:
        raise RuntimeError("All model fits failed")
    
    # Extract results
    best_mean, best_cov = best_model.get_mode()
    
    return MixtureResult(
        model_type=best_type,
        best_mean=best_mean,
        best_covariance=best_cov,
        log_likelihood=best_model.log_likelihood_,
        aic=best_model.compute_aic(n_samples),
        bic=best_model.compute_bic(n_samples),
        n_iterations=best_model.n_iter_,
        converged=best_model.converged_,
        components=best_model.components_,
        uniform_weight=best_model.uniform_weight_
    )


def fit_with_outer_loop(
    samples: npt.NDArray[np.float64],
    n_outer_iterations: int = 5,
    n_resamples: int = 500,
    aic_weight: float = 0.5,
    bic_weight: float = 0.5,
    reg_covar: float = 1e-4,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    bounding_box: Optional[Tuple[float, float, float, float]] = None
) -> MixtureResult:
    """
    Fit mixture model with outer loop for stability (re-sampling strategy).
    
    Algorithm:
        1. Loop T times:
            a. Bootstrap resample from original samples
            b. Fit Model A vs Model B → Select Winner
            c. Store winning prediction
        2. Aggregate predictions from all iterations
    
    Args:
        samples: Original data samples of shape (N, 2).
        n_outer_iterations: Number of outer loop iterations (T).
        n_resamples: Number of samples to draw in each bootstrap.
        aic_weight: Weight for AIC.
        bic_weight: Weight for BIC.
        reg_covar: Regularization strength.
        max_iter: Maximum EM iterations per fit.
        tol: Convergence threshold.
        random_state: Random seed.
    
    Returns:
        Aggregated MixtureResult with mean of winning predictions.
    """
    rng = np.random.default_rng(random_state)
    samples = np.asarray(samples, dtype=np.float64)
    n_samples = len(samples)
    
    winning_means: List[npt.NDArray[np.float64]] = []
    winning_covs: List[npt.NDArray[np.float64]] = []
    model_type_counts: Dict[str, int] = {'unimodal': 0, 'bimodal': 0}
    all_results: List[MixtureResult] = []
    
    for t in range(n_outer_iterations):
        # Bootstrap resample
        indices = rng.choice(n_samples, size=min(n_resamples, n_samples), replace=True)
        resampled = samples[indices]
        
        # Fit and select model
        seed = None if random_state is None else random_state + t
        try:
            result = select_best_model(
                resampled,
                aic_weight=aic_weight,
                bic_weight=bic_weight,
                reg_covar=reg_covar,
                max_iter=max_iter,
                tol=tol,
                random_state=seed,
                bounding_box=bounding_box
            )
            
            winning_means.append(result.best_mean)
            winning_covs.append(result.best_covariance)
            model_type_counts[result.model_type] += 1
            all_results.append(result)

            # Agregar registro para inspeccionar las covarianzas
            logger.debug(f"Outer iteration {t+1}: {result.model_type} won")
            logger.debug(f"Winning covariance (iteration {t+1}):\n{result.best_covariance}")
            
            # Verificar si la covarianza contiene valores inválidos
            if np.any(np.isnan(result.best_covariance)) or np.any(np.isinf(result.best_covariance)):
                logger.warning(f"Invalid covariance detected in iteration {t+1}: {result.best_covariance}")
            
            logger.debug(f"Outer iteration {t+1}: {result.model_type} won")
        except Exception as e:
            logger.warning(f"Outer iteration {t+1} failed: {e}")
            continue
    
    if not winning_means:
        raise RuntimeError("All outer loop iterations failed")
    
    # Aggregate results
    aggregated_mean = np.mean(winning_means, axis=0)
    

    if len(winning_covs) == 1:
        logger.warning("Only one covariance in winning_covs. Using it directly as aggregated_cov.")
        aggregated_cov = winning_covs[0]
    else:
        # For covariance, use the mean of covariances + variance of means
        mean_of_covs = np.mean(winning_covs, axis=0)
        var_of_means = np.cov(np.array(winning_means).T)
        if var_of_means.ndim == 0:
            var_of_means = np.array([[var_of_means, 0], [0, var_of_means]])
        aggregated_cov = mean_of_covs + var_of_means


    # Verificar si la covarianza agregada es válida
    logger.debug(f"Aggregated covariance:\n{aggregated_cov}")
    if np.any(np.isnan(aggregated_cov)) or np.any(np.isinf(aggregated_cov)):
        logger.warning(f"Invalid aggregated covariance: {aggregated_cov}")
    
    # Determine dominant model type
    dominant_type = max(model_type_counts, key=model_type_counts.get)
    
    # Use the best single result's auxiliary info
    best_result = min(all_results, key=lambda r: r.aic *aic_weight + r.bic * bic_weight)
    
    return MixtureResult(
        model_type=dominant_type,
        best_mean=aggregated_mean,
        best_covariance=aggregated_cov,
        log_likelihood=best_result.log_likelihood,
        aic=best_result.aic,
        bic=best_result.bic,
        n_iterations=sum(r.n_iterations for r in all_results),
        converged=all(r.converged for r in all_results),
        components=best_result.components,
        uniform_weight=best_result.uniform_weight
    )


# =============================================================================
# Legacy Class for Backward Compatibility
# =============================================================================

"""
Mathematical Core: Pure NumPy implementations of statistical algorithms.

This subpackage contains the mathematical "brain" of the uncertainty quantification
system. All implementations are framework-agnostic (no PyTorch/TensorFlow dependencies)
to ensure portability and testability.

Modules:
-------
- mixture.py: Restricted Gaussian Mixture Model for bimodal pose distributions
- sampling.py: Monte Carlo sampling strategies from heatmap distributions
- mrf_decoder.py: Markov Random Field graph decoding and tree MAP inference
- skeleton.py: Kinematic skeleton topology and bone length priors

Design Principle:
----------------
Separation of concerns: Keep pure mathematical algorithms isolated from
deep learning frameworks. This enables:
    1. Independent testing with synthetic data
    2. Easy migration to other frameworks (JAX, TensorFlow)
    3. CPU-only deployments without GPU dependencies
"""

from typing import List

__all__: List[str] = ["mixture", "sampling", "mrf_decoder", "skeleton"]

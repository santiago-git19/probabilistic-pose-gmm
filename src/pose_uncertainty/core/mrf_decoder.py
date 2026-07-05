"""
Markov Random Field Decoder for Skeleton-aware Pose Refinement.

Implements exact **Belief Propagation** (max-product message passing) on the
tree-structured COCO skeleton graph to resolve inter-keypoint ambiguities
(limb swaps) that arise from per-keypoint GMM decoding.

Algorithm Overview
------------------
1. Each keypoint *k* provides a set of *candidates* (GMM component means)
   with associated unary scores (mixing weight + compactness).
2. The skeleton graph defines *pairwise potentials* based on anatomical
   bone lengths: a candidate pair ``(mu_parent, mu_child)`` is penalised
   if their Euclidean distance deviates from the expected bone length.
3. **Leaves-to-root pass**: each child sends a message to its parent
   indicating, for each parent candidate, the best achievable child score.
4. **Root-to-leaves backtrack**: starting from the root's best candidate,
   each child recovers which of its candidates was optimal given the
   parent's choice.
5. The output is the globally optimal configuration of keypoint coordinates
   that maximises the joint probability ``P(pose) = prod(unaries) * prod(pairwise)``.

Mathematical Formulation
------------------------
For a tree graph G = (V, E):

    max_{x} ∏_{i∈V} φ_i(x_i) · ∏_{(i,j)∈E} ψ_{ij}(x_i, x_j)

where:
    - φ_i(x_i) = unary potential of candidate x_i for node i
    - ψ_{ij}(x_i, x_j) = pairwise potential measuring bone-length compatibility

We work in log-space for numerical stability:

    max_{x} Σ_{i∈V} log φ_i(x_i) + Σ_{(i,j)∈E} log ψ_{ij}(x_i, x_j)

References
----------
- Bishop, C. "Pattern Recognition and Machine Learning" (Ch. 8.4)
- Felzenszwalb & Huttenlocher, "Pictorial Structures for Object Recognition" (2005)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .skeleton import SkeletonGraph, COCO_SKELETON

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class UnaryPotential:
    """Candidate set for a single keypoint node.

    Attributes
    ----------
    means : list of ndarray
        Candidate positions ``[mu_1, mu_2, ...]``, each shape ``(2,)``.
    covariances : list of ndarray
        Covariance matrices ``[Sigma_1, Sigma_2, ...]``, each shape ``(2,2)``.
    weights : list of float
        GMM mixing weights ``[pi_1, pi_2, ...]`` (sum to ~1).
    """

    means: List[npt.NDArray[np.float64]]
    covariances: List[npt.NDArray[np.float64]]
    weights: List[float]

    @property
    def num_candidates(self) -> int:
        return len(self.means)


# ──────────────────────────────────────────────────────────────────────────
# MRF Decoder
# ──────────────────────────────────────────────────────────────────────────

class MRFDecoder:
    """Skeleton-aware decoder using Belief Propagation on the pose graph.

    Only keypoints with K >= 2 GMM components participate in the MRF
    optimisation.  Unimodal keypoints (K=1) are passed through unchanged.

    Parameters
    ----------
    skeleton : SkeletonGraph
        Skeleton topology with bone-length statistics.
    bone_length_sigma : float
        Controls tolerance of the pairwise bone-length prior.
        The effective standard deviation of the Gaussian prior is
        ``bone_length_sigma * empirical_std * sqrt(area)``.
        Higher values = more permissive.  Default 2.0.
    use_covariance_score : bool
        If True, incorporate the inverse determinant of each candidate's
        covariance matrix into the unary score (favouring compact
        components).  Default True.
    """

    def __init__(
        self,
        skeleton: SkeletonGraph = COCO_SKELETON,
        bone_length_sigma: float = 2.0,
        use_covariance_score: bool = True,
    ) -> None:
        self.skeleton = skeleton
        self.bone_length_sigma = bone_length_sigma
        self.use_covariance_score = use_covariance_score

    # ── Unary score ──────────────────────────────────────────────────────

    def _compute_unary_scores(
        self, unary: UnaryPotential
    ) -> npt.NDArray[np.float64]:
        """Compute log-unary scores for each candidate.

        Score = log(pi_k) + optional compactness bonus

        Returns shape ``(num_candidates,)``.
        """
        scores = np.zeros(unary.num_candidates, dtype=np.float64)

        for k in range(unary.num_candidates):
            # Base: log of mixing weight
            w = max(unary.weights[k], 1e-30)
            scores[k] = math.log(w)

            # Optional: penalise dispersed candidates via log(1/det(Sigma))
            if self.use_covariance_score:
                det = np.linalg.det(unary.covariances[k])
                det = max(det, 1e-30)
                # Using -0.5 * log(det) as a precision bonus
                scores[k] += -0.5 * math.log(det)

        return scores

    # ── Pairwise score ───────────────────────────────────────────────────

    def _compute_pairwise_score(
        self,
        parent_pos: npt.NDArray[np.float64],
        child_pos: npt.NDArray[np.float64],
        parent_idx: int,
        child_idx: int,
        scale: float,
    ) -> float:
        """Compute log-pairwise potential between a parent and child candidate.

        Uses a Gaussian prior on bone length:

            log ψ(x_p, x_c) = -0.5 * ((d - μ_bone) / σ_bone)²

        Parameters
        ----------
        parent_pos, child_pos : ndarray of shape (2,)
            Candidate positions.
        parent_idx, child_idx : int
            Keypoint indices (for looking up bone stats).
        scale : float
            ``sqrt(area)`` for de-normalising bone-length ratios.
        """
        d = float(np.linalg.norm(parent_pos - child_pos))

        mean_ratio, std_ratio = self.skeleton.get_bone_stats(parent_idx, child_idx)
        bone_mean = mean_ratio * scale
        bone_std = std_ratio * scale * self.bone_length_sigma

        if bone_std < 1e-6:
            return 0.0

        z = (d - bone_mean) / bone_std
        return -0.5 * z * z

    # ── Belief Propagation ───────────────────────────────────────────────

    def decode(
        self,
        unary_potentials: Dict[int, UnaryPotential],
        area: float,
    ) -> Dict[int, npt.NDArray[np.float64]]:
        """Run max-product Belief Propagation and return optimal coordinates.

        Parameters
        ----------
        unary_potentials : dict
            ``{keypoint_id: UnaryPotential}`` for each keypoint.
            Keypoints not present are left unchanged.
        area : float
            Bounding-box area (used to de-normalise bone-length stats).

        Returns
        -------
        dict
            ``{keypoint_id: coords_xy}`` with the globally optimal
            coordinate for each keypoint that participated in decoding.
        """
        scale = math.sqrt(max(area, 1.0))

        # Precompute unary log-scores for every node
        unary_scores: Dict[int, npt.NDArray[np.float64]] = {}
        for node, up in unary_potentials.items():
            unary_scores[node] = self._compute_unary_scores(up)

        # ── Pass 1: Leaves -> Root (collect messages) ────────────────────
        # message[child] = for each parent candidate, the (best child score,
        #                   best child index)
        # Shape: messages_to_parent[child] has shape (num_parent_candidates,)
        # with the max aggregate score, and argmax for backtracking.
        messages_score: Dict[int, npt.NDArray[np.float64]] = {}
        messages_argmax: Dict[int, npt.NDArray[np.int64]] = {}

        for node in self.skeleton.leaves_to_root_order():
            if node == self.skeleton.root:
                continue  # root sends no message

            parent = self.skeleton.parent(node)
            if parent is None:
                continue

            # Skip if either node or parent has no unary potentials
            if node not in unary_potentials or parent not in unary_potentials:
                continue

            up_child = unary_potentials[node]
            up_parent = unary_potentials[parent]
            child_unary = unary_scores[node]

            num_parent_cands = up_parent.num_candidates
            num_child_cands = up_child.num_candidates

            # For each parent candidate p, find the best child candidate c
            msg_score = np.full(num_parent_cands, -np.inf, dtype=np.float64)
            msg_argmax = np.zeros(num_parent_cands, dtype=np.int64)

            for p in range(num_parent_cands):
                parent_pos = up_parent.means[p]

                for c in range(num_child_cands):
                    child_pos = up_child.means[c]

                    # Total score for this child candidate given parent candidate p
                    score = child_unary[c]
                    score += self._compute_pairwise_score(
                        parent_pos, child_pos, parent, node, scale
                    )
                    # Include messages from node's own children (already computed)
                    for grandchild in self.skeleton.children(node):
                        if grandchild in messages_score:
                            # messages_score[grandchild] has one entry per candidate of node
                            if c < len(messages_score[grandchild]):
                                score += messages_score[grandchild][c]

                    if score > msg_score[p]:
                        msg_score[p] = score
                        msg_argmax[p] = c

            messages_score[node] = msg_score
            messages_argmax[node] = msg_argmax

        # ── Pass 2: Root -> Leaves (backtrack) ───────────────────────────
        chosen: Dict[int, int] = {}  # node -> chosen candidate index

        # Root: choose best candidate considering its unary + messages from children
        root = self.skeleton.root
        if root in unary_potentials:
            up_root = unary_potentials[root]
            root_unary = unary_scores[root]
            best_root_score = -np.inf
            best_root_idx = 0

            for r in range(up_root.num_candidates):
                score = root_unary[r]
                for child in self.skeleton.children(root):
                    if child in messages_score:
                        if r < len(messages_score[child]):
                            score += messages_score[child][r]
                if score > best_root_score:
                    best_root_score = score
                    best_root_idx = r

            chosen[root] = best_root_idx

        # BFS from root to propagate choices
        for node in self.skeleton.tree_order():
            if node == root:
                continue

            parent = self.skeleton.parent(node)
            if parent is None or parent not in chosen:
                # Parent not decoded -> use argmax unary for this node
                if node in unary_scores:
                    chosen[node] = int(np.argmax(unary_scores[node]))
                continue

            if node not in messages_argmax:
                # No message was sent (node might be unimodal or missing)
                if node in unary_scores:
                    chosen[node] = int(np.argmax(unary_scores[node]))
                continue

            parent_choice = chosen[parent]
            argmax_arr = messages_argmax[node]
            if parent_choice < len(argmax_arr):
                chosen[node] = int(argmax_arr[parent_choice])
            else:
                chosen[node] = int(np.argmax(unary_scores.get(node, [0])))

        # ── Collect results ──────────────────────────────────────────────
        result: Dict[int, npt.NDArray[np.float64]] = {}
        for node, cand_idx in chosen.items():
            if node in unary_potentials:
                up = unary_potentials[node]
                cand_idx = min(cand_idx, up.num_candidates - 1)
                result[node] = up.means[cand_idx].copy()

        return result

    def decode_pose(
        self,
        gmm_results: list,
        area: float,
        current_coords: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """High-level API: refine a full pose using MRF graph decoding.

        Only keypoints whose GMM selected K=2 (bimodal) participate in
        the MRF.  Unimodal keypoints keep their original coordinates.

        Parameters
        ----------
        gmm_results : list of MixtureResult
            One per keypoint (17 for COCO).
        area : float
            Bounding-box area for bone-length de-normalisation.
        current_coords : ndarray of shape (num_kp, 2)
            Current decoded coordinates (fallback for keypoints not in MRF).

        Returns
        -------
        ndarray of shape (num_kp, 2)
            Refined coordinates.
        """
        from ..utils.types import MixtureResult

        num_kp = len(gmm_results)
        refined = current_coords.copy()

        # Build unary potentials only for bimodal keypoints (K=2)
        unary_potentials: Dict[int, UnaryPotential] = {}
        bimodal_nodes: List[int] = []

        for k in range(num_kp):
            mr: MixtureResult = gmm_results[k]
            comps = mr.components

            if len(comps) >= 2:
                # Bimodal -> provide all candidates to MRF
                means = [c.mean.astype(np.float64) for c in comps]
                covs = [c.covariance.astype(np.float64) for c in comps]
                weights = [float(c.weight) for c in comps]
                unary_potentials[k] = UnaryPotential(
                    means=means, covariances=covs, weights=weights
                )
                bimodal_nodes.append(k)
            else:
                # Unimodal -> single candidate (no ambiguity)
                # Still add to unary_potentials so pairwise constraints
                # with bimodal neighbours can be evaluated
                means = [c.mean.astype(np.float64) for c in comps]
                covs = [c.covariance.astype(np.float64) for c in comps]
                weights = [float(c.weight) for c in comps]
                unary_potentials[k] = UnaryPotential(
                    means=means, covariances=covs, weights=weights
                )

        if not bimodal_nodes:
            logger.debug("MRF decode: no bimodal keypoints, skipping")
            return refined

        logger.debug(
            "MRF decode: %d bimodal keypoints: %s",
            len(bimodal_nodes), bimodal_nodes,
        )

        # Run Belief Propagation
        mrf_coords = self.decode(unary_potentials, area)

        # Replace coordinates for all nodes that MRF resolved
        for node, coords in mrf_coords.items():
            refined[node] = coords.astype(np.float32)

        return refined

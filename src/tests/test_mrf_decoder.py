"""
Tests for the MRF Graph Decoder (Propuesta C).

Validates:
1. SkeletonGraph topology (17 nodes, 16 edges, tree structure)
2. MRFDecoder pairwise potentials (penalise impossible bone lengths)
3. MRFDecoder.decode() with synthetic swap scenarios
4. MRFDecoder.decode_pose() end-to-end integration
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pose_uncertainty.core.skeleton import (
    COCO_SKELETON,
    COCO_SKELETON_EDGES,
    COCO_BONE_LENGTH_STATS,
    COCO_KEYPOINT_NAMES,
    NUM_COCO_KEYPOINTS,
    SkeletonGraph,
)
from pose_uncertainty.core.mrf_decoder import MRFDecoder, UnaryPotential
from pose_uncertainty.utils.types import MixtureComponent, MixtureResult


# ══════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def skeleton() -> SkeletonGraph:
    return COCO_SKELETON


@pytest.fixture
def decoder() -> MRFDecoder:
    return MRFDecoder(skeleton=COCO_SKELETON, bone_length_sigma=2.0)


def _make_mixture_result(
    mean: np.ndarray,
    components: list[MixtureComponent],
    model_type: str = "bimodal",
) -> MixtureResult:
    """Helper to build a MixtureResult for testing."""
    return MixtureResult(
        model_type=model_type,
        best_mean=mean.astype(np.float64),
        best_covariance=np.eye(2, dtype=np.float64),
        log_likelihood=-100.0,
        aic=200.0,
        bic=210.0,
        n_iterations=10,
        converged=True,
        components=components,
        uniform_weight=0.05,
    )


def _make_component(mean: np.ndarray, weight: float = 0.5) -> MixtureComponent:
    """Helper to build a single MixtureComponent."""
    return MixtureComponent(
        mean=mean.astype(np.float32),
        covariance=np.eye(2, dtype=np.float32) * 2.0,
        weight=weight,
        n_samples=100,
    )


# ══════════════════════════════════════════════════════════════════════════
# 1. SkeletonGraph topology tests
# ══════════════════════════════════════════════════════════════════════════

class TestSkeletonGraph:
    """Validate the COCO skeleton graph structure."""

    def test_num_nodes(self, skeleton: SkeletonGraph):
        assert skeleton.num_nodes == 17

    def test_num_edges(self, skeleton: SkeletonGraph):
        assert len(skeleton.edges) == 16

    def test_is_tree(self, skeleton: SkeletonGraph):
        """A tree with N nodes has exactly N-1 edges."""
        assert len(skeleton.edges) == skeleton.num_nodes - 1

    def test_root_is_nose(self, skeleton: SkeletonGraph):
        assert skeleton.root == 0
        assert skeleton.parent(0) is None

    def test_every_non_root_has_parent(self, skeleton: SkeletonGraph):
        for i in range(1, skeleton.num_nodes):
            assert skeleton.parent(i) is not None, (
                f"Node {i} ({COCO_KEYPOINT_NAMES[i]}) has no parent"
            )

    def test_tree_order_starts_with_root(self, skeleton: SkeletonGraph):
        order = skeleton.tree_order()
        assert order[0] == skeleton.root
        assert len(order) == skeleton.num_nodes

    def test_leaves_to_root_ends_with_root(self, skeleton: SkeletonGraph):
        order = skeleton.leaves_to_root_order()
        assert order[-1] == skeleton.root
        assert len(order) == skeleton.num_nodes

    def test_children_of_nose(self, skeleton: SkeletonGraph):
        children = skeleton.children(0)
        # nose connects to: left_eye(1), right_eye(2), left_shoulder(5), right_shoulder(6)
        assert set(children) == {1, 2, 5, 6}

    def test_wrists_are_leaves(self, skeleton: SkeletonGraph):
        # left_wrist(9) and right_wrist(10) should have no children
        assert skeleton.children(9) == []
        assert skeleton.children(10) == []

    def test_ankles_are_leaves(self, skeleton: SkeletonGraph):
        assert skeleton.children(15) == []
        assert skeleton.children(16) == []

    def test_bone_stats_coverage(self, skeleton: SkeletonGraph):
        """Every edge should have bone-length statistics."""
        for parent, child in skeleton.edges:
            mean, std = skeleton.get_bone_stats(parent, child)
            assert mean > 0, f"Edge ({parent},{child}) has non-positive mean"
            assert std > 0, f"Edge ({parent},{child}) has non-positive std"

    def test_bone_stats_symmetry_approximate(self):
        """Left/right symmetric bone pairs should have similar statistics."""
        symmetric_pairs = [
            ((0, 1), (0, 2)),     # nose->left_eye vs nose->right_eye
            ((5, 7), (6, 8)),     # shoulder->elbow
            ((7, 9), (8, 10)),    # elbow->wrist
            ((11, 13), (12, 14)), # hip->knee
            ((13, 15), (14, 16)), # knee->ankle
        ]
        for left_edge, right_edge in symmetric_pairs:
            m_l, s_l = COCO_BONE_LENGTH_STATS[left_edge]
            m_r, s_r = COCO_BONE_LENGTH_STATS[right_edge]
            assert abs(m_l - m_r) < 0.05, (
                f"Asymmetric means: {left_edge}={m_l:.4f} vs {right_edge}={m_r:.4f}"
            )


# ══════════════════════════════════════════════════════════════════════════
# 2. Pairwise potential tests
# ══════════════════════════════════════════════════════════════════════════

class TestPairwisePotential:
    """Validate that pairwise potentials penalise anatomically impossible bones."""

    def test_perfect_bone_length_gives_zero_penalty(self, decoder: MRFDecoder):
        """When d == bone_mean, the log-potential should be 0."""
        # shoulder(5)->elbow(7): mean ~0.413, std ~0.137
        area = 10000.0  # sqrt = 100
        scale = math.sqrt(area)
        mean_ratio, _ = COCO_BONE_LENGTH_STATS[(5, 7)]
        perfect_distance = mean_ratio * scale

        parent_pos = np.array([50.0, 50.0])
        child_pos = parent_pos + np.array([perfect_distance, 0.0])

        score = decoder._compute_pairwise_score(parent_pos, child_pos, 5, 7, scale)
        assert abs(score) < 1e-6, f"Expected ~0 penalty, got {score}"

    def test_impossible_bone_length_is_penalised(self, decoder: MRFDecoder):
        """A bone 10x the expected length should receive a strong penalty."""
        area = 10000.0
        scale = math.sqrt(area)
        mean_ratio, _ = COCO_BONE_LENGTH_STATS[(5, 7)]

        parent_pos = np.array([50.0, 50.0])
        # Place child 10x the expected distance
        child_pos = parent_pos + np.array([mean_ratio * scale * 10.0, 0.0])

        score = decoder._compute_pairwise_score(parent_pos, child_pos, 5, 7, scale)
        assert score < -10.0, f"Expected strong penalty, got {score}"

    def test_closer_candidate_scores_better(self, decoder: MRFDecoder):
        """Between two candidates, the one with a more realistic bone length wins."""
        area = 10000.0
        scale = math.sqrt(area)
        mean_ratio, std_ratio = COCO_BONE_LENGTH_STATS[(5, 7)]
        expected_d = mean_ratio * scale

        parent_pos = np.array([50.0, 50.0])
        good_child = parent_pos + np.array([expected_d, 0.0])
        bad_child = parent_pos + np.array([expected_d * 5.0, 0.0])

        score_good = decoder._compute_pairwise_score(parent_pos, good_child, 5, 7, scale)
        score_bad = decoder._compute_pairwise_score(parent_pos, bad_child, 5, 7, scale)

        assert score_good > score_bad, (
            f"Good candidate ({score_good:.2f}) should score better than "
            f"bad candidate ({score_bad:.2f})"
        )


# ══════════════════════════════════════════════════════════════════════════
# 3. Belief Propagation decode tests
# ══════════════════════════════════════════════════════════════════════════

class TestMRFDecode:
    """Test the full Belief Propagation decoding on synthetic scenarios."""

    def test_swap_correction_elbow(self, decoder: MRFDecoder):
        """
        Scenario: left_elbow(7) has two candidates:
          - Candidate A (low weight): near left_shoulder(5) — anatomically correct
          - Candidate B (high weight): near right_shoulder(6) — SWAP!

        The MRF should pick candidate A because the bone-length constraint
        from left_shoulder favours it.
        """
        area = 10000.0
        scale = math.sqrt(area)

        # Fixed positions for anchors (unimodal, no ambiguity)
        # Place shoulders far apart so a swap incurs a massive pairwise penalty
        left_shoulder_pos = np.array([20.0, 50.0])
        right_shoulder_pos = np.array([80.0, 50.0])
        nose_pos = np.array([50.0, 30.0])

        # left_elbow: correct position near left_shoulder
        elbow_mean = COCO_BONE_LENGTH_STATS[(5, 7)][0]
        correct_elbow = left_shoulder_pos + np.array([0.0, elbow_mean * scale])
        # swapped: near right_shoulder
        swapped_elbow = right_shoulder_pos + np.array([0.0, elbow_mean * scale])

        # Build unary potentials
        unaries = {}
        # Nose (root, unimodal)
        unaries[0] = UnaryPotential(
            means=[nose_pos], covariances=[np.eye(2) * 2.0], weights=[1.0]
        )
        # Left shoulder (unimodal)
        unaries[5] = UnaryPotential(
            means=[left_shoulder_pos], covariances=[np.eye(2) * 2.0], weights=[1.0]
        )
        # Right shoulder (unimodal)
        unaries[6] = UnaryPotential(
            means=[right_shoulder_pos], covariances=[np.eye(2) * 2.0], weights=[1.0]
        )
        # Left elbow: BIMODAL with SWAP
        # Candidate B (swapped) has HIGHER weight (0.55 vs 0.45)
        unaries[7] = UnaryPotential(
            means=[correct_elbow, swapped_elbow],
            covariances=[np.eye(2) * 2.0, np.eye(2) * 2.0],
            weights=[0.45, 0.55],  # argmax would pick the wrong one!
        )

        result = decoder.decode(unaries, area)

        # The MRF should pick the correct elbow (candidate 0)
        chosen_elbow = result[7]
        dist_to_correct = np.linalg.norm(chosen_elbow - correct_elbow)
        dist_to_swapped = np.linalg.norm(chosen_elbow - swapped_elbow)

        assert dist_to_correct < dist_to_swapped, (
            f"MRF should pick the anatomically correct elbow. "
            f"dist_correct={dist_to_correct:.2f}, dist_swapped={dist_to_swapped:.2f}"
        )

    def test_unimodal_passthrough(self, decoder: MRFDecoder):
        """When all keypoints are unimodal, MRF should return their single candidate."""
        area = 10000.0

        unaries = {}
        positions = {}
        for k in range(NUM_COCO_KEYPOINTS):
            pos = np.array([float(k * 10), float(k * 5)])
            positions[k] = pos
            unaries[k] = UnaryPotential(
                means=[pos], covariances=[np.eye(2) * 2.0], weights=[1.0]
            )

        result = decoder.decode(unaries, area)

        for k in range(NUM_COCO_KEYPOINTS):
            if k in result:
                np.testing.assert_allclose(
                    result[k], positions[k], atol=1e-6,
                    err_msg=f"Keypoint {k} should be unchanged"
                )

    def test_consistent_pose_preserved(self, decoder: MRFDecoder):
        """An already-consistent pose should not be modified by the MRF."""
        area = 10000.0
        scale = math.sqrt(area)

        # Build a realistic pose with proper bone lengths
        pose = np.zeros((NUM_COCO_KEYPOINTS, 2))
        pose[0] = [50.0, 20.0]   # nose
        pose[1] = [45.0, 18.0]   # left_eye
        pose[2] = [55.0, 18.0]   # right_eye
        pose[3] = [40.0, 20.0]   # left_ear
        pose[4] = [60.0, 20.0]   # right_ear
        pose[5] = [35.0, 40.0]   # left_shoulder
        pose[6] = [65.0, 40.0]   # right_shoulder
        pose[7] = [30.0, 60.0]   # left_elbow
        pose[8] = [70.0, 60.0]   # right_elbow
        pose[9] = [25.0, 80.0]   # left_wrist
        pose[10] = [75.0, 80.0]  # right_wrist
        pose[11] = [40.0, 70.0]  # left_hip
        pose[12] = [60.0, 70.0]  # right_hip
        pose[13] = [38.0, 90.0]  # left_knee
        pose[14] = [62.0, 90.0]  # right_knee
        pose[15] = [36.0, 110.0] # left_ankle
        pose[16] = [64.0, 110.0] # right_ankle

        unaries = {}
        for k in range(NUM_COCO_KEYPOINTS):
            # All unimodal — should pass through unchanged
            unaries[k] = UnaryPotential(
                means=[pose[k].copy()],
                covariances=[np.eye(2) * 2.0],
                weights=[1.0],
            )

        result = decoder.decode(unaries, area)

        for k in range(NUM_COCO_KEYPOINTS):
            if k in result:
                np.testing.assert_allclose(
                    result[k], pose[k], atol=1e-6,
                    err_msg=f"Keypoint {k} should not be modified"
                )


# ══════════════════════════════════════════════════════════════════════════
# 4. decode_pose integration test
# ══════════════════════════════════════════════════════════════════════════

class TestDecodePose:
    """Test the high-level decode_pose API."""

    def test_decode_pose_with_swap(self, decoder: MRFDecoder):
        """
        Full integration: pass MixtureResult objects and verify swap correction.
        """
        area = 10000.0
        scale = math.sqrt(area)

        # Build a simple skeleton
        nose = np.array([50.0, 20.0])
        left_shoulder = np.array([20.0, 40.0])
        right_shoulder = np.array([80.0, 40.0])

        elbow_offset = COCO_BONE_LENGTH_STATS[(5, 7)][0] * scale
        correct_left_elbow = left_shoulder + np.array([0.0, elbow_offset])
        swapped_left_elbow = right_shoulder + np.array([0.0, elbow_offset])

        # Build MixtureResult list (17 keypoints)
        gmm_results = []
        current_coords = np.zeros((NUM_COCO_KEYPOINTS, 2), dtype=np.float32)

        for k in range(NUM_COCO_KEYPOINTS):
            if k == 0:
                pos = nose
            elif k == 5:
                pos = left_shoulder
            elif k == 6:
                pos = right_shoulder
            elif k == 7:
                # Bimodal: argmax picks swapped (weight 0.6 > 0.4)
                comp_correct = _make_component(correct_left_elbow, weight=0.4)
                comp_swapped = _make_component(swapped_left_elbow, weight=0.6)
                mr = _make_mixture_result(
                    mean=swapped_left_elbow,  # argmax would pick this!
                    components=[comp_correct, comp_swapped],
                    model_type="bimodal",
                )
                gmm_results.append(mr)
                current_coords[k] = swapped_left_elbow.astype(np.float32)
                continue
            else:
                pos = np.array([float(k * 5), float(k * 10)])

            comp = _make_component(pos, weight=1.0)
            mr = _make_mixture_result(
                mean=pos,
                components=[comp],
                model_type="unimodal",
            )
            gmm_results.append(mr)
            current_coords[k] = pos.astype(np.float32)

        refined = decoder.decode_pose(gmm_results, area, current_coords)

        # Left elbow should be corrected to the anatomically correct position
        dist_to_correct = np.linalg.norm(refined[7] - correct_left_elbow)
        dist_to_swapped = np.linalg.norm(refined[7] - swapped_left_elbow)

        assert dist_to_correct < dist_to_swapped, (
            f"decode_pose should correct the swap. "
            f"dist_correct={dist_to_correct:.2f}, dist_swapped={dist_to_swapped:.2f}"
        )

    def test_decode_pose_no_bimodal_returns_unchanged(self, decoder: MRFDecoder):
        """When all keypoints are unimodal, decode_pose should return coords unchanged."""
        area = 10000.0
        current_coords = np.random.rand(NUM_COCO_KEYPOINTS, 2).astype(np.float32) * 100

        gmm_results = []
        for k in range(NUM_COCO_KEYPOINTS):
            comp = _make_component(current_coords[k], weight=1.0)
            mr = _make_mixture_result(
                mean=current_coords[k],
                components=[comp],
                model_type="unimodal",
            )
            gmm_results.append(mr)

        refined = decoder.decode_pose(gmm_results, area, current_coords)

        # All unimodal -> no bimodal nodes -> MRF logs "skipping" and returns coords
        # But since we still add unimodal nodes to unary_potentials, the MRF will
        # run but should preserve the single candidate for each node
        for k in range(NUM_COCO_KEYPOINTS):
            np.testing.assert_allclose(
                refined[k], current_coords[k], atol=0.01,
                err_msg=f"Unimodal keypoint {k} should be unchanged"
            )

"""
COCO Skeleton Graph for Kinematic Constraint Decoding.

Defines the topology of the human skeleton as a tree graph (17 COCO
keypoints, 16 edges / bones) together with empirical bone-length
statistics extracted from COCO train2017 annotations.

The graph is used by :class:`~pose_uncertainty.core.mrf_decoder.MRFDecoder`
to resolve inter-keypoint ambiguities via Belief Propagation.

Bone-length statistics
----------------------
Each edge stores ``(mean_ratio, std_ratio)`` — the mean and standard
deviation of the Euclidean distance between the two endpoints normalised
by ``sqrt(bbox_area)``.  These were computed from 11 004 COCO train2017
annotations using ``scripts/extract_bone_stats.py``, considering only
keypoint pairs where **both** endpoints have visibility >= 2.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────
# COCO keypoint names (standard 17-point format)
# ──────────────────────────────────────────────────────────────────────────

COCO_KEYPOINT_NAMES: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

NUM_COCO_KEYPOINTS: int = 17

# ──────────────────────────────────────────────────────────────────────────
# Skeleton edges  (parent_idx, child_idx)
# ──────────────────────────────────────────────────────────────────────────

COCO_SKELETON_EDGES: List[Tuple[int, int]] = [
    # Head
    (0, 1),   # nose -> left_eye
    (0, 2),   # nose -> right_eye
    (1, 3),   # left_eye -> left_ear
    (2, 4),   # right_eye -> right_ear
    # Torso (nose serves as proxy root connecting to shoulders)
    (0, 5),   # nose -> left_shoulder
    (0, 6),   # nose -> right_shoulder
    (5, 11),  # left_shoulder -> left_hip
    (6, 12),  # right_shoulder -> right_hip
    # Arms
    (5, 7),   # left_shoulder -> left_elbow
    (7, 9),   # left_elbow -> left_wrist
    (6, 8),   # right_shoulder -> right_elbow
    (8, 10),  # right_elbow -> right_wrist
    # Legs
    (11, 13), # left_hip -> left_knee
    (13, 15), # left_knee -> left_ankle
    (12, 14), # right_hip -> right_knee
    (14, 16), # right_knee -> right_ankle
]

# ──────────────────────────────────────────────────────────────────────────
# Empirical bone-length statistics from COCO train2017
# Format: {(parent, child): (mean_ratio, std_ratio)}
# Normalised by sqrt(bbox_area).
# Extracted with scripts/extract_bone_stats.py.
# ──────────────────────────────────────────────────────────────────────────

COCO_BONE_LENGTH_STATS: Dict[Tuple[int, int], Tuple[float, float]] = {
    ( 0,  1): (0.086697, 0.041642),  # nose -> left_eye
    ( 0,  2): (0.086987, 0.040777),  # nose -> right_eye
    ( 1,  3): (0.142436, 0.070436),  # left_eye -> left_ear
    ( 2,  4): (0.142714, 0.070405),  # right_eye -> right_ear
    ( 0,  5): (0.375405, 0.154303),  # nose -> left_shoulder
    ( 0,  6): (0.379020, 0.153544),  # nose -> right_shoulder
    ( 5, 11): (0.693674, 0.156694),  # left_shoulder -> left_hip
    ( 6, 12): (0.690588, 0.157064),  # right_shoulder -> right_hip
    ( 5,  7): (0.402244, 0.135029),  # left_shoulder -> left_elbow
    ( 7,  9): (0.311928, 0.125481),  # left_elbow -> left_wrist
    ( 6,  8): (0.401648, 0.136314),  # right_shoulder -> right_elbow
    ( 8, 10): (0.315018, 0.128668),  # right_elbow -> right_wrist
    (11, 13): (0.499586, 0.139764),  # left_hip -> left_knee
    (13, 15): (0.514086, 0.133683),  # left_knee -> left_ankle
    (12, 14): (0.498670, 0.140628),  # right_hip -> right_knee
    (14, 16): (0.513950, 0.134331),  # right_knee -> right_ankle
}


# ──────────────────────────────────────────────────────────────────────────
# SkeletonGraph class
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class SkeletonGraph:
    """Tree-structured graph of the human skeleton.

    The graph is a rooted tree (root = nose, index 0) with 17 nodes and
    16 directed edges.  It exposes helpers for tree traversal required by
    the Belief Propagation algorithm.

    Parameters
    ----------
    edges : list of (int, int)
        Directed edges ``(parent, child)``.
    bone_stats : dict
        ``{(parent, child): (mean_ratio, std_ratio)}``
    root : int
        Index of the root node (default 0 = nose).
    """

    edges: List[Tuple[int, int]] = field(default_factory=lambda: list(COCO_SKELETON_EDGES))
    bone_stats: Dict[Tuple[int, int], Tuple[float, float]] = field(
        default_factory=lambda: dict(COCO_BONE_LENGTH_STATS)
    )
    root: int = 0
    num_nodes: int = NUM_COCO_KEYPOINTS

    # Computed in __post_init__
    _children: Dict[int, List[int]] = field(default_factory=dict, init=False, repr=False)
    _parent: Dict[int, Optional[int]] = field(default_factory=dict, init=False, repr=False)
    _topo_order: List[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Build adjacency structures and topological order."""
        self._children = defaultdict(list)
        self._parent = {i: None for i in range(self.num_nodes)}

        for parent, child in self.edges:
            self._children[parent].append(child)
            self._parent[child] = parent

        # BFS from root -> topological order (parents before children)
        self._topo_order = []
        visited = set()
        queue = deque([self.root])
        visited.add(self.root)

        while queue:
            node = queue.popleft()
            self._topo_order.append(node)
            for child in self._children[node]:
                if child not in visited:
                    visited.add(child)
                    queue.append(child)

        # Any disconnected nodes get appended at the end
        for i in range(self.num_nodes):
            if i not in visited:
                self._topo_order.append(i)

    def children(self, node: int) -> List[int]:
        """Return children of *node* in the tree."""
        return self._children.get(node, [])

    def parent(self, node: int) -> Optional[int]:
        """Return parent of *node*, or ``None`` if root."""
        return self._parent.get(node)

    def tree_order(self) -> List[int]:
        """Return nodes in BFS order (root first = parents before children)."""
        return list(self._topo_order)

    def leaves_to_root_order(self) -> List[int]:
        """Return nodes from leaves to root (reverse BFS)."""
        return list(reversed(self._topo_order))

    def get_bone_stats(self, parent: int, child: int) -> Tuple[float, float]:
        """Return ``(mean_ratio, std_ratio)`` for the given edge.

        Falls back to ``(0.3, 0.15)`` if edge not found.
        """
        return self.bone_stats.get((parent, child), (0.3, 0.15))

    def is_connected(self, a: int, b: int) -> bool:
        """Check if nodes *a* and *b* are directly connected by an edge."""
        return (a, b) in self.bone_stats or (b, a) in self.bone_stats


# ──────────────────────────────────────────────────────────────────────────
# Module-level singleton for the default COCO skeleton
# ──────────────────────────────────────────────────────────────────────────

COCO_SKELETON = SkeletonGraph()

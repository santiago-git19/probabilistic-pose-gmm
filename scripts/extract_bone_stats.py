"""
Extract empirical bone length statistics from COCO train2017 annotations.

For each bone (edge) in the COCO skeleton, computes the mean and standard
deviation of the Euclidean distance between connected keypoints, normalised
by sqrt(bbox_area).  Only keypoint pairs where **both** endpoints are visible
(visibility >= 2) are included.

Output: a Python dict literal printed to stdout, suitable for copy-pasting
into skeleton.py, plus a summary table.

Usage:
    python scripts/extract_bone_stats.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ── COCO skeleton topology (parent -> child) ──────────────────────────────
# 17 COCO keypoints:
#  0  nose
#  1  left_eye          2  right_eye
#  3  left_ear           4  right_ear
#  5  left_shoulder      6  right_shoulder
#  7  left_elbow         8  right_elbow
#  9  left_wrist        10  right_wrist
# 11  left_hip          12  right_hip
# 13  left_knee         14  right_knee
# 15  left_ankle        16  right_ankle

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Each edge is (parent_idx, child_idx)
SKELETON_EDGES: List[Tuple[int, int]] = [
    # Head
    (0, 1),   # nose -> left_eye
    (0, 2),   # nose -> right_eye
    (1, 3),   # left_eye -> left_ear
    (2, 4),   # right_eye -> right_ear
    # Torso
    (0, 5),   # nose -> left_shoulder   (proxy for neck->shoulder)
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


def main() -> None:
    ann_path = Path("data/coco/annotations/person_keypoints_train2017.json")
    if not ann_path.exists():
        print(f"ERROR: Annotation file not found at {ann_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading annotations from {ann_path} …")
    with open(ann_path, "r") as f:
        coco = json.load(f)

    annotations = coco["annotations"]
    print(f"  Total annotations: {len(annotations)}")

    # Accumulate normalised bone lengths per edge
    bone_lengths: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    skipped_no_area = 0
    skipped_no_kps = 0

    for ann in annotations:
        # Need valid area
        area = ann.get("area", 0)
        if area <= 0:
            skipped_no_area += 1
            continue

        kps = ann.get("keypoints", [])
        if len(kps) < 17 * 3:
            skipped_no_kps += 1
            continue

        # Parse keypoints: [x1, y1, v1, x2, y2, v2, ...]
        coords = np.array(kps, dtype=np.float64).reshape(17, 3)
        xs = coords[:, 0]
        ys = coords[:, 1]
        vis = coords[:, 2]  # 0=not labelled, 1=labelled-occluded, 2=visible

        scale = math.sqrt(area)

        for parent, child in SKELETON_EDGES:
            # Only use pairs where both keypoints are labelled AND visible
            if vis[parent] >= 2 and vis[child] >= 2:
                dx = xs[parent] - xs[child]
                dy = ys[parent] - ys[child]
                d = math.sqrt(dx * dx + dy * dy)
                bone_lengths[(parent, child)].append(d / scale)

    print(f"  Skipped (no area): {skipped_no_area}")
    print(f"  Skipped (no kps):  {skipped_no_kps}")
    print()

    # ── Compute stats ─────────────────────────────────────────────────────
    print("=" * 80)
    print(f"{'Edge':<40s}  {'N':>7s}  {'mean':>8s}  {'std':>8s}  {'median':>8s}")
    print("-" * 80)

    stats_dict_lines: List[str] = []

    for parent, child in SKELETON_EDGES:
        samples = bone_lengths[(parent, child)]
        n = len(samples)
        if n < 10:
            print(f"  ({parent:2d}, {child:2d})  {COCO_KEYPOINT_NAMES[parent]:>16s} -> "
                  f"{COCO_KEYPOINT_NAMES[child]:<16s}  WARNING: only {n} samples")
            mean_val, std_val = 0.0, 1.0
        else:
            arr = np.array(samples)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            median_val = float(np.median(arr))
            pname = COCO_KEYPOINT_NAMES[parent]
            cname = COCO_KEYPOINT_NAMES[child]
            label = f"({parent:2d}, {child:2d})  {pname:>16s} -> {cname:<16s}"
            print(f"  {label}  {n:7d}  {mean_val:8.4f}  {std_val:8.4f}  {median_val:8.4f}")

        # Format for Python dict
        stats_dict_lines.append(
            f"    ({parent:2d}, {child:2d}): ({mean_val:.6f}, {std_val:.6f}),"
            f"  # {COCO_KEYPOINT_NAMES[parent]} -> {COCO_KEYPOINT_NAMES[child]}"
        )

    print("=" * 80)
    print()

    # ── Print copy-pasteable dict ─────────────────────────────────────────
    print("# Copy this into skeleton.py as COCO_BONE_LENGTH_STATS")
    print("# Format: {(parent_idx, child_idx): (mean_ratio, std_ratio)}")
    print("COCO_BONE_LENGTH_STATS = {")
    for line in stats_dict_lines:
        print(line)
    print("}")


if __name__ == "__main__":
    main()

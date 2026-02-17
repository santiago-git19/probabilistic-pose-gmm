import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pose_uncertainty.evaluation.runner import _gt_to_arrays
from pose_uncertainty.models.base import COCO_KEYPOINT_NAMES
from pose_uncertainty.utils.types import ImageSample, Keypoint


def test_gt_to_arrays_aligns_crowdpose_by_name_to_coco17() -> None:
    sample = ImageSample(
        image_id=1,
        image_path="dummy.jpg",
        image_array=np.zeros((32, 32, 3), dtype=np.uint8),
        bbox=(1.0, 2.0, 10.0, 20.0),
        dataset_source="crowdpose",
        ground_truth_keypoints=[
            Keypoint(id=0, x=10.0, y=11.0, confidence=1.0, name="left_shoulder"),
            Keypoint(id=1, x=12.0, y=13.0, confidence=0.5, name="right_shoulder"),
            Keypoint(id=2, x=14.0, y=15.0, confidence=1.0, name="left_wrist"),
            Keypoint(id=3, x=16.0, y=17.0, confidence=0.0, name="right_wrist"),
            Keypoint(id=12, x=18.0, y=19.0, confidence=1.0, name="head"),
            Keypoint(id=13, x=20.0, y=21.0, confidence=1.0, name="neck"),
        ],
    )

    coords, vis, area = _gt_to_arrays(
        sample,
        target_keypoint_names=COCO_KEYPOINT_NAMES,
        target_num_keypoints=len(COCO_KEYPOINT_NAMES),
    )

    assert coords.shape == (17, 2)
    assert vis.shape == (17,)
    assert area == 200.0

    # Mapped body joints
    np.testing.assert_array_equal(coords[5], np.array([10.0, 11.0], dtype=np.float32))
    np.testing.assert_array_equal(coords[6], np.array([12.0, 13.0], dtype=np.float32))
    np.testing.assert_array_equal(coords[9], np.array([14.0, 15.0], dtype=np.float32))
    np.testing.assert_array_equal(coords[10], np.array([16.0, 17.0], dtype=np.float32))

    assert vis[5] == 2
    assert vis[6] == 1
    assert vis[9] == 2
    assert vis[10] == 0

    # Unmapped crowdpose-only joints must stay invisible in COCO-17 space
    assert vis[0] == 0  # nose
    np.testing.assert_array_equal(coords[0], np.array([0.0, 0.0], dtype=np.float32))


def test_gt_to_arrays_keeps_coco_names_consistent() -> None:
    sample = ImageSample(
        image_id=2,
        image_path="dummy.jpg",
        image_array=np.zeros((16, 16, 3), dtype=np.uint8),
        bbox=(0.0, 0.0, 5.0, 6.0),
        dataset_source="coco",
        ground_truth_keypoints=[
            Keypoint(id=0, x=1.0, y=2.0, confidence=1.0, name="nose"),
            Keypoint(id=1, x=3.0, y=4.0, confidence=0.5, name="left_eye"),
            Keypoint(id=2, x=5.0, y=6.0, confidence=0.0, name="right_eye"),
        ],
    )

    coords, vis, area = _gt_to_arrays(
        sample,
        target_keypoint_names=COCO_KEYPOINT_NAMES,
        target_num_keypoints=len(COCO_KEYPOINT_NAMES),
    )

    assert area == 30.0
    np.testing.assert_array_equal(coords[0], np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(coords[1], np.array([3.0, 4.0], dtype=np.float32))
    np.testing.assert_array_equal(coords[2], np.array([5.0, 6.0], dtype=np.float32))

    assert vis[0] == 2
    assert vis[1] == 1
    assert vis[2] == 0

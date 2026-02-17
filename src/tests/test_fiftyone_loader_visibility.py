"""Tests for visibility/tag filtering semantics in fiftyone_loader.py."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pose_uncertainty.visualization import fiftyone_loader as loader


class _DummyDynamic(dict):
    def __init__(self) -> None:
        super().__init__()
        self.tags = []


class _DummyKeypoint(_DummyDynamic):
    def __init__(self, points, label=None, tags=None, **kwargs):
        super().__init__()
        self.points = points
        self.label = label
        self.tags = list(tags) if tags is not None else []
        for k, v in kwargs.items():
            setattr(self, k, v)


class _DummyKeypoints:
    def __init__(self, keypoints):
        self.keypoints = keypoints


class _DummyPolyline(_DummyDynamic):
    def __init__(self, points, closed, filled, label=None, tags=None, **kwargs):
        super().__init__()
        self.points = points
        self.closed = closed
        self.filled = filled
        self.label = label
        self.tags = list(tags) if tags is not None else []
        for k, v in kwargs.items():
            setattr(self, k, v)


class _DummyPolylines:
    def __init__(self, polylines):
        self.polylines = polylines


class _DummyFO:
    Keypoint = _DummyKeypoint
    Keypoints = _DummyKeypoints
    Polyline = _DummyPolyline
    Polylines = _DummyPolylines


def test_normalise_visibility_code_accepts_only_coco_codes():
    assert loader._normalise_visibility_code(0) == 0
    assert loader._normalise_visibility_code(1) == 1
    assert loader._normalise_visibility_code(2) == 2

    assert loader._normalise_visibility_code(99) == 2
    assert loader._normalise_visibility_code(-1) == 2
    assert loader._normalise_visibility_code(None) == 2
    assert loader._normalise_visibility_code("foo") == 2


def test_visibility_payload_occluded_has_backwards_compatible_aliases():
    code, label, tags = loader._visibility_payload(1)
    assert code == 1
    assert label == "occluded"
    assert "occluded" in tags
    assert "occlusion" in tags


def test_to_fo_keypoints_labeled_uses_gt_visibility_over_coords_column():
    coords = np.array(
        [
            [10.0, 20.0, 2.0],
            [30.0, 40.0, 2.0],
        ],
        dtype=np.float32,
    )
    gt = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    out = loader._to_fo_keypoints_labeled(coords, img_w=100, img_h=100, fo=_DummyFO, gt_coords_with_vis=gt)
    kp0, kp1 = out.keypoints

    assert kp0.label == "not_labeled"
    assert kp0["keypoint_name"] == "Nose"
    assert kp0["visibility_code"] == 0
    assert kp0["visibility"] == "not_labeled"
    assert "not_labeled" in kp0.tags
    assert "unlabeled" in kp0.tags

    assert kp1.label == "occluded"
    assert kp1["keypoint_name"] == "L_Eye"
    assert kp1["visibility_code"] == 1
    assert kp1["visibility"] == "occluded"
    assert "occluded" in kp1.tags
    assert "occlusion" in kp1.tags


def test_to_fo_keypoints_labeled_invalid_visibility_falls_back_to_visible():
    coords = np.array([[10.0, 20.0, 10.0]], dtype=np.float32)

    out = loader._to_fo_keypoints_labeled(coords, img_w=100, img_h=100, fo=_DummyFO)
    kp = out.keypoints[0]

    assert kp.label == "visible"
    assert kp["visibility_code"] == 2
    assert kp["visibility"] == "visible"
    assert "visible" in kp.tags


def test_ellipses_per_kp_labeled_exposes_visibility_fields_and_tags():
    gmm_per_kp = [
        {
            "means": np.array([[20.0, 20.0]], dtype=np.float64),
            "covariances": np.array([[[3.0, 0.0], [0.0, 2.0]]], dtype=np.float64),
        },
        {
            "means": np.array([[40.0, 40.0]], dtype=np.float64),
            "covariances": np.array([[[2.5, 0.2], [0.2, 1.5]]], dtype=np.float64),
        },
    ]
    gt = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    out = loader._ellipses_per_kp_labeled(
        gmm_per_kp=gmm_per_kp,
        img_w=100,
        img_h=100,
        packet={},
        fo=_DummyFO,
        gt_coords_with_vis=gt,
    )

    assert out is not None
    assert len(out.polylines) == 2

    p0, p1 = out.polylines

    assert p0["visibility_code"] == 0
    assert p0["visibility"] == "not_labeled"
    assert "not_labeled" in p0.tags
    assert "unlabeled" in p0.tags

    assert p1["visibility_code"] == 1
    assert p1["visibility"] == "occluded"
    assert "occluded" in p1.tags
    assert "occlusion" in p1.tags


def test_fiftyone_filter_labels_filters_keypoints_by_visibility_tag():
    fo = pytest.importorskip("fiftyone")
    F = fo.ViewField

    ds = fo.Dataset()
    try:
        gt = np.array(
            [
                [10.0, 10.0, 2.0],
                [20.0, 20.0, 1.0],
            ],
            dtype=np.float32,
        )
        keypoints = loader._to_fo_keypoints_labeled(
            coords=gt,
            img_w=100,
            img_h=100,
            fo=fo,
            gt_coords_with_vis=gt,
        )

        sample = fo.Sample(filepath=str(Path(__file__).resolve()))
        sample["ground_truth"] = keypoints
        ds.add_sample(sample)

        filtered = ds.filter_labels("ground_truth", F("tags").contains("occluded"))
        out = filtered.first()["ground_truth"].keypoints

        assert len(out) == 1
        assert out[0]["visibility"] == "occluded"
        assert "occluded" in out[0].tags
    finally:
        ds.delete()


def test_fiftyone_filter_labels_filters_keypoints_by_visibility_label():
    fo = pytest.importorskip("fiftyone")
    F = fo.ViewField

    ds = fo.Dataset()
    try:
        gt = np.array(
            [
                [10.0, 10.0, 2.0],
                [20.0, 20.0, 1.0],
            ],
            dtype=np.float32,
        )
        keypoints = loader._to_fo_keypoints_labeled(
            coords=gt,
            img_w=100,
            img_h=100,
            fo=fo,
            gt_coords_with_vis=gt,
        )

        sample = fo.Sample(filepath=str(Path(__file__).resolve()))
        sample["ground_truth"] = keypoints
        ds.add_sample(sample)

        filtered = ds.filter_labels("ground_truth", F("label") == "occluded")
        out = filtered.first()["ground_truth"].keypoints

        assert len(out) == 1
        assert out[0].label == "occluded"
        assert out[0]["visibility"] == "occluded"
    finally:
        ds.delete()

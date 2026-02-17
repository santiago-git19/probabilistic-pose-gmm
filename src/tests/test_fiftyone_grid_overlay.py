"""Quick smoke test for _image_coords_to_grid_keypoints."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# Mock fiftyone
class MockKP:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.tags = []
    def __setitem__(self, k, v):
        self.__dict__[k] = v

class MockFO:
    @staticmethod
    def Keypoint(**kw): return MockKP(**kw)

from pose_uncertainty.visualization.fiftyone_loader import _image_coords_to_grid_keypoints

metadata = {
    'input_center': [320.0, 240.0],
    'input_scale': [200.0, 300.0],
    'input_size': [192, 256],
}
hm_avg = np.zeros((17, 64, 48), dtype=np.float32)
rng = np.random.default_rng(42)
gt_coords = np.column_stack([
    rng.uniform(250, 390, 17),
    rng.uniform(120, 360, 17),
    np.ones(17) * 2,
]).astype(np.float32)

packet = {
    'ground_truth': {'coords': gt_coords},
    'predictions': {'baseline': {'coords': gt_coords + rng.standard_normal((17, 3)).astype(np.float32) * 5}},
    'aggregation': {'heatmap_avg': hm_avg, 'mmpose_metadata': metadata},
}

rects = [
    {'x0': 50.0 + (k % 4) * 200, 'y0': 50.0 + (k // 4) * 200, 'w': 180.0, 'h': 180.0}
    for k in range(17)
]
fo = MockFO()

# Test GT
result_gt = _image_coords_to_grid_keypoints(
    packet, 'ground_truth', 'coords',
    n_kp=17, hm_w=48, hm_h=64, rects=rects, grid_w=850, grid_h=1050, fo=fo,
)
print(f"GT keypoints on grid: {len(result_gt)}")
assert len(result_gt) == 17, f"Expected 17, got {len(result_gt)}"

# Test baseline
result_base = _image_coords_to_grid_keypoints(
    packet, 'predictions', 'baseline.coords',
    n_kp=17, hm_w=48, hm_h=64, rects=rects, grid_w=850, grid_h=1050, fo=fo,
)
print(f"Baseline keypoints on grid: {len(result_base)}")
assert len(result_base) == 17, f"Expected 17, got {len(result_base)}"

# Verify all points have [0,1] normalized coords
for i, kp in enumerate(result_gt):
    px, py = kp.points[0]
    assert 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0, f"KP {i} out of range: ({px}, {py})"

# Test missing metadata fallback
bad_packet = {
    'ground_truth': {'coords': gt_coords},
    'aggregation': {'heatmap_avg': hm_avg},  # no mmpose_metadata
}
result_bad = _image_coords_to_grid_keypoints(
    bad_packet, 'ground_truth', 'coords',
    n_kp=17, hm_w=48, hm_h=64, rects=rects, grid_w=850, grid_h=1050, fo=fo,
)
assert len(result_bad) == 0, "Should return empty list when metadata is missing"
print("Missing metadata fallback: OK")

print("\nALL TESTS PASSED")

"""Smoke test for the refactored grouped FiftyOne loader."""
import sys
import logging
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)

from src.pose_uncertainty.evaluation.storage import load_deep_analysis
from src.pose_uncertainty.visualization.fiftyone_loader import (
    _packet_to_samples,
    _CACHE_SUBDIR,
)
import fiftyone as fo

DATA_DIR = Path("outputs/2026-02-11/12-40-42")


def main():
    cache_dir = DATA_DIR / _CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load first packet
    pkl = sorted(DATA_DIR.glob("*.pkl.gz"))[0]
    log.info("Loading %s ...", pkl.name)
    packet = load_deep_analysis(str(pkl))

    # ----- inspect shapes -----
    agg = packet.get("aggregation", {})
    hm = agg.get("heatmap_avg")
    sp = agg.get("sampling_points")
    gmm = packet.get("gmm_model", {})
    tta = packet.get("tta_data", [])

    print(f"heatmap_avg   shape: {hm.shape if hm is not None else None}")
    print(f"sampling_pts  shape: {sp.shape if sp is not None else None}")
    m = gmm.get("means")
    c = gmm.get("covariances")
    print(f"gmm means     shape: {m.shape if m is not None else None}")
    print(f"gmm covs      shape: {c.shape if c is not None else None}")
    print(f"gmm n_comp          : {gmm.get('n_components')}")
    print(f"tta_data entries    : {len(tta)}")
    for i, e in enumerate(tta):
        hm_e = e.get("heatmap")
        img_e = e.get("image")
        print(
            f"  tta[{i}] name={e.get('name')}, "
            f"image={img_e.shape if img_e is not None else None}, "
            f"hm={hm_e.shape if hm_e is not None else None}"
        )

    # ----- build samples -----
    t0 = time.time()
    samples = _packet_to_samples(packet, cache_dir, fo)
    elapsed = time.time() - t0
    print(f"\nBuilt {len(samples)} samples in {elapsed:.1f}s")

    for s in samples:
        grp = s["group"]
        fname = Path(s.filepath).name
        print(f"  slice={grp.name:30s}  file={fname}")
        overlay_names = [
            "mc_samples",
            "gmm_ellipses",
            "ground_truth",
            "prediction_base",
            "prediction_ours",
            "uncertainty_ellipses",
        ]
        found = [f for f in overlay_names if s.has_field(f)]
        if found:
            print(f"    overlays: {found}")

    print("\nDONE")


if __name__ == "__main__":
    main()

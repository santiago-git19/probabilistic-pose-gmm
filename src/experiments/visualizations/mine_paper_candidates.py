"""
Paper Candidate Miner: Scan benchmarks and deep profiling packets to find top qualitative examples.

Categorizes packets into:
1. Clear Wins (Large Delta OKS)
2. Swap Resolutions (Baseline swapped L/R, Ours fixed it)
3. Multimodal / Ambiguity Cases (GMM has multiple significant components)
4. Occlusion Uncertainty (Occluded joints with calibrated high variance)
5. Single-Joint Jumps (Individual keypoint massive improvements)
6. Limitations / Regressions (For honest ablation discussions)

Usage:
    # Scan default/latest outputs
    poetry run python src/experiments/visualizations/mine_paper_candidates.py

    # Scan specific output directory
    poetry run python src/experiments/visualizations/mine_paper_candidates.py --data-dir "outputs/2026-07-06/12-38-24"

    # Filter and launch FiftyOne with only the selected candidates
    poetry run python src/experiments/visualizations/mine_paper_candidates.py --data-dir "outputs/2026-07-06/12-38-24" --launch-fiftyone
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COCO_KP_NAMES = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine qualitative candidates for paper figures.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to folder containing .pkl.gz analysis packets (defaults to latest in outputs/)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top candidates to display per category (default: 5)",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="paper_candidates.json",
        help="Path to save candidate metadata JSON (default: paper_candidates.json)",
    )
    parser.add_argument(
        "--launch-fiftyone",
        action="store_true",
        help="If set, creates and launches a FiftyOne dataset containing only the mined candidates",
    )
    return parser.parse_args()


def find_latest_outputs_dir() -> Optional[Path]:
    """Auto-detect the most recent directory with .pkl.gz packets."""
    outputs_root = Path("outputs")
    if not outputs_root.exists():
        return None

    # Search for all subdirectories containing .pkl.gz files
    all_packet_dirs = []
    for p in outputs_root.rglob("*.pkl.gz"):
        parent = p.parent
        if parent not in all_packet_dirs:
            all_packet_dirs.append(parent)

    if not all_packet_dirs:
        return None

    # Sort by modification time or folder name
    all_packet_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return all_packet_dirs[0]


def analyze_packet(pkl_path: Path) -> Optional[Dict[str, Any]]:
    """Load a packet and extract key diagnostic features for mining."""
    try:
        from src.pose_uncertainty.evaluation.storage import load_deep_analysis
        packet = load_deep_analysis(str(pkl_path))
    except Exception as e:
        logger.warning("Failed to load %s: %s", pkl_path.name, e)
        return None

    meta = packet.get("meta", {})
    metrics = packet.get("metrics", {})
    gt = packet.get("ground_truth", {})
    gmm_per_kp = packet.get("gmm_per_kp", [])

    image_id = meta.get("image_id", 0)
    sample_type = meta.get("sample_type", "unknown")
    oks_base = float(metrics.get("oks_base", 0.0))
    oks_ours = float(metrics.get("oks_ours", 0.0))
    delta_oks = float(metrics.get("delta_oks", oks_ours - oks_base))
    entropy = float(metrics.get("entropy", 0.0))
    cov_vol = float(metrics.get("covariance_volume", 0.0))
    swaps_base = int(metrics.get("swaps_base", 0))
    swaps_ours = int(metrics.get("swaps_ours", 0))
    swaps_corrected = int(metrics.get("swaps_corrected", 0))

    per_kp_ours = metrics.get("per_kp_oks_ours")
    per_kp_base = metrics.get("per_kp_oks_base")
    per_kp_delta = []
    max_single_kp_jump = 0.0
    max_single_kp_name = ""

    if per_kp_ours is not None and per_kp_base is not None:
        p_ours = np.asarray(per_kp_ours, dtype=np.float32)
        p_base = np.asarray(per_kp_base, dtype=np.float32)
        n_kp = min(len(p_ours), len(p_base), len(COCO_KP_NAMES))
        for k in range(n_kp):
            o_val = p_ours[k]
            b_val = p_base[k]
            if not np.isnan(o_val) and not np.isnan(b_val):
                d = float(o_val - b_val)
                per_kp_delta.append((COCO_KP_NAMES[k], d, float(b_val), float(o_val)))
                if d > max_single_kp_jump:
                    max_single_kp_jump = d
                    max_single_kp_name = COCO_KP_NAMES[k]

    # Multimodal detection: inspect GMM per keypoint
    bimodal_keypoints = []
    for k, kp_info in enumerate(gmm_per_kp):
        if not isinstance(kp_info, dict):
            continue
        weights = np.asarray(kp_info.get("weights", []), dtype=np.float64).reshape(-1)
        means = np.asarray(kp_info.get("means", []), dtype=np.float64)
        if len(weights) >= 2 and len(means) >= 2:
            # Check if second component has at least 20% weight
            sorted_w = np.sort(weights)[::-1]
            if sorted_w[1] >= 0.20:
                dist = float(np.linalg.norm(means[0] - means[1]))
                if dist >= 3.0:  # At least 3 heatmap pixels apart
                    kp_name = COCO_KP_NAMES[k] if k < len(COCO_KP_NAMES) else f"KP_{k}"
                    bimodal_keypoints.append({
                        "keypoint": kp_name,
                        "w1": float(sorted_w[0]),
                        "w2": float(sorted_w[1]),
                        "mode_dist": dist,
                    })

    # Occlusion analysis
    gt_coords = gt.get("coords")
    occluded_kps = []
    if isinstance(gt_coords, np.ndarray) and gt_coords.shape[1] >= 3:
        for k in range(min(len(gt_coords), len(COCO_KP_NAMES))):
            vis = int(gt_coords[k, 2])
            if vis == 1:  # Occluded
                occluded_kps.append(COCO_KP_NAMES[k])

    return {
        "file_path": str(pkl_path.resolve()),
        "file_name": pkl_path.name,
        "image_id": image_id,
        "sample_type": sample_type,
        "oks_base": oks_base,
        "oks_ours": oks_ours,
        "delta_oks": delta_oks,
        "entropy": entropy,
        "cov_vol": cov_vol,
        "swaps_base": swaps_base,
        "swaps_ours": swaps_ours,
        "swaps_corrected": swaps_corrected,
        "max_single_kp_jump": max_single_kp_jump,
        "max_single_kp_name": max_single_kp_name,
        "bimodal_keypoints": bimodal_keypoints,
        "num_bimodal_kps": len(bimodal_keypoints),
        "occluded_kps": occluded_kps,
        "num_occluded_kps": len(occluded_kps),
    }


def print_table(title: str, items: List[Dict[str, Any]], columns: List[Tuple[str, str, int]]):
    """Print formatted ASCII table."""
    print("\n" + "=" * 90)
    print(f" {title.upper()} (Top {len(items)})")
    print("=" * 90)

    header = " | ".join(f"{col_name:<{width}}" for col_name, _, width in columns)
    print(header)
    print("-" * len(header))

    for item in items:
        row = []
        for _, key, width in columns:
            val = item.get(key, "")
            if isinstance(val, float):
                val_str = f"{val:.4f}"
            elif isinstance(val, list):
                val_str = ", ".join(str(x) for x in val[:2])
                if len(val) > 2: val_str += "..."
            else:
                val_str = str(val)
            row.append(f"{val_str:<{width}}")
        print(" | ".join(row))


def main() -> None:
    args = parse_args()

    if args.data_dir:
        data_path = Path(args.data_dir)
    else:
        data_path = find_latest_outputs_dir()
        if not data_path:
            logger.error("No outputs directory found. Please specify --data-dir <path>")
            return

    if not data_path.exists():
        logger.error("Directory not found: %s", data_path)
        return

    pkl_files = list(data_path.glob("*.pkl.gz"))
    if not pkl_files:
        logger.error("No .pkl.gz packets found in %s", data_path)
        return

    logger.info("Scanning %d packets in: %s", len(pkl_files), data_path)

    records: List[Dict[str, Any]] = []
    for p in pkl_files:
        rec = analyze_packet(p)
        if rec:
            records.append(rec)

    if not records:
        logger.error("No valid analysis packets could be processed.")
        return

    # Categorize candidates
    # 1. Clear Wins (Highest Delta OKS with good final OKS)
    wins = sorted([r for r in records if r["oks_ours"] >= 0.70], key=lambda x: x["delta_oks"], reverse=True)

    # 2. Swap Resolutions
    swaps = sorted([r for r in records if r["swaps_corrected"] > 0 or (r["swaps_base"] > 0 and r["swaps_ours"] == 0)],
                   key=lambda x: (x["swaps_corrected"], x["delta_oks"]), reverse=True)

    # 3. Multimodal / Bimodal Ambiguities
    bimodal = sorted([r for r in records if r["num_bimodal_kps"] > 0],
                     key=lambda x: (x["num_bimodal_kps"], x["entropy"]), reverse=True)

    # 4. Occlusion Uncertainty (Occluded joints + High Covariance / Entropy)
    occluded = sorted([r for r in records if r["num_occluded_kps"] > 0 and r["delta_oks"] > 0],
                      key=lambda x: (x["num_occluded_kps"], x["cov_vol"]), reverse=True)

    # 5. Single Joint Dramatic Rescues
    single_kp = sorted(records, key=lambda x: x["max_single_kp_jump"], reverse=True)

    # 6. Regressions (for honesty / limitation analysis)
    regressions = sorted(records, key=lambda x: x["delta_oks"])

    # Display Top Tables
    cols_wins = [
        ("File", "file_name", 28),
        ("ID", "image_id", 6),
        ("OKS Base", "oks_base", 9),
        ("OKS Ours", "oks_ours", 9),
        ("Delta OKS", "delta_oks", 9),
        ("Best Joint Rescue", "max_single_kp_name", 18),
    ]
    print_table("1. Top Overall Wins (Major Performance Improvements)", wins[:args.top_k], cols_wins)

    cols_swaps = [
        ("File", "file_name", 28),
        ("ID", "image_id", 6),
        ("Swaps Base", "swaps_base", 11),
        ("Swaps Ours", "swaps_ours", 11),
        ("Corrected", "swaps_corrected", 10),
        ("Delta OKS", "delta_oks", 8),
    ]
    print_table("2. Top Swap Resolutions (Symmetry/Limb Ambiguity Fixed)", swaps[:args.top_k], cols_swaps)

    cols_bimodal = [
        ("File", "file_name", 28),
        ("ID", "image_id", 6),
        ("Bimodal Joints", "num_bimodal_kps", 15),
        ("Entropy", "entropy", 9),
        ("Cov Vol", "cov_vol", 9),
        ("Delta OKS", "delta_oks", 8),
    ]
    print_table("3. Multimodal Mixture Cases (GMM Captures Multiple Modes)", bimodal[:args.top_k], cols_bimodal)

    cols_occ = [
        ("File", "file_name", 28),
        ("ID", "image_id", 6),
        ("Occluded Joints", "num_occluded_kps", 16),
        ("Cov Vol", "cov_vol", 9),
        ("Entropy", "entropy", 9),
        ("Delta OKS", "delta_oks", 8),
    ]
    print_table("4. Occlusion Uncertainty (Model Detects High Uncertainty on Hidden Joints)", occluded[:args.top_k], cols_occ)

    cols_single = [
        ("File", "file_name", 28),
        ("ID", "image_id", 6),
        ("Joint", "max_single_kp_name", 12),
        ("Joint dOKS", "max_single_kp_jump", 12),
        ("Overall dOKS", "delta_oks", 12),
    ]
    print_table("5. Single-Joint Massive Rescues", single_kp[:args.top_k], cols_single)

    # Save to JSON
    selected_dict = {
        "data_dir": str(data_path.resolve()),
        "top_wins": wins[:args.top_k],
        "top_swaps": swaps[:args.top_k],
        "top_bimodal": bimodal[:args.top_k],
        "top_occluded": occluded[:args.top_k],
        "top_single_joint_rescues": single_kp[:args.top_k],
        "top_regressions": regressions[:args.top_k],
    }

    out_json_path = Path(args.out_json)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(selected_dict, f, indent=2)
    logger.info("Saved mined candidate metadata to: %s", out_json_path.resolve())

    # Optional FiftyOne Launcher with filtered candidates only
    if args.launch_fiftyone:
        # Collect unique packet paths
        chosen_paths = set()
        for cat_list in [wins[:args.top_k], swaps[:args.top_k], bimodal[:args.top_k], occluded[:args.top_k], single_kp[:args.top_k]]:
            for item in cat_list:
                chosen_paths.add(Path(item["file_path"]))

        logger.info("\nLaunching FiftyOne with ONLY %d curated candidate packets...", len(chosen_paths))
        import fiftyone as fo
        from src.pose_uncertainty.evaluation.storage import load_deep_analysis
        from src.pose_uncertainty.visualization.fiftyone_loader import _packet_to_samples

        dataset_name = "Paper_Curated_Candidates"
        if fo.dataset_exists(dataset_name):
            fo.delete_dataset(dataset_name)

        ds = fo.Dataset(name=dataset_name)
        ds.persistent = False
        ds.add_group_field("group", default="original")

        cache_dir = data_path / "fiftyone_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        for p_path in chosen_paths:
            try:
                pkt = load_deep_analysis(str(p_path))
                samples = _packet_to_samples(pkt, cache_dir, fo)
                if samples:
                    ds.add_samples(samples)
            except Exception as e:
                logger.warning("Failed loading %s: %s", p_path.name, e)

        session = fo.launch_app(ds, address="0.0.0.0", remote=True)
        print("\n" + "=" * 70)
        print(f"[OK] FiftyOne Curated Dataset '{dataset_name}' active on http://localhost:5151")
        print("Press Ctrl+C to close FiftyOne server.")
        print("=" * 70)
        session.wait()


if __name__ == "__main__":
    main()

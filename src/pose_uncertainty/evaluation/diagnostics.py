"""
Diagnostics – select focus groups from mass-evaluation results.

After :meth:`EvaluationRunner.run_mass_evaluation` produces
``results_metadata.parquet``, this module analyses the scalar metrics to
identify the most *interesting* images for a Deep Profiling pass.

Four disjoint focus groups are built (priority order prevents duplicates):

1. **Wins**: our GMM method clearly beats the baseline
   (``oks_ours > oks_base + 0.1``).
2. **Regressions**: the baseline is clearly better
   (``oks_base > oks_ours + 0.1``).
3. **High Uncertainty**: covariance volume in the top 10 %
   (``covariance_vol > quantile(0.90)``).
4. **Edge Cases**: both methods struggle
   (``oks_base < 0.5  AND  oks_ours < 0.5``).

Additional per-keypoint outlier analysis:

5. **Top_Single_KP_Gains**: images where any single keypoint had the highest
   positive delta_oks improvement.
6. **Worst_Single_KP_Drops**: images where any single keypoint had the worst
   negative delta_oks drop.

Usage
-----
>>> from pose_uncertainty.evaluation.diagnostics import select_focus_groups, select_keypoint_outliers
>>> groups = select_focus_groups("outputs/.../results_metadata.parquet", cfg)
>>> # groups = {"Wins": [42, 87, ...], "Regressions": [...], ...}
>>> 
>>> kp_outliers = select_keypoint_outliers("outputs/.../results_metadata.parquet")
>>> # kp_outliers = {"Top_Single_KP_Gains": [...], "Worst_Single_KP_Drops": [...]}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------

_DEFAULT_N_SAMPLES: int = 50
_OKS_IMPROVEMENT_THRESHOLD: float = 0.1
_OKS_FAILURE_THRESHOLD: float = 0.5
_COV_VOL_QUANTILE: float = 0.90


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_focus_groups(
    parquet_path: str,
    config: Dict[str, Any],
) -> Dict[str, List[int]]:
    """Analyse mass-evaluation results and return focus-group image IDs.

    Parameters
    ----------
    parquet_path : str
        Path to ``results_metadata.parquet`` (produced by
        ``EvaluationRunner.run_mass_evaluation``).
    config : dict
        Hydra configuration (or plain dict).  Reads
        ``config["evaluation"]["n_samples_per_group"]``; falls back to 50.

    Returns
    -------
    dict[str, list[int]]
        Mapping ``group_name → list_of_image_ids``.  Groups are
        ``"Wins"``, ``"Regressions"``, ``"High_Uncertainty"``,
        ``"Edge_Cases"``.

    Side-effects
    ------------
    Saves a JSON copy at ``<parquet_dir>/focus_groups_ids.json``.
    """
    # ---- load data --------------------------------------------------------
    pq = Path(parquet_path)
    if not pq.exists():
        raise FileNotFoundError(f"Parquet file not found: {pq}")

    df = pd.read_parquet(pq)
    logger.info("Loaded %d rows from %s", len(df), pq.name)

    # ---- read config ------------------------------------------------------
    eval_cfg = config.get("evaluation", {}) if isinstance(config, dict) else {}
    n_samples: int = int(eval_cfg.get("n_samples_per_group", _DEFAULT_N_SAMPLES))

    # ---- build groups (priority order, no duplicates) ---------------------
    used_ids: set = set()
    groups: Dict[str, List[int]] = {}

    # 1. Wins
    wins_mask = df["oks_ours"] > (df["oks_base"] + _OKS_IMPROVEMENT_THRESHOLD)
    groups["Wins"] = _sample_ids(df, wins_mask, used_ids, n_samples)

    # 2. Regressions
    reg_mask = df["oks_base"] > (df["oks_ours"] + _OKS_IMPROVEMENT_THRESHOLD)
    groups["Regressions"] = _sample_ids(df, reg_mask, used_ids, n_samples)

    # 3. High Uncertainty
    cov_col = "covariance_vol"
    if cov_col in df.columns:
        q90 = df[cov_col].quantile(_COV_VOL_QUANTILE)
        hu_mask = df[cov_col] > q90
    else:
        logger.warning("Column '%s' not found; skipping High_Uncertainty group.", cov_col)
        hu_mask = pd.Series(False, index=df.index)
    groups["High_Uncertainty"] = _sample_ids(df, hu_mask, used_ids, n_samples)

    # 4. Edge Cases
    ec_mask = (df["oks_base"] < _OKS_FAILURE_THRESHOLD) & (
        df["oks_ours"] < _OKS_FAILURE_THRESHOLD
    )
    groups["Edge_Cases"] = _sample_ids(df, ec_mask, used_ids, n_samples)

    # 5. Per-keypoint outliers (Top gains and worst drops) -----------------
    logger.info("Analyzing per-keypoint outliers...")
    kp_outliers = select_keypoint_outliers(str(pq), n_samples=n_samples)
    
    # Merge keypoint outliers into main groups dict
    for kp_group_name, kp_ids in kp_outliers.items():
        # Filter out any IDs already used in previous groups
        new_ids = [img_id for img_id in kp_ids if img_id not in used_ids]
        groups[kp_group_name] = new_ids
        used_ids.update(new_ids)

    # ---- summary ----------------------------------------------------------
    for grp, ids in groups.items():
        logger.info("  %-20s -> %d images", grp, len(ids))

    # ---- persist to JSON for traceability ---------------------------------
    output_dir = pq.parent
    json_path = output_dir / "focus_groups_ids.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(groups, fh, indent=2)
    logger.info("Saved focus group IDs to %s", json_path)

    return groups


def select_keypoint_outliers(
    parquet_path: str,
    n_samples: int = 20,
) -> Dict[str, List[int]]:
    """Select images with extreme per-keypoint delta_oks values.

    Analyses per-keypoint delta_oks columns (e.g., delta_oks_nose,
    delta_oks_left_knee) to find images where a single keypoint had
    an exceptionally good or bad improvement.

    Parameters
    ----------
    parquet_path : str
        Path to ``results_metadata.parquet`` containing per-keypoint columns.
    n_samples : int, optional
        Number of images to select per group (default: 20).

    Returns
    -------
    dict[str, list[int]]
        Mapping with two groups:
        - ``"Top_Single_KP_Gains"``: images with highest positive delta_oks
          for any single keypoint.
        - ``"Worst_Single_KP_Drops"``: images with lowest negative delta_oks
          for any single keypoint.

    Side-effects
    ------------
    Saves a JSON copy at ``<parquet_dir>/keypoint_outliers_ids.json``.
    """
    # ---- load data --------------------------------------------------------
    pq = Path(parquet_path)
    if not pq.exists():
        raise FileNotFoundError(f"Parquet file not found: {pq}")

    df = pd.read_parquet(pq)
    logger.info("Loaded %d rows from %s for keypoint outlier analysis", len(df), pq.name)

    # ---- find delta_oks_* columns -----------------------------------------
    delta_cols = [col for col in df.columns if col.startswith("delta_oks_") and col != "delta_oks"]
    if not delta_cols:
        logger.warning("No per-keypoint delta_oks columns found in DataFrame")
        return {"Top_Single_KP_Gains": [], "Worst_Single_KP_Drops": []}

    logger.info("Found %d per-keypoint delta_oks columns", len(delta_cols))

    # ---- compute max/min across all keypoints for each image --------------
    # For each row, find the maximum delta_oks value across all keypoints
    df_deltas = df[delta_cols].copy()
    
    # Filter rows that have at least one non-NaN value
    valid_rows_mask = df_deltas.notna().any(axis=1)
    
    if not valid_rows_mask.any():
        logger.warning("No valid delta_oks values found in DataFrame")
        return {"Top_Single_KP_Gains": [], "Worst_Single_KP_Drops": []}
    
    df["max_kp_gain"] = pd.to_numeric(df_deltas.max(axis=1, skipna=True), errors="coerce").astype(float)
    df["min_kp_drop"] = pd.to_numeric(df_deltas.min(axis=1, skipna=True), errors="coerce").astype(float)

    # Track which keypoint caused the max/min (only for valid rows)
    df["best_kp"] = None
    df["worst_kp"] = None
    df.loc[valid_rows_mask, "best_kp"] = df_deltas[valid_rows_mask].idxmax(axis=1, skipna=True)
    df.loc[valid_rows_mask, "worst_kp"] = df_deltas[valid_rows_mask].idxmin(axis=1, skipna=True)

    # ---- select top gains -------------------------------------------------
    # Sort by max_kp_gain descending, take top n_samples (exclude rows with all NaN)
    valid_gains = df[valid_rows_mask & df["max_kp_gain"].notna()]
    if len(valid_gains) == 0:
        logger.warning("No valid gains found")
        gain_ids = []
        top_gains = df.head(0)  # Empty DataFrame
    else:
        top_gains = valid_gains.nlargest(min(n_samples, len(valid_gains)), "max_kp_gain")
        gain_ids = top_gains["image_id"].tolist()

    logger.info("Top_Single_KP_Gains: selected %d images", len(gain_ids))
    for idx, row in top_gains.head(5).iterrows():
        kp_name = str(row["best_kp"]).replace("delta_oks_", "") if pd.notna(row["best_kp"]) else "unknown"
        logger.debug(
            "  image_id=%d, max_gain=%.4f at %s",
            row["image_id"], row["max_kp_gain"], kp_name
        )

    # ---- select worst drops -----------------------------------------------
    # Sort by min_kp_drop ascending (most negative first), take top n_samples (exclude rows with all NaN)
    valid_drops = df[valid_rows_mask & df["min_kp_drop"].notna()]
    if len(valid_drops) == 0:
        logger.warning("No valid drops found")
        drop_ids = []
        worst_drops = df.head(0)  # Empty DataFrame
    else:
        worst_drops = valid_drops.nsmallest(min(n_samples, len(valid_drops)), "min_kp_drop")
        drop_ids = worst_drops["image_id"].tolist()

    logger.info("Worst_Single_KP_Drops: selected %d images", len(drop_ids))
    for idx, row in worst_drops.head(5).iterrows():
        kp_name = str(row["worst_kp"]).replace("delta_oks_", "") if pd.notna(row["worst_kp"]) else "unknown"
        logger.debug(
            "  image_id=%d, min_drop=%.4f at %s",
            row["image_id"], row["min_kp_drop"], kp_name
        )

    # ---- build result dictionary ------------------------------------------
    groups = {
        "Top_Single_KP_Gains": [int(i) for i in gain_ids],
        "Worst_Single_KP_Drops": [int(i) for i in drop_ids],
    }

    # ---- persist to JSON for traceability ---------------------------------
    output_dir = pq.parent
    json_path = output_dir / "keypoint_outliers_ids.json"

    # Include metadata about which keypoint caused the outlier
    metadata = {
        "Top_Single_KP_Gains": [
            {
                "image_id": int(row["image_id"]),
                "max_gain": float(row["max_kp_gain"]),
                "keypoint": str(row["best_kp"]).replace("delta_oks_", "") if pd.notna(row["best_kp"]) else "unknown",
            }
            for _, row in top_gains.iterrows()
        ] if len(top_gains) > 0 else [],
        "Worst_Single_KP_Drops": [
            {
                "image_id": int(row["image_id"]),
                "min_drop": float(row["min_kp_drop"]),
                "keypoint": str(row["worst_kp"]).replace("delta_oks_", "") if pd.notna(row["worst_kp"]) else "unknown",
            }
            for _, row in worst_drops.iterrows()
        ] if len(worst_drops) > 0 else [],
    }

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Saved keypoint outlier metadata to %s", json_path)

    return groups


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_ids(
    df: pd.DataFrame,
    mask: pd.Series,
    used: set,
    n: int,
) -> List[int]:
    """Return up to *n* image IDs matching *mask*, excluding already *used*."""
    candidates = df.loc[mask & ~df["image_id"].isin(used), "image_id"]
    if len(candidates) > n:
        candidates = candidates.sample(n=n, random_state=42)
    selected = candidates.tolist()
    used.update(selected)
    return [int(i) for i in selected]

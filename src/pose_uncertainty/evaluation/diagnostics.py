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

Usage
-----
>>> from pose_uncertainty.evaluation.diagnostics import select_focus_groups
>>> groups = select_focus_groups("outputs/.../results_metadata.parquet", cfg)
>>> # groups = {"Wins": [42, 87, ...], "Regressions": [...], ...}
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

    # ---- summary ----------------------------------------------------------
    for grp, ids in groups.items():
        logger.info("  %-20s → %d images", grp, len(ids))

    # ---- persist to JSON for traceability ---------------------------------
    output_dir = pq.parent
    json_path = output_dir / "focus_groups_ids.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(groups, fh, indent=2)
    logger.info("Saved focus group IDs to %s", json_path)

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

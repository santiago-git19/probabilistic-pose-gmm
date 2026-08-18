"""Retroactive Upload of Historical Benchmark Runs to Weights & Biases.

Scans ``outputs/`` searching for folders containing ``results_metadata.parquet``:

1. Reads ``.hydra/config.yaml`` (if present) to reconstruct run configuration.
2. Generates a descriptive run name from key hyperparameter variables
   (model architecture, TTA flip, photometric augmentations, sampling temperature).
3. Computes aggregate scalar metrics on the Parquet dataframe (OKS, NLL, Entropy, etc.).
4. Creates a run in W&B via ``wandb.init`` / ``wandb.log`` / ``wandb.finish``.

Usage::

    cd <project_root>
    poetry run python src/experiments/tracking/upload_past_runs_to_wandb.py

    # Dry-run inspection without creating W&B runs:
    poetry run python src/experiments/tracking/upload_past_runs_to_wandb.py --dry-run

    # Filter runs after a specific date:
    poetry run python src/experiments/tracking/upload_past_runs_to_wandb.py --after 2026-02-15
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ===================================================================
# 1.  Run Discovery
# ===================================================================

def discover_runs(
    root: Path,
    *,
    after: Optional[datetime.date] = None,
) -> List[Path]:
    """Recursively discover folders containing ``results_metadata.parquet``.

    Parameters
    ----------
    root : Path
        Search root directory (typically ``outputs/``).
    after : date, optional
        Only return runs whose date folder is >= *after*.

    Returns
    -------
    list[Path]
        Chronologically sorted run directories.
    """
    runs: List[Path] = []
    for parquet in sorted(root.rglob("results_metadata.parquet")):
        run_dir = parquet.parent

        # Date filter (folder has format YYYY-MM-DD)
        if after is not None:
            try:
                date_part = _extract_date_from_path(run_dir)
                if date_part and date_part < after:
                    continue
            except ValueError:
                pass

        runs.append(run_dir)

    log.info("Discovered %d folders with parquet in %s", len(runs), root)
    return runs


def _extract_date_from_path(p: Path) -> Optional[datetime.date]:
    """Extract first path component matching ``YYYY-MM-DD``."""
    for part in p.relative_to(OUTPUTS_DIR).parts:
        try:
            return datetime.date.fromisoformat(part)
        except ValueError:
            continue
    return None


# ===================================================================
# 2.  Config Loader
# ===================================================================

def load_run_config(run_dir: Path) -> Dict[str, Any]:
    """Read ``.hydra/config.yaml`` and return as native dictionary.

    If absent, returns a minimal fallback dictionary.
    Hydra interpolations (``${…}``) are stripped to prevent errors when
    uploading to W&B.
    """
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        log.warning("  No .hydra/config.yaml found in %s - partial config.", run_dir.name)
        return {"_source": str(run_dir), "_hydra_config_missing": True}

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Clean Hydra interpolations that W&B cannot resolve
    return _strip_interpolations(raw) if raw else {}


def _strip_interpolations(obj: Any) -> Any:
    """Replace ``${…}`` with literal string representation."""
    if isinstance(obj, dict):
        return {k: _strip_interpolations(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_interpolations(v) for v in obj]
    if isinstance(obj, str) and "${" in obj:
        return f"[unresolved] {obj}"
    return obj


# ===================================================================
# 3.  Smart Name Generation
# ===================================================================

def generate_run_name(run_dir: Path, cfg: Dict[str, Any]) -> str:
    """Generate descriptive run name from config and path.

    Format: ``<model>_<tta_flip>_<photo_augs>_T<temp>_<date>_<time>``

    Examples:
        ``hrnet_w32_Flip_Bright+Noise_T0.3_0211_1240``
        ``hrnet_w32_NoFlip_NoPhoto_T0.1_0209_1537``
    """
    parts: List[str] = []

    # ---- Model ---------------------------------------------------------
    model_name = _deep_get(cfg, "model.name", "unknown_model")
    parts.append(str(model_name))

    # ---- TTA - Flip ----------------------------------------------------
    flip_enabled = _deep_get(cfg, "flip.enabled", False)
    flip_indices = _deep_get(cfg, "flip.flip_indices", None)
    if flip_enabled:
        parts.append("Flip" if flip_indices else "FlipNoIdx")
    else:
        parts.append("NoFlip")

    # ---- TTA - Photometric augmentations --------------------------------
    photo_augs: List[str] = []
    _PHOTO_KEYS = {
        "brightness": "Bright",
        "contrast": "Contr",
        "noise": "Noise",
        "blur": "Blur",
    }
    for key, short in _PHOTO_KEYS.items():
        if _deep_get(cfg, f"photometric.{key}.enabled", False):
            val = _photo_value(cfg, key)
            photo_augs.append(f"{short}{val}" if val else short)

    parts.append("+".join(photo_augs) if photo_augs else "NoPhoto")

    # ---- Sampling temperature ------------------------------------------
    temp = _deep_get(cfg, "sampling.temperature", None)
    if temp is not None:
        parts.append(f"T{temp}")

    # ---- Num samples ---------------------------------------------------
    n_samples = _deep_get(cfg, "sampling.num_samples", None)
    if n_samples is not None:
        parts.append(f"S{n_samples}")

    # ---- Model selection criterion (AIC vs BIC) -------------------------
    aic_w = _deep_get(cfg, "aic_weight", 0)
    bic_w = _deep_get(cfg, "bic_weight", 0)
    if aic_w and not bic_w:
        parts.append("AIC")
    elif bic_w and not aic_w:
        parts.append("BIC")
    elif aic_w and bic_w:
        parts.append(f"AIC{aic_w}+BIC{bic_w}")

    # ---- Folder timestamp ----------------------------------------------
    rel = run_dir.relative_to(OUTPUTS_DIR)
    rel_parts = rel.parts

    if rel_parts[0] == "comparisons":
        return f"{model_name}_{rel_parts[-1]}" if model_name != "unknown_model" else rel_parts[-1]

    if len(rel_parts) >= 2:
        date_str = rel_parts[0].replace("2026-", "").replace("-", "")
        time_str = rel_parts[1].replace("-", "")[:4]
        parts.append(f"{date_str}_{time_str}")
    elif len(rel_parts) == 1:
        parts.append(rel_parts[0])

    return "_".join(parts)


def _deep_get(d: Dict, dotted_key: str, default: Any = None) -> Any:
    """Safe lookup ``d['a']['b']['c']`` with dotted key ``'a.b.c'``."""
    keys = dotted_key.split(".")
    node = d
    for k in keys:
        if isinstance(node, dict):
            node = node.get(k)
        else:
            return default
        if node is None:
            return default
    return node


def _photo_value(cfg: Dict, key: str) -> str:
    """Return concise numeric suffix for a photometric augmentation."""
    if key == "brightness":
        v = _deep_get(cfg, "photometric.brightness.delta")
        return str(int(v)) if v is not None else ""
    if key == "contrast":
        v = _deep_get(cfg, "photometric.contrast.range")
        if isinstance(v, list) and len(v) == 2:
            return f"{v[0]}-{v[1]}"
        return ""
    if key == "noise":
        v = _deep_get(cfg, "photometric.noise.sigma")
        return str(int(v)) if v is not None else ""
    if key == "blur":
        v = _deep_get(cfg, "photometric.blur.kernel_size")
        return str(int(v)) if v is not None else ""
    return ""


# ===================================================================
# 4.  Metric Computation
# ===================================================================

METRIC_COLUMNS = [
    "oks_base", "oks_ours", "delta_oks",
    "nll", "entropy", "covariance_vol", "n_components",
]


def compute_aggregate_metrics(run_dir: Path) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Read parquet file and compute global aggregate metrics.

    Returns
    -------
    df : pd.DataFrame
        Raw dataframe.
    metrics : dict
        Scalar metrics dictionary ready for ``wandb.log``.
    """
    parquet_path = run_dir / "results_metadata.parquet"
    df = pd.read_parquet(parquet_path)

    metrics: Dict[str, float] = {"n_images": len(df)}

    for col in METRIC_COLUMNS:
        if col in df.columns:
            metrics[f"{col}_mean"] = float(df[col].mean())
            metrics[f"{col}_std"] = float(df[col].std())
            metrics[f"{col}_median"] = float(df[col].median())
        else:
            metrics[f"{col}_mean"] = float("nan")

    # Percentage of wins (delta_oks > 0) and regressions (delta_oks < 0)
    if "delta_oks" in df.columns:
        metrics["pct_wins"] = float((df["delta_oks"] > 0).mean() * 100)
        metrics["pct_regressions"] = float((df["delta_oks"] < 0).mean() * 100)
        metrics["pct_ties"] = float((df["delta_oks"] == 0).mean() * 100)

    return df, metrics


# ===================================================================
# 5.  W&B Upload
# ===================================================================

def upload_run(
    run_dir: Path,
    cfg: Dict[str, Any],
    name: str,
    metrics: Dict[str, float],
    *,
    dry_run: bool = False,
    project: str = "tfg-pose-estimation",
    tags: Optional[List[str]] = None,
) -> None:
    """Create W&B run, log config and metrics, and terminate run."""
    if dry_run:
        log.info("  [DRY-RUN] Name: %s", name)
        log.info("  [DRY-RUN] Metrics: %s", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()})
        return

    import wandb

    run = None
    try:
        auto_tags = _auto_tags(cfg)
        if tags:
            auto_tags.extend(tags)

        run = wandb.init(
            project=project,
            name=name,
            config=cfg,
            tags=auto_tags,
            job_type="historical_upload",
            notes=f"Retroactive upload from {run_dir.relative_to(PROJECT_ROOT)}",
            reinit=True,
        )

        wandb.log(metrics)
        log.info("  [OK] Run '%s' uploaded successfully.", name)

    except Exception:
        log.exception("  [FAIL] Error uploading run '%s'", name)

    finally:
        if run is not None:
            try:
                wandb.finish()
            except Exception:
                log.warning("  wandb.finish() failed for '%s'", name)


def _auto_tags(cfg: Dict[str, Any]) -> List[str]:
    """Generate automated metadata tags from configuration."""
    tags: List[str] = ["historical"]

    # Dataset
    ds = _deep_get(cfg, "dataset.name")
    if ds:
        tags.append(str(ds))

    # Model
    model = _deep_get(cfg, "model.name")
    if model:
        tags.append(str(model))

    # TTA
    if _deep_get(cfg, "flip.enabled", False):
        tags.append("tta_flip")
    if any(_deep_get(cfg, f"photometric.{k}.enabled", False)
           for k in ("brightness", "contrast", "noise", "blur")):
        tags.append("tta_photometric")

    return tags


# ===================================================================
# 6.  CLI Orchestration
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload historical benchmark runs from outputs/ to Weights & Biases.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display planned actions without creating W&B runs.",
    )
    parser.add_argument(
        "--after",
        type=str,
        default=None,
        help="Filter runs with date >= YYYY-MM-DD (e.g., 2026-02-15).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="tfg-pose-estimation",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        default=None,
        help="Additional tags for all runs.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=str(OUTPUTS_DIR),
        help="Outputs root directory (default: outputs/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outputs = Path(args.outputs_dir)
    after_date: Optional[datetime.date] = None
    if args.after:
        after_date = datetime.date.fromisoformat(args.after)

    log.info("=" * 70)
    log.info("  UPLOAD PAST RUNS TO WEIGHTS & BIASES")
    log.info("  Directory: %s", outputs)
    if after_date:
        log.info("  Date filter: >= %s", after_date)
    if args.dry_run:
        log.info("  *** DRY-RUN MODE (no runs will be created) ***")
    log.info("=" * 70)

    # ---- Discover runs -------------------------------------------------
    runs = discover_runs(outputs, after=after_date)
    if not runs:
        log.warning("No runs with results_metadata.parquet found.")
        return

    # ---- Process each run ----------------------------------------------
    ok_count = 0
    fail_count = 0

    for idx, run_dir in enumerate(runs, start=1):
        rel = run_dir.relative_to(PROJECT_ROOT)
        log.info("")
        log.info("[%d/%d] %s", idx, len(runs), rel)

        try:
            # 1) Config
            cfg = load_run_config(run_dir)

            # 2) Smart Name
            name = generate_run_name(run_dir, cfg)
            log.info("  Name: %s", name)

            # 3) Metrics
            _df, metrics = compute_aggregate_metrics(run_dir)
            log.info(
                "  n=%d | OKS_ours=%.4f | ΔOKS=%.4f | Entropy=%.4f",
                int(metrics["n_images"]),
                metrics.get("oks_ours_mean", float("nan")),
                metrics.get("delta_oks_mean", float("nan")),
                metrics.get("entropy_mean", float("nan")),
            )

            # 4) Upload
            upload_run(
                run_dir,
                cfg,
                name,
                metrics,
                dry_run=args.dry_run,
                project=args.project,
                tags=args.tags,
            )
            ok_count += 1

        except Exception:
            log.exception("  [FAIL] Error processing %s", rel)
            fail_count += 1

    # ---- Summary -------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info("  SUMMARY: %d OK  |  %d FAIL  |  %d TOTAL", ok_count, fail_count, len(runs))
    log.info("=" * 70)


if __name__ == "__main__":
    main()

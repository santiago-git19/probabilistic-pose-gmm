"""
Weights & Biases (wandb) integration utilities.

Provides a **context-manager** ``wandb_run`` that safely handles
``wandb.init`` / ``wandb.finish`` lifecycle, plus helpers for logging
scalar metrics and summary tables.

Design decisions
----------------
* **Graceful degradation**: if ``wandb`` is not installed or the user has
  disabled tracking (``wandb.enabled: false`` in Hydra config), every
  function becomes a no-op.  This guarantees the local pipeline never
  breaks because of wandb issues.
* **Hydra-safe config**: ``OmegaConf.DictConfig`` is converted to a
  plain Python dict before passing to ``wandb.init``.
* The module never imports ``wandb`` at the top level so the rest of the
  code can run without the dependency installed.

Usage in experiment scripts::

    from src.pose_uncertainty.tracking import wandb_run, log_metrics

    with wandb_run(cfg, name="Baseline", tags=["tta"]):
        df = runner.run_mass_evaluation()
        log_metrics({...})
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, List, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal flag – set on first import attempt
# ---------------------------------------------------------------------------
_WANDB_AVAILABLE: Optional[bool] = None


def _check_wandb() -> bool:
    """Lazy-check whether ``wandb`` is importable.  Caches the result."""
    global _WANDB_AVAILABLE
    if _WANDB_AVAILABLE is None:
        try:
            import wandb as _  # noqa: F401
            _WANDB_AVAILABLE = True
        except ImportError:
            _WANDB_AVAILABLE = False
            logger.warning(
                "wandb is not installed – experiment tracking disabled.  "
                "Install with:  pip install wandb  or  poetry add wandb"
            )
    return _WANDB_AVAILABLE


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _resolve_config(cfg: DictConfig) -> Dict[str, Any]:
    """Convert a Hydra ``DictConfig`` to a plain dict safe for wandb.

    Uses ``OmegaConf.to_container(resolve=True)`` so interpolations like
    ``${paths.project_root}`` are expanded.  Keys that fail to resolve
    (e.g. missing interpolations) are silently dropped.
    """
    try:
        return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)  # type: ignore[return-value]
    except Exception:
        # Último recurso: serializar a YAML y volver a parsear
        import yaml
        try:
            return yaml.safe_load(OmegaConf.to_yaml(cfg, resolve=True))
        except Exception:
            logger.warning("Could not fully resolve Hydra config for wandb; logging partial config.")
            return {"_raw_yaml": OmegaConf.to_yaml(cfg, resolve=False)}


def _wandb_enabled(cfg: DictConfig) -> bool:
    """Return whether wandb tracking is enabled according to the config."""
    return bool(OmegaConf.select(cfg, "wandb.enabled", default=True))


# ---------------------------------------------------------------------------
# Context manager: safe run lifecycle
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def wandb_run(
    cfg: DictConfig,
    *,
    name: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    group: Optional[str] = None,
    notes: Optional[str] = None,
    job_type: Optional[str] = None,
):
    """Context manager that wraps a single ``wandb`` *run*.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config – will be serialized and logged as the run config.
    name : str, optional
        Human-readable run name (e.g. experiment name).
    tags : sequence of str, optional
        Tags for filtering runs in the W&B dashboard.
    group : str, optional
        Group runs that belong to the same logical experiment.
    notes : str, optional
        Free-form description shown in the dashboard.
    job_type : str, optional
        Type of job (e.g. ``"evaluation"``, ``"benchmark"``).

    Yields
    ------
    ``wandb.Run | None``
        The active run object, or *None* if wandb is unavailable /
        disabled.

    Notes
    -----
    ``wandb.finish()`` is **always** called in the ``finally`` block,
    even if the wrapped code raises an exception.

    Example::

        with wandb_run(cfg, name="Baseline", tags=["coco", "tta"]) as run:
            df = runner.run_mass_evaluation()
            log_metrics({"oks_mean": df["oks_ours"].mean()})
    """
    if not _check_wandb() or not _wandb_enabled(cfg):
        logger.info("wandb tracking disabled – skipping run initialization.")
        yield None
        return

    import wandb  # noqa: E402 – lazy import

    # ---- Resolve config safely ----------------------------------------
    resolved_config = _resolve_config(cfg)

    # ---- Project / entity from config or defaults ---------------------
    project = OmegaConf.select(cfg, "wandb.project", default="tfg-pose-estimation")
    entity = OmegaConf.select(cfg, "wandb.entity", default=None)

    run = None
    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=list(tags) if tags else None,
            group=group,
            notes=notes,
            job_type=job_type,
            config=resolved_config,
            reinit=True,  # allows multiple runs in the same process
        )
        logger.info("wandb run initialized: %s (project=%s)", run.name, project)
        yield run

    except Exception:
        logger.exception("wandb run failed – finishing run gracefully.")
        raise

    finally:
        # Guarantee clean teardown of wandb run
        if run is not None:
            try:
                wandb.finish()
                logger.info("wandb run finished: %s", run.name)
            except Exception:
                logger.warning("wandb.finish() raised an exception.", exc_info=True)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_metrics(
    metrics: Dict[str, Any],
    *,
    step: Optional[int] = None,
    prefix: Optional[str] = None,
) -> None:
    """Log a flat dictionary of scalar metrics to the active wandb run.

    Parameters
    ----------
    metrics : dict
        Keys are metric names, values are numbers (int / float).
    step : int, optional
        Global step (if iterating).
    prefix : str, optional
        If given, each key is prefixed: ``f"{prefix}/{key}"``.
    """
    if not _check_wandb():
        return

    import wandb

    if wandb.run is None:
        logger.debug("log_metrics called but no active wandb run – skipping.")
        return

    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    wandb.log(metrics, step=step)


def log_summary_table(
    key: str,
    dataframe: "pd.DataFrame",  # type: ignore[name-defined]  # noqa: F821
) -> None:
    """Log a pandas DataFrame as a ``wandb.Table`` in the run summary.

    Parameters
    ----------
    key : str
        Name of the artefact in the W&B dashboard.
    dataframe : pd.DataFrame
        The data to upload.
    """
    if not _check_wandb():
        return

    import wandb

    if wandb.run is None:
        logger.debug("log_summary_table called but no active wandb run – skipping.")
        return

    try:
        table = wandb.Table(dataframe=dataframe)
        wandb.log({key: table})
        logger.info("Logged wandb.Table '%s' (%d rows)", key, len(dataframe))
    except Exception:
        logger.warning("Failed to log wandb table '%s'.", key, exc_info=True)

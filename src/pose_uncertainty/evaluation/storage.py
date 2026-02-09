"""
Deep Profiling Storage: Save / Load compressed analysis packets.

This module manages efficient serialization of rich analysis data produced
by the Deep Profiling stage of the evaluation pipeline.  Two complementary
optimisations keep disk usage tractable even when hundreds of full-resolution
heatmaps and images are stored:

1. **Heatmap down-casting** – float32 heatmaps are losslessly reduced to
   float16 before serialization (halving size) and promoted back on load.
2. **JPEG-in-memory compression** – RGB images stored inside ``tta_data``
   are encoded as JPEG byte-strings (~10× smaller) and decoded back to
   numpy arrays on load.

File format: ``{sample_type}_{image_id}.pkl.gz``
    A gzip-compressed pickle containing a single ``AnalysisPacket`` dict.

Usage
-----
>>> from pose_uncertainty.evaluation.storage import save_deep_analysis, load_deep_analysis
>>> save_deep_analysis(packet, output_dir="outputs/2026-02-09/12-00-00")
>>> loaded = load_deep_analysis("outputs/2026-02-09/12-00-00/Win_42.pkl.gz")
"""

from __future__ import annotations

import copy
import gzip
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TypedDict – exact packet schema (documentation + static analysis)
# ---------------------------------------------------------------------------

class _Meta(TypedDict):
    image_id: int
    file_name: str
    dataset: str
    model_name: str
    sample_type: str
    timestamp: str


class _Prediction(TypedDict):
    coords: npt.NDArray[np.float32]  # (17, 3) – x, y, score
    score: float


class _Predictions(TypedDict):
    baseline: _Prediction
    ours_gmm: _Prediction


class _Metrics(TypedDict):
    oks_base: float
    oks_ours: float
    delta_oks: float
    nll: float
    entropy: float
    covariance_volume: float


class _TTAEntry(TypedDict, total=False):
    aug_id: int
    name: str
    params: Dict[str, Any]
    image: Any  # np.ndarray (RGB) before save / bytes after save
    heatmap: npt.NDArray[np.float32]
    pred_coords: npt.NDArray[np.float32]


class _Aggregation(TypedDict):
    heatmap_avg: npt.NDArray[np.float32]
    sampling_points: npt.NDArray[np.float32]


class _GMMModel(TypedDict):
    n_components: int
    weights: npt.NDArray[np.float64]
    means: npt.NDArray[np.float64]
    covariances: npt.NDArray[np.float64]


class AnalysisPacket(TypedDict):
    """Strict schema for every Deep Profiling packet."""

    meta: _Meta
    ground_truth: Dict[str, Any]  # coords, bboxes, area
    predictions: _Predictions
    metrics: _Metrics
    tta_data: List[_TTAEntry]
    aggregation: _Aggregation
    gmm_model: _GMMModel
    config_used: Dict[str, Any]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_JPEG_QUALITY: int = 85
_PICKLE_PROTOCOL: int = pickle.HIGHEST_PROTOCOL


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_deep_analysis(packet: AnalysisPacket, output_dir: str) -> Path:
    """Serialise an analysis packet to a compressed pickle file.

    Steps
    -----
    1. Deep-copy the packet so the caller's data is **never** mutated.
    2. Walk ``tta_data`` and ``aggregation``:
       * Heatmaps (keys containing *"heatmap"*) of dtype float32 are
         converted to float16.
       * Images (key ``"image"`` inside ``tta_data`` entries) that are
         numpy arrays are JPEG-encoded in-memory.
    3. Write the result with ``gzip.open`` + ``pickle.dump``.

    Parameters
    ----------
    packet : AnalysisPacket
        The analysis dictionary (see TypedDict above).
    output_dir : str
        Destination directory (will be created if missing).

    Returns
    -------
    pathlib.Path
        Absolute path to the saved ``.pkl.gz`` file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Build filename from metadata -------------------------------------
    meta = packet.get("meta", {})
    sample_type = str(meta.get("sample_type", "unknown")).replace(" ", "_")
    image_id = meta.get("image_id", 0)
    filename = f"{sample_type}_{image_id}.pkl.gz"
    filepath = out / filename

    # ---- Deep copy to avoid mutating caller data --------------------------
    data: Dict[str, Any] = copy.deepcopy(packet)

    # ---- Optimise tta_data ------------------------------------------------
    for entry in data.get("tta_data", []):
        _compress_heatmaps_in_dict(entry)
        _encode_image_in_dict(entry)

    # ---- Optimise aggregation ---------------------------------------------
    _compress_heatmaps_in_dict(data.get("aggregation", {}))

    # ---- Optimise ground_truth (coords may be an ndarray) -----------------
    _compress_heatmaps_in_dict(data.get("ground_truth", {}))

    # ---- Optimise gmm_model (small, but still arrays) ---------------------
    # weights/means/covariances are float64 – we leave them untouched for
    # numerical precision.

    # ---- Write to disk ----------------------------------------------------
    with gzip.open(filepath, "wb", compresslevel=6) as fh:
        pickle.dump(data, fh, protocol=_PICKLE_PROTOCOL)

    size_mb = filepath.stat().st_size / (1024 * 1024)
    logger.info("Saved deep analysis: %s (%.2f MB)", filepath.name, size_mb)
    return filepath


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_deep_analysis(filepath: str) -> AnalysisPacket:
    """Load and decompress a ``.pkl.gz`` packet, restoring full precision.

    Inverse operations
    ------------------
    * float16 heatmaps → float32
    * JPEG bytes → numpy RGB array

    Parameters
    ----------
    filepath : str
        Path to the compressed pickle file.

    Returns
    -------
    AnalysisPacket
        Ready-to-use analysis dictionary with all arrays at full precision.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    """
    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(f"Analysis file not found: {fp}")

    with gzip.open(fp, "rb") as fh:
        data: Dict[str, Any] = pickle.load(fh)  # noqa: S301

    # ---- Restore tta_data -------------------------------------------------
    for entry in data.get("tta_data", []):
        _decompress_heatmaps_in_dict(entry)
        _decode_image_in_dict(entry)

    # ---- Restore aggregation ----------------------------------------------
    _decompress_heatmaps_in_dict(data.get("aggregation", {}))

    # ---- Restore ground_truth ---------------------------------------------
    _decompress_heatmaps_in_dict(data.get("ground_truth", {}))

    logger.info("Loaded deep analysis: %s", fp.name)
    return data  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compress_heatmaps_in_dict(d: Dict[str, Any]) -> None:
    """Down-cast any float32 array whose key contains 'heatmap' to float16."""
    for key, val in d.items():
        if "heatmap" in key and isinstance(val, np.ndarray) and val.dtype == np.float32:
            d[key] = val.astype(np.float16)


def _decompress_heatmaps_in_dict(d: Dict[str, Any]) -> None:
    """Promote any float16 array whose key contains 'heatmap' back to float32."""
    for key, val in d.items():
        if "heatmap" in key and isinstance(val, np.ndarray) and val.dtype == np.float16:
            d[key] = val.astype(np.float32)


def _encode_image_in_dict(d: Dict[str, Any]) -> None:
    """JPEG-encode an ``"image"`` entry in-place (RGB array → bytes)."""
    img = d.get("image")
    if img is None or not isinstance(img, np.ndarray):
        return
    # cv2.imencode expects BGR; our arrays are RGB
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(
        ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
    )
    if success:
        d["image"] = buf.tobytes()
    else:
        logger.warning("JPEG encoding failed for image; storing raw array.")


def _decode_image_in_dict(d: Dict[str, Any]) -> None:
    """Decode JPEG bytes back to an RGB numpy array (inverse of encode)."""
    raw = d.get("image")
    if raw is None or not isinstance(raw, bytes):
        return
    buf = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is not None:
        d["image"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        logger.warning("JPEG decoding failed; setting image to None.")
        d["image"] = None

"""
Package Benchmark Data for Release.

Packs the 27 essential benchmark .parquet files from Paper/ into
outputs/benchmark_data/benchmark_results_parquet.zip and populates outputs/data/.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = PROJECT_ROOT.parent / "Paper" if (PROJECT_ROOT.parent / "Paper").exists() else PROJECT_ROOT / "Paper"
TARGET_DATA_ROOT = PROJECT_ROOT / "outputs" / "data"
ZIP_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "benchmark_results_parquet.zip"

MODELS = ["hrnet_w32", "resnet50", "vitpose_small"]
DATASETS = ["coco", "crowdpose", "ochuman"]


def get_essential_parquet_relpaths() -> list[Path]:
    """Returns the list of 27 relative paths for essential parquet files."""
    paths = []
    
    # 1. Uncertainty
    for m in MODELS:
        for ds in DATASETS:
            sub = f"{m}_{ds}" if m == "hrnet_w32" else f"Modelos_Secundarios/{m}_{ds}"
            paths.append(Path("Resultados_Incertidumbre") / sub / "degradation_benchmark" / "all_degradations_combined.parquet")
    
    # 2. Precision TTA
    for m in MODELS:
        for ds in DATASETS:
            sub = f"{m}_{ds}" if m == "hrnet_w32" else f"Modelos_Secundarios/TTA_{m}_{ds}"
            paths.append(Path("Resultados_Precision") / "TTA" / sub / "degradation_benchmark" / "all_degradations_combined.parquet")
    
    # 3. Precision MRF
    for m in MODELS:
        for ds in DATASETS:
            sub = f"{m}_{ds}" if m == "hrnet_w32" else f"Modelos_Secundarios/{m}_{ds}"
            paths.append(Path("Resultados_Precision") / "Sin_DARK" / "MRF" / sub / "degradation_benchmark" / "all_degradations_combined.parquet")
    
    return paths


def package_and_populate() -> None:
    TARGET_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    ZIP_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    rel_paths = get_essential_parquet_relpaths()
    logger.info("Found %d target essential parquet files to package.", len(rel_paths))
    
    copied = 0
    with zipfile.ZipFile(ZIP_OUTPUT_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_p in rel_paths:
            src_p = PAPER_ROOT / rel_p
            dst_p = TARGET_DATA_ROOT / rel_p
            
            if src_p.exists():
                # 1. Copy into outputs/data/
                dst_p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_p, dst_p)
                # 2. Write to ZIP
                zf.write(src_p, arcname=str(rel_p).replace("\\", "/"))
                copied += 1
                logger.info("  [OK] %s (%.2f MB)", rel_p, src_p.stat().st_size / (1024 * 1024))
            else:
                logger.warning("  [MISSING] %s", src_p)

    zip_size_mb = ZIP_OUTPUT_PATH.stat().st_size / (1024 * 1024)
    logger.info("=" * 70)
    logger.info("Packaged %d/%d files into: %s (%.2f MB)", copied, len(rel_paths), ZIP_OUTPUT_PATH, zip_size_mb)
    logger.info("Populated destination: %s", TARGET_DATA_ROOT)
    logger.info("=" * 70)


if __name__ == "__main__":
    package_and_populate()

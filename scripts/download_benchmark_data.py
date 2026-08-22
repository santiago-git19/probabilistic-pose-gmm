"""
Automated Benchmark Data Downloader.

Downloads the official pre-evaluated benchmark .parquet results (~55 MB compressed)
from GitHub Releases / Zenodo and extracts them into outputs/data/.
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "outputs" / "data"
LOCAL_ZIP_PATH = PROJECT_ROOT / "outputs" / "benchmark_results_parquet.zip"

DEFAULT_RELEASE_URL = (
    "https://github.com/santiago-git19/robust-pose-tta/releases/download/v1.0.0/benchmark_results_parquet.zip"
)

EXPECTED_FILES = [
    # Uncertainty
    "Resultados_Incertidumbre/hrnet_w32_coco/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Incertidumbre/hrnet_w32_crowdpose/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Incertidumbre/hrnet_w32_ochuman/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Incertidumbre/Modelos_Secundarios/resnet50_coco/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Incertidumbre/Modelos_Secundarios/resnet50_crowdpose/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Incertidumbre/Modelos_Secundarios/resnet50_ochuman/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Incertidumbre/Modelos_Secundarios/vitpose_small_coco/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Incertidumbre/Modelos_Secundarios/vitpose_small_crowdpose/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Incertidumbre/Modelos_Secundarios/vitpose_small_ochuman/degradation_benchmark/all_degradations_combined.parquet",
    # Precision TTA
    "Resultados_Precision/TTA/hrnet_w32_coco/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Precision/TTA/hrnet_w32_crowdpose/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Precision/TTA/hrnet_w32_ochuman/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Precision/TTA/Modelos_Secundarios/TTA_resnet50_coco/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Precision/TTA/Modelos_Secundarios/TTA_resnet50_crowdpose/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Precision/TTA/Modelos_Secundarios/TTA_resnet50_ochuman/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Precision/TTA/Modelos_Secundarios/TTA_vitpose_small_coco/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Precision/TTA/Modelos_Secundarios/TTA_vitpose_small_crowdpose/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Precision/TTA/Modelos_Secundarios/TTA_vitpose_small_ochuman/degradation_benchmark/all_degradations_combined.parquet",
    # Precision MRF
    "Resultados_Precision/Sin_DARK/MRF/hrnet_w32_coco/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Precision/Sin_DARK/MRF/hrnet_w32_crowdpose/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Precision/Sin_DARK/MRF/hrnet_w32_ochuman/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Precision/Sin_DARK/MRF/Modelos_Secundarios/resnet50_coco/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Precision/Sin_DARK/MRF/Modelos_Secundarios/resnet50_crowdpose/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Precision/Sin_DARK/MRF/Modelos_Secundarios/resnet50_ochuman/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Precision/Sin_DARK/MRF/Modelos_Secundarios/vitpose_small_coco/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Precision/Sin_DARK/MRF/Modelos_Secundarios/vitpose_small_crowdpose/degradation_benchmark/all_degradations_combined.parquet",
    "Resultados_Precision/Sin_DARK/MRF/Modelos_Secundarios/vitpose_small_ochuman/degradation_benchmark/all_degradations_combined.parquet",
]


def check_data_complete(target_dir: Path) -> bool:
    """Check whether all 27 expected benchmark parquet files exist."""
    missing = [f for f in EXPECTED_FILES if not (target_dir / f).exists()]
    return len(missing) == 0


def download_progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Print download progress in percentage."""
    if total_size > 0:
        downloaded = block_num * block_size
        pct = min(100.0, (downloaded / total_size) * 100.0)
        sys.stdout.write(f"\rDownloading benchmark data: {pct:.1f}% ({downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB)")
        sys.stdout.flush()


def download_and_extract(
    url: str = DEFAULT_RELEASE_URL,
    target_dir: Path = DATA_ROOT,
    force: bool = False,
) -> bool:
    """Download and extract parquet benchmark files."""
    target_dir.mkdir(parents=True, exist_ok=True)
    
    if not force and check_data_complete(target_dir):
        logger.info("[OK] All 27 benchmark parquet files are already present in: %s", target_dir)
        return True

    # Check if local zip exists
    if LOCAL_ZIP_PATH.exists():
        logger.info("Found local benchmark zip: %s. Extracting...", LOCAL_ZIP_PATH)
        with zipfile.ZipFile(LOCAL_ZIP_PATH, "r") as zf:
            zf.extractall(target_dir)
        logger.info("[SUCCESS] Extracted local archive to: %s", target_dir)
        return True

    # Download from remote URL
    logger.info("Downloading benchmark parquet package from: %s", url)
    temp_zip = target_dir / "temp_benchmark_data.zip"
    try:
        urllib.request.urlretrieve(url, temp_zip, reporthook=download_progress_hook)
        print() # newline after progress hook
        logger.info("Extracting %s to %s...", temp_zip.name, target_dir)
        with zipfile.ZipFile(temp_zip, "r") as zf:
            zf.extractall(target_dir)
        temp_zip.unlink(missing_ok=True)
        logger.info("[SUCCESS] Benchmark parquet data successfully installed in: %s", target_dir)
        return True
    except Exception as e:
        logger.error("Failed to download benchmark data from %s: %s", url, e)
        logger.info("Tip: If you have the dataset locally in Paper/, run 'poetry run python scripts/package_benchmark_data.py'")
        if temp_zip.exists():
            temp_zip.unlink()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and install pre-computed benchmark parquet files.")
    parser.add_argument("--url", type=str, default=DEFAULT_RELEASE_URL, help="URL to download the parquet zip from")
    parser.add_argument("--dest", type=str, default=str(DATA_ROOT), help="Destination directory (default: outputs/data)")
    parser.add_argument("--force", action="store_true", help="Force re-download and extraction")
    args = parser.parse_args()

    success = download_and_extract(url=args.url, target_dir=Path(args.dest), force=args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

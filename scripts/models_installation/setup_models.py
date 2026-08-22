# scripts/models_installation/setup_models.py
"""Cross-platform automated environment asset setup.

Clones MMPose repository, applies Python 3.11 / PEP 517 compatibility patch,
and downloads all required pretrained model checkpoints.
Runs natively on Windows, Linux, and macOS without bash or make dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root and models_installation directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_INST_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MODELS_INST_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_INST_DIR))

REPO_DIR = PROJECT_ROOT / "models" / "mmpose"
PATCH_SCRIPT = MODELS_INST_DIR / "patch_mmpose.py"
DOWNLOAD_SCRIPT = MODELS_INST_DIR / "download_manager.py"


def step_clone_and_patch_mmpose():
    """Clone MMPose repository and apply compatibility patch."""
    print("=" * 70)
    print(">>> [1/2] MMPose Local Repository Setup")
    print("=" * 70)

    if REPO_DIR.exists() and (REPO_DIR / "setup.py").exists():
        print(f"[OK] MMPose repository already present at: {REPO_DIR}")
    else:
        print(f"[INFO] Cloning MMPose (depth=1) into: {REPO_DIR}...")
        REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/open-mmlab/mmpose.git",
                    str(REPO_DIR),
                ],
                check=True,
            )
            print("[SUCCESS] MMPose repository cloned successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to clone MMPose repository: {e}")
            sys.exit(1)

    # Apply compatibility patch
    print("[INFO] Applying PEP 517 compatibility patch to MMPose setup.py...")
    try:
        from patch_mmpose import patch_setup_py

        patch_setup_py()
    except ImportError:
        # Fallback to subprocess
        subprocess.run([sys.executable, str(PATCH_SCRIPT)], check=True)


def step_download_model_weights():
    """Download pretrained backbone weights."""
    print("\n" + "=" * 70)
    print(">>> [2/2] Pretrained Model Checkpoints Download")
    print("=" * 70)

    try:
        from download_manager import MODELS, download_model

        for name, info in MODELS.items():
            download_model(name, info)
    except ImportError:
        subprocess.run([sys.executable, str(DOWNLOAD_SCRIPT)], check=True)


def main():
    print("=" * 70)
    print("   ROBUST POSE TTA - ENVIRONMENT ASSET SETUP")
    print("=" * 70)

    step_clone_and_patch_mmpose()
    step_download_model_weights()

    print("\n" + "=" * 70)
    print("[SUCCESS] All model assets and MMPose engine configured successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

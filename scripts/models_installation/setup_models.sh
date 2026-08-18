#!/bin/bash

# Paths
REPO_DIR="models/mmpose"

# --- 1. CLONE REPOSITORY (Git) ---
if [ -d "$REPO_DIR" ]; then
    echo "[INFO] MMPose repository detected."
else
    echo "[INFO] Cloning MMPose (depth=1)..."
    git clone --depth 1 https://github.com/open-mmlab/mmpose.git $REPO_DIR

    # Apply compatibility patch
    echo "[INFO] Applying patch to setup.py..."
    if [ -f "scripts/models_installation/patch_mmpose.py" ]; then
        python scripts/models_installation/patch_mmpose.py
    fi
fi

# --- 2. DOWNLOAD WEIGHTS VIA PYTHON ---
echo "[INFO] Starting weights download via Python..."
python scripts/models_installation/download_manager.py
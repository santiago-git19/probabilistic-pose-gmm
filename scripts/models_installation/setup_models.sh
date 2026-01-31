#!/bin/bash

# Rutas
REPO_DIR="models/mmpose"

# --- 1. CLONAR EL REPOSITORIO (Git) ---
if [ -d "$REPO_DIR" ]; then
    echo "[INFO] Repositorio MMPose detectado."
else
    echo "[INFO] Clonando MMPose (depth=1)..."
    git clone --depth 1 https://github.com/open-mmlab/mmpose.git $REPO_DIR

    # Aplicar el parche de compatibilidad
    echo "[INFO] Aplicando parche a setup.py..."
    if [ -f "scripts/models_installation/patch_mmpose.py" ]; then
        python scripts/models_installation/patch_mmpose.py
    fi
fi

# --- 2. DELEGAR DESCARGAS A PYTHON ---
# Usamos el script robusto que acabamos de crear
echo "[INFO] Iniciando descarga de pesos con Python..."
python scripts/models_installation/download_manager.py
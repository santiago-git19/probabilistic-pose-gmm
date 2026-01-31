#!/bin/bash

# Este script debe ejecutarse desde la raíz del proyecto.
# Ejemplo: bash scripts/setup_models.sh

# --- CONFIGURACIÓN ---
# Rutas relativas desde la raíz del proyecto
REPO_DIR="models/mmpose"
WEIGHTS_DIR="models/weights"

# --- SEGURIDAD: Verificar que estamos en la raíz ---
if [ ! -f "pyproject.toml" ]; then
    echo "❌ ERROR: Debes ejecutar este script desde la raíz del proyecto."
    echo "   Uso: bash scripts/setup_models.sh"
    exit 1
fi

# Crear directorios
mkdir -p $WEIGHTS_DIR

echo "========================================================"
echo "  CONFIGURACIÓN DEL ENTORNO DE MODELOS (TFG)"
echo "========================================================"

# 1. CLONAR MMPOSE
if [ -d "$REPO_DIR" ]; then
    echo "[INFO] Repositorio MMPose detectado."
else
    echo "[1/4] Clonando MMPose (depth=1)..."
    git clone --depth 1 https://github.com/open-mmlab/mmpose.git $REPO_DIR
fi

# 2. DESCARGAR MODELOS (WGET)
echo "[2/4] Verificando Pesos (Checkpoints)..."

# Función auxiliar para descargar si no existe
download_if_missing() {
    local url=$1
    local dest=$2
    local name=$3
    
    if [ ! -f "$dest" ]; then
        echo " -> Descargando $name..."
        wget -q --show-progress -c "$url" -O "$dest"
    else
        echo " -> $name ya está listo."
    fi
}

# ResNet50
download_if_missing \
    "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192-043c654f_20220916.pth" \
    "$WEIGHTS_DIR/resnet50.pth" \
    "ResNet50"

# HRNet-W32
download_if_missing \
    "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e1b_20220909.pth" \
    "$WEIGHTS_DIR/hrnet_w32.pth" \
    "HRNet-W32"

# ViTPose-Base
download_if_missing \
    "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216e9e0e_20230314.pth" \
    "$WEIGHTS_DIR/vitpose_base.pth" \
    "ViTPose-Base"

echo "========================================================"
echo "  ✅ INSTALACIÓN COMPLETADA"
echo "========================================================"
# src/tests/check_env.py
import os
import sys
import torch
import warnings

# Ignorar warnings de inicialización
warnings.filterwarnings("ignore")

def main():
    print("=== REPORTE DE ENTORNO TFG ===")
    
    # 1. VERIFICAR HARDWARE
    print(f"✅ PyTorch Versión: {torch.__version__}")
    print(f"✅ CUDA Disponible: {torch.cuda.is_available()}")
    
    # 2. DEFINIR RUTAS ABSOLUTAS
    # Guardamos donde estamos ahora (root del proyecto)
    PROJECT_ROOT = os.path.abspath(os.getcwd())
    
    # Definimos donde está MMPose
    MMPOSE_ROOT = os.path.join(PROJECT_ROOT, "models", "mmpose")
    
    # Rutas a los archivos (Absolutas)
    config_file = os.path.join(
        MMPOSE_ROOT, "configs", 
        "body_2d_keypoint", "topdown_heatmap", "coco", 
        "td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
    )
    checkpoint_file = os.path.join(PROJECT_ROOT, "models", "weights", "hrnet_w32.pth")

    # 3. VERIFICAR EXISTENCIA FÍSICA
    if not os.path.exists(MMPOSE_ROOT):
        print(f"❌ ERROR: No encuentro la carpeta models/mmpose en:\n   {MMPOSE_ROOT}")
        return
    if not os.path.exists(config_file):
        print(f"❌ ERROR: No encuentro el config en:\n   {config_file}")
        return
    if not os.path.exists(checkpoint_file):
        print(f"❌ ERROR: No encuentro los pesos en:\n   {checkpoint_file}")
        return

    # 4. CARGA DE MODELO (ESTRATEGIA "CAMBIO DE CONTEXTO")
    print("🔄 Intentando cargar HRNet-W32...")
    
    try:
        from mmpose.apis import init_model
        from mmpose.utils import register_all_modules
        
        register_all_modules()
        
        # --- EL TRUCO DE INGENIERÍA ---
        # Cambiamos el directorio de trabajo a models/mmpose temporalmente.
        # Esto hace que las rutas relativas internas 'configs/_base_...' funcionen.
        print(f"   📂 Cambiando contexto a: {MMPOSE_ROOT}")
        os.chdir(MMPOSE_ROOT)
        
        # Inicializamos el modelo (ahora MMPose se siente como en casa)
        model = init_model(config_file, checkpoint_file, device='cpu')
        
        # Volvemos a casa (importante para el resto del script)
        os.chdir(PROJECT_ROOT)
        print(f"   📂 Contexto restaurado a: {PROJECT_ROOT}")
        
        print("✅ ¡ÉXITO! El modelo se ha cargado en memoria.")
        print(f"   (Modelo: HRNet cargado correctamente con {sum(p.numel() for p in model.parameters()):,} parámetros)")
        
    except Exception as e:
        # Asegurarnos de volver al directorio original si algo falla
        os.chdir(PROJECT_ROOT)
        print(f"❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
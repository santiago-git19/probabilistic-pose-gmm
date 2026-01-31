# scripts/models_installation/download_manager.py
import urllib.request
import os
import sys

# --- CONFIGURACIÓN DE ESPEJOS ---
MODELS = {
    "ResNet50": {
        "mirrors": [
            "https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth",
        ],
        "path": "models/weights/resnet50.pth"
    },
    "HRNet-W32": {
        "mirrors": [
            "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth",
        ],
        "path": "models/weights/hrnet_w32.pth"
    },
    "ViTPose-Base": {
        "mirrors": [

            # MIRROR 2:¡
            "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth"
        ],
        "path": "models/weights/vitpose_small_mmpose.pth",
        # Si todo falla, mostramos este mensaje
        "manual_url": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth" 
    }
}

def report_progress(block_num, block_size, total_size):
    """Barra de progreso visual"""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded * 100 / total_size
        bar_length = 40
        filled_length = int(bar_length * percent / 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f'\r[{bar}] {percent:.1f}% ({downloaded / (1024*1024):.1f} MB)')
        sys.stdout.flush()

def download_model(name, info):
    dest = info['path']
    mirrors = info['mirrors']
    
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"\n🔍 Procesando {name}...")
    
    # 1. Verificar si ya existe
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if size < 10 * 1024 * 1024: 
            print(f"   ⚠️ Archivo existente corrupto ({size} bytes). Borrando...")
            try:
                os.remove(dest)
            except:
                pass
        else:
            print(f"   ✅ {name} ya existe y es válido ({size / (1024*1024):.1f} MB). Saltando.")
            return

    # 2. Bucle de intentos (Mirrors)
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)

    success = False
    for i, url in enumerate(mirrors):
        print(f"   ⬇️ Intento {i+1}: Descargando desde espejo {i+1}...")
        try:
            urllib.request.urlretrieve(url, dest, report_progress)
            if os.path.exists(dest) and os.path.getsize(dest) > 10 * 1024 * 1024:
                print("\n   ✅ Descarga completada con éxito.")
                success = True
                break
        except Exception as e:
            print(f"\n   ❌ Falló el espejo {i+1}: {e}")
            if os.path.exists(dest):
                try: os.remove(dest)
                except: pass
    
    if not success:
        print(f"\n   ❌ NO SE PUDO DESCARGAR AUTOMÁTICAMENTE {name}.")
        print("   ⚠️  MOTIVO: Los enlaces oficiales están en OneDrive y bloquean scripts.")
        if "manual_url" in info:
            print(f"\n   👉 SOLUCIÓN MANUAL:")
            print(f"   1. Abre este enlace: {info['manual_url']}")
            print(f"   2. Descarga el archivo 'vitpose_base_coco_256x192.pth'")
            print(f"   3. Muévelo a esta carpeta: {os.path.abspath(dest)}")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 GESTOR DE DESCARGAS: MIRRORS COMUNITARIOS")
    for name, info in MODELS.items():
        download_model(name, info)
    print("\n✨ Proceso finalizado.")
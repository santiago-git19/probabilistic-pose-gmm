"""
Script de diagnóstico para inspeccionar la salida de MMPose.

Este script analiza la estructura de datos devuelta por inference_topdown
para identificar dónde están los heatmaps reales.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "mmpose"))

from pose_uncertainty.models.adapters import ModelPathResolver
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

def inspect_result_structure(result, prefix="", max_depth=5):
    """Recursivamente inspecciona la estructura de un objeto."""
    if max_depth == 0:
        return
    
    print(f"{prefix}Type: {type(result).__name__}")
    
    # Si es un objeto con atributos
    if hasattr(result, "__dict__"):
        attrs = [a for a in dir(result) if not a.startswith("_")]
        print(f"{prefix}Attributes: {attrs[:10]}{'...' if len(attrs) > 10 else ''}")
        
        # Inspeccionar algunos atributos clave
        for attr in ["pred_fields", "heatmaps", "pred_instances", "keypoints", "keypoint_scores"]:
            if hasattr(result, attr):
                value = getattr(result, attr)
                print(f"{prefix}  .{attr}:")
                inspect_result_structure(value, prefix + "    ", max_depth - 1)
    
    # Si es numpy array o tensor
    elif isinstance(result, (np.ndarray, list)):
        if isinstance(result, np.ndarray):
            print(f"{prefix}Shape: {result.shape}, Dtype: {result.dtype}")
            print(f"{prefix}Range: [{result.min():.3e}, {result.max():.3e}]")
        else:
            print(f"{prefix}Length: {len(result)}")
            if len(result) > 0:
                inspect_result_structure(result[0], prefix + "  [0]: ", max_depth - 1)
    
    # Si tiene método cpu/numpy (tensor de PyTorch)
    elif hasattr(result, "cpu"):
        print(f"{prefix}PyTorch Tensor: shape={result.shape}, device={result.device}")
        try:
            arr = result.cpu().numpy()
            print(f"{prefix}As numpy: shape={arr.shape}, range=[{arr.min():.3e}, {arr.max():.3e}]")
        except:
            pass


def main():
    """Ejecuta el diagnóstico."""
    print("=" * 80)
    print("DIAGNÓSTICO DE SALIDA DE MMPOSE")
    print("=" * 80)
    
    # Configurar paths
    resolver = ModelPathResolver(PROJECT_ROOT)
    register_all_modules()
    
    # Probar con ResNet-50
    model_name = "resnet50"
    config_path = resolver.resolve_config(
        "body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py"
    )
    checkpoint_path = resolver.resolve_checkpoint("resnet50.pth")
    
    print(f"\nCargando modelo: {model_name}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Cambiar al directorio de mmpose
    original_cwd = os.getcwd()
    os.chdir(str(resolver.mmpose_root))
    
    try:
        model = init_model(str(config_path), str(checkpoint_path), device="cpu")
        print("✅ Modelo cargado")
        
        # Crear imagen de prueba
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = np.array([[100, 100, 300, 400]])  # (x1, y1, x2, y2)
        
        print("\n🔄 Ejecutando inferencia...")
        results = inference_topdown(model, test_image, bboxes=bbox)
        
        print(f"\n📊 Resultados de inferencia (total: {len(results)}):")
        
        for i, result in enumerate(results):
            print(f"\n{'='*70}")
            print(f"Resultado #{i}:")
            print(f"{'='*70}")
            inspect_result_structure(result, "  ")
            
            # Intentar acceder a diferentes ubicaciones donde podrían estar los heatmaps
            print(f"\n🔍 Buscando heatmaps en ubicaciones comunes:")
            
            locations = [
                ("result.pred_fields.heatmaps", lambda r: r.pred_fields.heatmaps),
                ("result.pred_fields", lambda r: r.pred_fields),
                ("result.output.heatmaps", lambda r: r.output.heatmaps if hasattr(r, 'output') else None),
                ("result.heatmaps", lambda r: r.heatmaps if hasattr(r, 'heatmaps') else None),
            ]
            
            for name, getter in locations:
                try:
                    value = getter(result)
                    if value is not None:
                        print(f"\n  ✅ Encontrado en: {name}")
                        inspect_result_structure(value, "      ")
                except Exception as e:
                    print(f"  ❌ {name}: {type(e).__name__}: {e}")
            
            # Inspeccionar pred_instances
            if hasattr(result, "pred_instances"):
                print(f"\n📦 pred_instances:")
                pred = result.pred_instances
                if hasattr(pred, "keypoints"):
                    kp = pred.keypoints[0]
                    print(f"  keypoints shape: {kp.shape}")
                    print(f"  keypoints range: x[{kp[:, 0].min():.1f}, {kp[:, 0].max():.1f}], "
                          f"y[{kp[:, 1].min():.1f}, {kp[:, 1].max():.1f}]")
                if hasattr(pred, "keypoint_scores"):
                    scores = pred.keypoint_scores[0]
                    print(f"  scores shape: {scores.shape}")
                    print(f"  scores range: [{scores.min():.3f}, {scores.max():.3f}]")
                    print(f"  scores > 0.3: {(scores > 0.3).sum()}/{len(scores)}")
    
    finally:
        os.chdir(original_cwd)
    
    print("\n" + "=" * 80)
    print("DIAGNÓSTICO COMPLETO")
    print("=" * 80)


if __name__ == "__main__":
    main()

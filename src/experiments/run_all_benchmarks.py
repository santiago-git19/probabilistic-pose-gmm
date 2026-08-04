import subprocess
import sys
import os
import time

def run_experiment(model, dataset_args, log_file, timestamp):
    """Ejecuta un experimento de hydra y guarda su salida en un archivo de log."""
    ds_name = [arg for arg in dataset_args if arg.startswith("dataset.name=")][0].split("=")[1]
    out_dir = f"outputs/benchmarks/{timestamp}_{model}_{ds_name}"
    
    cmd = [
        sys.executable,
        "src/experiments/run_degradation_benchmark.py",
        f"model={model}",
        f"hydra.run.dir={out_dir}",
        f"logging.output_dir={out_dir}"
    ] + dataset_args
    
    print(f"[{model} | {ds_name}] Iniciando experimento... (Guardando salida en {out_dir})")
    
    # Redirigimos stdout y stderr al archivo log para no ensuciar la terminal
    with open(log_file, "w", encoding="utf-8") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return process

def main():
    models = ["hrnet_w32", "resnet50", "vitpose_small"]
    
    # Definimos los argumentos exactos (overrides de Hydra) para cada dataset 
    # basándonos en tu config.yaml
    datasets = {
        "coco": [
            "dataset.name=coco",
            "dataset.root_dir=${paths.project_root}/data/coco",
            "dataset.annotations=annotations/person_keypoints_val2017.json",
            "dataset.images=val2017"
        ],
        "crowdpose": [
            "dataset.name=crowdpose",
            "dataset.root_dir=${paths.project_root}/data/crowdpose",
            "dataset.annotations=json/crowdpose_val.json",
            "dataset.images=images"
        ],
        "ochuman": [
            "dataset.name=ochuman",
            "dataset.root_dir=${paths.project_root}/data/ochuman",
            "dataset.annotations=annotations/ochuman_coco_format_val_range_0.00_1.00.json",
            "dataset.images=images"
        ]
    }
    
    # Carpeta para guardar los logs de las terminales virtuales
    os.makedirs("logs", exist_ok=True)
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"=== INICIANDO BATERÍA PARA EL MODELO: {model.upper()} ===")
        print(f"{'='*60}")
        
        processes = []
        
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        
        # 1. Crear los 3 procesos (uno por dataset) en paralelo
        for ds_name, ds_args in datasets.items():
            log_file = f"logs/benchmark_{model}_{ds_name}.log"
            p = run_experiment(model, ds_args, log_file, timestamp)
            processes.append((ds_name, p))
            
        # 2. Esperar a que los 3 hilos/procesos terminen antes de pasar al siguiente modelo
        print("\nEsperando a que los 3 datasets terminen (puedes monitorear los archivos .log en la carpeta 'logs/')...")
        for ds_name, p in processes:
            p.wait() # Esto bloquea hasta que termine este proceso
            
            if p.returncode == 0:
                print(f"[{model} | {ds_name}] FINALIZADO CON ÉXITO.")
            else:
                print(f"[{model} | {ds_name}] ERROR (Código {p.returncode}). Por favor revisa 'logs/benchmark_{model}_{ds_name}.log'.")
                
        print(f"\nBatería para {model} completada. Limpiando memoria y esperando 5 segundos...")
        time.sleep(5)
        
    print("\n" + "*"*50)
    print("TODOS LOS BENCHMARKS HAN FINALIZADO.")
    print("*"*50)

if __name__ == "__main__":
    main()

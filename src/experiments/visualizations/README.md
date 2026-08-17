# Visualization Suite for Research Paper (Information Fusion / Elsevier)

Este módulo contiene todos los scripts dedicados al minado de datos, análisis de paquetes diagnósticos (`.pkl.gz`), perfilado bajo demanda y renderizado vectorial de figuras cualitativas listas para publicación.

---

## Estructura del Módulo

| Script | Propósito | Entrada Principal | Salida Principal |
| :--- | :--- | :--- | :--- |
| **`find_figure_candidates.py`** | Motor de consultas analíticas sobre los `.parquet` (239 columnas) de todas las etapas del pipeline (`Incertidumbre`, `TTA`, `MRF`). | Tablas `.parquet` | Catálogo de candidatos en consola / JSON. |
| **`generate_candidate_packets.py`** | Perfilador profundo bajo demanda para candidatos seleccionados en `.parquet`. Ejecuta inferencia targeted y genera sus `.pkl.gz`. | Especificaciones `(image_id, dataset, model, exp)` | Archivos `.pkl.gz` y figuras renderizadas. |
| **`generate_all_paper_figures.py`** | Renderizador maestro de las 5 figuras cualitativas del paper (3 candidatos por figura, 15 figuras en total). | Paquetes `.pkl.gz` | Figuras en PDF vectorial + PNG a 300 DPI en `Paper/Paper/figures/visualizaciones/`. |
| **`export_paper_packet_figure.py`** | Exportador modular de un paquete individual en 4 paneles + capas limpias, crops y JSON numérico. | Archivo `.pkl.gz` o `--image-id` | Panel composite + carpeta de assets modulares. |
| **`mine_paper_candidates.py`** | Minador y clasificador de paquetes existentes en `outputs/` con interfaz interactiva en FiftyOne. | Carpetas con `.pkl.gz` | Tablas de victorias, swaps y servidor FiftyOne. |

---

## Guía de Uso Rápido

### 1. Minar candidatos en las tablas `.parquet`
```bash
poetry run python src/experiments/visualizations/find_figure_candidates.py --scan-all
```

### 2. Generar paquetes y renderizar figuras bajo demanda
```bash
poetry run python src/experiments/visualizations/generate_candidate_packets.py
```

### 3. Renderizar todas las figuras del paper desde paquetes locales
```bash
poetry run python src/experiments/visualizations/generate_all_paper_figures.py
```

### 4. Exportar assets modulares (PNG, PDF, crops, JSON) para una figura específica
```bash
poetry run python src/experiments/visualizations/export_paper_packet_figure.py --image-id 462 --export-modular
```

### 5. Explorar visualmente los paquetes en FiftyOne
```bash
poetry run python src/experiments/visualizations/mine_paper_candidates.py --launch-fiftyone
```

# Visualization Suite for Scientific Publications

This module contains scripts dedicated to candidate data mining, diagnostic packet (`.pkl.gz`) deep profiling, on-demand targeted inference, and publication-ready vector rendering for qualitative figures.

---

## Module Overview

| Script | Purpose | Primary Input | Primary Output |
| :--- | :--- | :--- | :--- |
| **`find_figure_candidates.py`** | Analytical query engine operating on comprehensive `.parquet` tables (239 columns) across all pipeline stages (`Uncertainty`, `TTA`, `MRF`). | Parquet tables | Candidate catalog in console and JSON. |
| **`generate_candidate_packets.py`** | On-demand targeted deep profiler for candidates mined from `.parquet`. Runs inference and serializes diagnostic `.pkl.gz` packets. | Specifications `(image_id, dataset, model, exp)` | Serialized `.pkl.gz` packets and rendered figures. |
| **`generate_all_paper_figures.py`** | Master renderer for paper qualitative figures across keypoint uncertainty, topological disambiguation, and degradation progression. | Diagnostic `.pkl.gz` packets | Vector PDF and 300 DPI PNG figures saved to `outputs/figures/visualizations/`. |
| **`export_paper_packet_figure.py`** | Modular exporter decomposing a single packet into multi-panel layouts, isolated layer elements, high-resolution crops, and numeric JSON data. | Packet file `.pkl.gz` or `--image-id` | Composite figure panel and directory of modular assets. |
| **`mine_paper_candidates.py`** | Miner and cataloger for existing packets across `outputs/` with interactive FiftyOne visualization interface. | Directories containing `.pkl.gz` | Summary tables of topological fixes and interactive FiftyOne server. |

---

## Quickstart Guide

### 1. Mine qualitative candidates from Parquet results
```bash
poetry run python src/experiments/visualizations/find_figure_candidates.py --scan-all
```

### 2. Generate diagnostic packets and render figures on demand
```bash
poetry run python src/experiments/visualizations/generate_candidate_packets.py
```

### 3. Render all publication figures from local packets
```bash
poetry run python src/experiments/visualizations/generate_all_paper_figures.py
```

### 4. Export modular assets (PNG, PDF, crops, JSON) for a specific candidate
```bash
poetry run python src/experiments/visualizations/export_paper_packet_figure.py --image-id 462 --export-modular
```

### 5. Interactively explore diagnostic packets in FiftyOne
```bash
poetry run python src/experiments/visualizations/mine_paper_candidates.py --launch-fiftyone
```

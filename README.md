# NEXUS: Adaptive Urban Routing for Mobility-Impaired Individuals via Graph Theory and Dynamic Impedance

## Abstract
Traditional routing systems optimize for distance or temporal efficiency, failing to account for the biomechanical cost imposed by topographic variance and micro-barriers (e.g., degraded surfaces, absence of sloped curbs). This project proposes NEXUS, a computational architecture that mitigates urban exclusion by calculating the route of minimal "biomechanical effort." Utilizing OpenStreetMap (OSM) data, Digital Elevation Models (DEM), and a Pessimistic Safe Fallback imputation heuristic, NEXUS applies an advanced dynamic routing algorithm (D* Lite) on a weighted spatial graph to dynamically recalibrate paths based on community-reported impediments (VGI).

## Project Architecture
* `data_pipeline/`: Handles ETL processes, spatial graph instantiation via OSMnx, and deterministic imputation of missing accessibility tags.
* `core/`: Contains the mathematical impedance model, dynamic graph state management, and the routing algorithms.
* `evaluation/`: Scripts for empirical validation, computing the Route Overlap Index, and generating spatial visualizations.

## Scientific Objectives
1. Extract and process geospatial, elevation, and infrastructure data.
2. Develop a modified cost function weighting micro-barriers and surface topology.
3. Implement a dynamic routing architecture capable of processing real-time community inputs without total graph recompilation.
4. Empirically validate the biomechanical efficiency of NEXUS against standard shortest-path algorithms.

## Mapillary + YOLO26 Segmentation Pipeline (Accessibility Barriers)

### Why cloud is mandatory for this stage
Mapillary Vistas is very large and impractical for most local laptops. NEXUS now supports a sharded workflow so you can:
- keep only metadata locally,
- process images in chunks on cloud GPU instances,
- avoid full-dataset download on your machine.

### Accessibility class taxonomy used in NEXUS
`data_pipeline/semantic_segmentation_training.py` remaps Mapillary labels into:
- `street`
- `sidewalk`
- `curb`
- `crossing`
- `steps`
- `ramp`
- `handrail`
- `tactile_paving`
- `lighting`
- `surface_obstacle`

This taxonomy is aligned with barrier analysis for mobility-impaired users.

### Step 1 — Prepare sharded manifests (local or cloud)
You need a metadata CSV containing at least:
- `image_id`
- `image_path` (URL/path)
- `latitude`
- `longitude`
- optional: `city`

Run:

```bash
python3 data_pipeline/semantic_segmentation_training.py prepare-manifests \
  --metadata-csv data_files/mapillary_metadata.csv \
  --output-dir data_files/mapillary_shards \
  --train-ratio 0.85 \
  --shard-size 5000 \
  --city-filter "Lourdes"
```

For explicit spatial filtering, use:

```bash
--bbox "min_lon,min_lat,max_lon,max_lat"
```

### Step 2 — Build YOLO dataset YAML
After cloud preprocessing converts masks/images to YOLO format:

```bash
python3 data_pipeline/semantic_segmentation_training.py build-yolo-config \
  --output-yaml data_files/yolo_accessibility.yaml \
  --train-images-dir /mnt/datasets/nexus/images/train \
  --val-images-dir /mnt/datasets/nexus/images/val
```

### Step 3 — Train YOLO26-seg (cloud GPU)
The training command itself must run in your cloud runtime (Colab Pro, Kaggle, AWS, GCP, etc.), for example with Ultralytics:

```bash
yolo task=segment mode=train \
  model=yolo26n-seg.pt \
  data=data_files/yolo_accessibility.yaml \
  epochs=100 imgsz=1024 batch=8 device=0 workers=8
```

If your environment does not provide `yolo26n-seg.pt`, use the exact YOLO26 checkpoint name available in your runtime.

### Step 4 — Run inference for Lourdes images
Run inference in cloud and export a JSON list with this shape:
- top-level list of image items,
- each image item has `image_id` and `detections`,
- each detection has `class_name`, `confidence`, `polygon_xy`.

### Step 5 — Score accessibility barriers
Compute per-image accessibility scores:

```bash
python3 data_pipeline/semantic_segmentation_training.py score-lourdes \
  --predictions-json data_files/lourdes_predictions.json \
  --output-csv data_files/lourdes_accessibility_scores.csv \
  --image-area 921600 \
  --confidence-threshold 0.30
```

### What this pipeline can do vs. what still needs external modules
Implemented in-repo:
- sharded metadata manifests for heavy dataset handling,
- class remapping from semantic labels to accessibility barriers,
- deterministic split generation (`train`/`val`),
- barrier scoring from segmentation outputs.

Must still be done externally (cloud stage):
- Mapillary API/authenticated image retrieval at scale,
- segmentation mask conversion into YOLO label files,
- actual YOLO26 training/inference runtime with GPU.

### Notes about harder attributes
Some requested indicators (e.g., sidewalk width, pavement smoothness quality, nuanced tactile conditions) are not reliably solved by semantic segmentation alone. The production approach is:
1. segmentation for object presence and rough extent,
2. monocular depth + camera calibration for width estimation,
3. secondary classifier or manual audit for texture/smoothness quality.

# Conditional Floor Plan Generation (CubiCasa5K)

This repository contains the implementation for an MSc Data Science
project on **conditional semantic floor plan generation** using the
CubiCasa5K dataset.

The objective is to generate structured 2D residential layouts from
simple inputs (building outline + room count), evaluate them using
spatial metrics, and improve geometric quality through lightweight
refinement.

------------------------------------------------------------------------

## Dataset

This project uses the **CubiCasa5K** dataset:

*CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan
Image Analysis*

Download links: - Zenodo: https://zenodo.org/record/2613548 - GitHub:
https://github.com/CubiCasa/CubiCasa5k

After downloading, place the dataset in:

data/cubicasa5k/<cat_name>/<sample_id>/model.svg

Example:

data/cubicasa5k/high_quality_architectural/10000/model.svg

The dataset is **not included** in this repository due to size
constraints.

------------------------------------------------------------------------

## Environment

-   Python 3.9
-   Apple MacBook Pro (M1)
-   PyTorch with Metal Performance Shaders (MPS)

Install dependencies:

```bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Pipeline Overview

The complete pipeline consists of:

1.  SVG preprocessing
2.  Dataset filtering
3.  Conditional UNet training
4.  Spatial evaluation
5.  Post-generation refinement

------------------------------------------------------------------------

## Step 1: Preprocess CubiCasa5K (SVG → NPZ)

```bash
python -m src.preprocess.preprocess_cubicasa --data_root data/cubicasa5k
--out_dir data/processed_npz
```

Optional:

```bash
python -m src.preprocess.preprocess_cubicasa --data_root data/cubicasa5k
--out_dir data/processed_npz --max_samples 500
```
------------------------------------------------------------------------

## Step 2: Inspect Generated NPZ Files
```bash
python scripts/inspect_npz.py
```

------------------------------------------------------------------------

## Step 3: Filter Dataset (≥ 3 rooms)
```bash
python scripts/filter_dataset.py
```

Filtered output:

```bash
data/processed_npz_clean/
```

------------------------------------------------------------------------

## Step 4: Compute Maximum Room Count

```bash
python scripts/compute_max_count.py
```

------------------------------------------------------------------------

## Step 5: Train Conditional UNet Baseline

```bash
python train_unet.py
```

Model checkpoints:

outputs/checkpoints/

------------------------------------------------------------------------

## Step 6: Evaluate Baseline Metrics

```bash
python -m scripts.evaluate_metrics
```

Results:

outputs/metrics_baseline.csv

------------------------------------------------------------------------

## Step 7: Refinement

### Morphological Refinement (Adopted)

```bash
python -m scripts.refine_morphology
```

Output:

outputs/metrics_refined_morph.csv

### Hill-Climbing Refinement (Experimental)

```bash
python -m scripts.refine_hillclimb
```

------------------------------------------------------------------------

## Current Status

-   SVG preprocessing implemented
-   Dataset filtering implemented
-   Conditional UNet baseline trained
-   Spatial evaluation metrics implemented
-   Morphological refinement implemented
-   Conditional GAN extension (planned)

------------------------------------------------------------------------

## License

This repository is intended for academic research purposes.

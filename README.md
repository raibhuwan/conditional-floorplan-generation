![Python](https://img.shields.io/badge/python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)

# Conditional Semantic Floor Plan Generation

This repository contains the implementation for an MSc Data Science project on conditional semantic floor plan generation using the CubiCasa5K dataset. The project investigates how two-dimensional semantic floor plan masks can be generated from a building outline and a room-count condition.

The aim of the project is not to generate complete architectural drawings with doors, windows, dimensions or room labels. Instead, the project focuses on generating semantic layout masks and evaluating them using both pixel-level and spatial metrics.

---

## Project Aim

The aim of this project is to develop and evaluate a conditional semantic floor plan generation framework that generates two-dimensional semantic layout masks from building-outline and room-count conditions.

The project compares:

- a U-Net baseline,
- a Pix2Pix-style conditional GAN,
- morphology-based refinement,
- and hill-climbing refinement.

The generated layouts are evaluated using semantic and spatial metrics, including mIoU, adjacency similarity, compactness, boundary violation rate and room-count error.

---

## Output Scope

The system output is a colourised semantic floor plan mask.

```text
building outline + room count -> semantic floor plan mask
```

The system does not generate a complete architectural drawing. The following elements are outside the scope of this project:

- doors,
- windows,
- room labels,
- dimensions,
- furniture,
- architectural symbols,
- and vector CAD-style drawings.

Producing a complete architectural drawing would require additional rendering, vectorisation or symbol-generation stages and is treated as future work.

---

## Dataset

This project uses the CubiCasa5K dataset:

**CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis**

Dataset links:

- Zenodo: https://zenodo.org/record/2613548
- GitHub: https://github.com/CubiCasa/CubiCasa5k

The dataset is not included in this repository due to size constraints.

Final experiment dataset:

```text
CubiCasa5K high_quality_architectural subset
Processed samples: 4566
Filtered clean samples: 3761
Train / validation / test split: 2632 / 564 / 565
MAX_COUNT: 32
```

---

## Key Features

- SVG-based CubiCasa5K preprocessing
- Semantic mask generation at 256 x 256 resolution
- Building-outline and room-count conditioning
- Conditional U-Net baseline
- Pix2Pix-style conditional GAN with U-Net generator and PatchGAN discriminator
- Morphology-based post-processing
- Hill-climbing refinement
- Held-out test evaluation
- Manual generation from boundary image and room-count input
- Training and evaluation on Apple M1 hardware using PyTorch MPS

---

## Repository Structure

```text
conditional-floorplan-generation/
|
|-- data/
|   |-- cubicasa5k/
|   |-- processed_npz_full/
|   |-- processed_npz_clean_full/
|
|-- outputs/
|   |-- checkpoints/
|   |-- generated/
|   |-- splits/
|   |-- metrics_*.csv
|
|-- scripts/
|   |-- create_split.py
|   |-- filter_dataset.py
|   |-- compute_max_count.py
|   |-- evaluate_metrics.py
|   |-- evaluate_cgan.py
|   |-- evaluate_unet_morphology.py
|   |-- evaluate_cgan_morphology.py
|   |-- evaluate_unet_hillclimb.py
|   |-- evaluate_cgan_hillclimb.py
|   |-- export_boundary_sample.py
|   |-- generate_floorplan.py
|
|-- src/
|   |-- data/
|   |-- models/
|   |-- preprocessing/
|   |-- refinement/
|
|-- train_unet.py
|-- train_cgan.py
|-- README.md
|-- RUNNING_PROJECT.md
|-- EXPERIMENTS.md
```

---

## Environment

The project was developed and tested using:

```text
Python 3.9
PyTorch
OpenCV
NumPy
Apple MacBook Pro M1
8 GB unified memory
Metal Performance Shaders (MPS)
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

The full command guide is provided in:

```text
RUNNING_PROJECT.md
```

That file includes the commands for:

- filtering the dataset,
- computing `MAX_COUNT`,
- creating the train / validation / test split,
- training U-Net,
- training the cGAN,
- evaluating all final methods,
- exporting a boundary image,
- and generating a semantic floor plan from boundary + room count.

Example final generation command:

```bash
python -m scripts.generate_floorplan \
  --outline_path inputs/boundary.png \
  --room_count 6 \
  --ckpt_path outputs/checkpoints/unet_base16_best.pt \
  --max_count 32 \
  --apply_morphology \
  --out_path outputs/generated/floorplan_6rooms.png \
  --mask_out_path outputs/generated/floorplan_6rooms.npy
```

---

## Final Models

### U-Net Baseline

```text
Checkpoint: outputs/checkpoints/unet_base16_best.pt
Best validation IoU: 0.4706
Best epoch: 20
```

### Pix2Pix-Style cGAN

```text
Checkpoint: outputs/checkpoints/cgan_unet_patchgan_best.pt
Best validation IoU: 0.3860
Best epoch: 20
```

The final selected generation route is U-Net with morphology refinement, because it achieved the strongest overall balance across the final held-out test metrics.

---

## Final Held-Out Test Results

Final evaluation was performed on 565 unseen test samples from the filtered full `high_quality_architectural` subset.

| Method | mIoU | Adj F1 | Compactness | BVR | RC-MAE |
|---|---:|---:|---:|---:|---:|
| U-Net baseline | 0.375 | 0.191 | 0.508 | 0.000 | 8.287 |
| U-Net + morphology | 0.381 | 0.194 | 0.629 | 0.001 | 7.156 |
| U-Net + hill-climbing | 0.292 | 0.192 | 0.551 | 0.135 | 4.361 |
| cGAN baseline | 0.363 | 0.192 | 0.508 | 0.000 | 9.377 |
| cGAN + morphology | 0.367 | 0.190 | 0.646 | 0.001 | 8.550 |
| cGAN + hill-climbing | 0.281 | 0.186 | 0.558 | 0.135 | 5.513 |

---

## Main Findings

The final results show that U-Net with morphology refinement achieved the best overall balance across semantic and spatial evaluation metrics. Morphology improved compactness and room-count error while preserving semantic quality reasonably well.

The Pix2Pix-style cGAN did not outperform the U-Net baseline in the final held-out evaluation. Hill-climbing reduced room-count error but also reduced mIoU and increased boundary violation, showing that improving one metric can damage other aspects of layout quality.

These findings support the need for multi-metric evaluation in floor plan generation. Pixel-level semantic accuracy alone is not sufficient to assess whether a generated layout is spatially coherent.

---

## Documentation

Detailed project documentation is provided in:

```text
RUNNING_PROJECT.md  - commands for running the project
EXPERIMENTS.md      - experiment log and final results
```

---

## Current Status

Completed:

- CubiCasa5K preprocessing
- dataset filtering
- train / validation / test split
- U-Net training
- cGAN training
- baseline evaluation
- morphology refinement
- hill-climbing refinement
- final held-out test evaluation
- manual semantic floor plan generation script

---

## Future Work

Future work may include:

- stronger conditioning using room adjacency graphs,
- explicit architectural constraints,
- improved room-count control,
- better boundary-aware refinement,
- vectorisation of semantic masks,
- and generation of complete architectural drawings with doors, windows and symbols.

---

## License

This repository is intended for academic research purposes.

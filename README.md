![Python](https://img.shields.io/badge/python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)

# Conditional Semantic Floor Plan Generation

This repository contains the implementation for an MSc Data Science project on conditional semantic floor-plan generation using the CubiCasa5K dataset. The project investigates whether a deep-learning model can generate a multi-class semantic floor-plan mask from a filled binary floor-plan support condition and an encoded connected-region count condition.

The output is an approximate semantic layout for experimental evaluation. It is not a complete or construction-ready architectural drawing.

## Project Aim

The project develops and evaluates a conditional semantic floor-plan generation framework and compares:

- a supervised U-Net baseline;
- a Pix2Pix-style conditional GAN using a U-Net generator and PatchGAN discriminator;
- morphology-based refinement; and
- compactness-based hill-climbing refinement.

The generated masks are assessed using:

- mean Intersection over Union (mIoU);
- adjacency F1;
- compactness;
- boundary violation rate (BVR); and
- connected-region count Mean Absolute Error (MAE).

## Input and Output Representation

The final model input contains two channels:

```text
filled binary support mask + encoded connected-region count -> semantic floor-plan mask
```

The count is produced from eight-connected components of the combined non-background, non-wall semantic mask. It is therefore a lightweight indicator of layout complexity rather than a manually verified architectural room count.

Historical implementation names such as `outline`, `room_count`, `--outline_path` and `--room_count` are retained for compatibility. The dissertation uses the more precise terms **support mask** and **encoded connected-region count**.

The system does not generate:

- verified individual room instances;
- doors or windows;
- room labels;
- dimensions;
- furniture;
- structural calculations;
- architectural symbols; or
- vector/CAD construction drawings.

## Dataset

The project uses the CubiCasa5K `high_quality_architectural` subset.

Dataset links:

- Zenodo: https://zenodo.org/record/2613548
- GitHub: https://github.com/CubiCasa/CubiCasa5k

The dataset is not included in this repository due to size constraints.

Final preprocessing and split evidence:

```text
Successfully processed floor-level samples: 4566
Excluded with encoded count below 3: 805
Retained samples: 3761
Training / validation / test: 2632 / 564 / 565
MAX_COUNT: 32
Seed: 42
```

## Key Features

- SVG-based CubiCasa5K preprocessing
- 256 x 256 nine-class semantic masks
- filled binary support-mask conditioning
- encoded connected-region count conditioning
- conditional U-Net baseline
- Pix2Pix-style cGAN with U-Net generator and PatchGAN discriminator
- morphology-based post-processing
- compactness-based hill-climbing refinement
- fixed train / validation / test split
- held-out multi-metric evaluation
- condition-sensitivity and determinism checks
- manual semantic-mask generation
- Apple M1 / PyTorch MPS experimentation

## Repository Structure

```text
conditional-floorplan-generation/
|
|-- data/
|   |-- cubicasa5k/
|   |-- processed_npz_full/
|   |-- processed_npz_clean_full/
|
|-- evidence/
|   |-- training/
|   |-- metrics/
|   |-- preprocessing/
|   |-- condition_sensitivity/
|   |-- splits/
|   |-- README.md
|
|-- outputs/
|   |-- checkpoints/
|   |-- generated/
|   |-- logs/
|   |-- splits/
|   |-- development and metric CSV files
|
|-- scripts/
|   |-- create_split.py
|   |-- filter_dataset.py
|   |-- compute_max_count.py
|   |-- evaluate_metrics.py
|   |-- evaluate_cgan_metrics.py
|   |-- evaluate_unet_morphology.py
|   |-- evaluate_cgan_morphology.py
|   |-- evaluate_unet_hillclimb.py
|   |-- evaluate_cgan_hillclimb.py
|   |-- export_boundary_sample.py
|   |-- generate_floorplan.py
|   |-- evidence/
|
|-- src/
|   |-- data/
|   |-- models/
|   |-- preprocess/
|   |-- refinement/
|
|-- train_unet.py
|-- train_cgan.py
|-- README.md
|-- RUNNING_PROJECT.md
|-- EXPERIMENTS.md
|-- preprocessing_evidence_audit.md
|-- condition_sensitivity_evidence_audit.md
```

## Environment

The project was developed using:

```text
Python 3.9
PyTorch
OpenCV
NumPy
Pillow
Apple MacBook Pro M1
8 GB unified memory
Metal Performance Shaders (MPS)
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Project

The complete command guide is provided in:

```text
RUNNING_PROJECT.md
```

It covers dataset filtering, `MAX_COUNT`, train/validation/test splitting, both training routes, all six final evaluations, support-image export and final generation.

Example final generation command:

```bash
python -m scripts.generate_floorplan \
  --outline_path inputs/boundary.png \
  --room_count 8 \
  --ckpt_path outputs/checkpoints/unet_base16_best.pt \
  --max_count 32 \
  --apply_morphology \
  --out_path outputs/generated/floorplan_8rooms.png \
  --mask_out_path outputs/generated/floorplan_8rooms.npy
```

The command retains historical argument names; `--outline_path` supplies the support image and `--room_count` supplies the encoded connected-region count condition.

## Final Training Evidence

The final dissertation uses the same validation mIoU calculation recorded during training for checkpoint comparison. The calculation obtains mIoU for each validation batch and averages the batch-level values across the validation partition.

### U-Net

```text
Selected epoch: 20
Validation mIoU at selected epoch: 0.399831
Training mIoU at epoch 20: 0.621130
Final training mIoU: 0.736845
Final validation mIoU: 0.369020
Approximate training time: 54.0 minutes
```

### Pix2Pix-Style cGAN

```text
Selected epoch: 20
Validation mIoU at selected epoch: 0.386000
Training mIoU at epoch 20: 0.659419
Final training mIoU: 0.748921
Final validation mIoU: 0.351045
Approximate training time: 101.7 minutes
```

## Final Held-Out Test Results

Final evaluation was performed on the same 565 held-out test samples for all six configurations.

| Method | mIoU | Adj F1 | Compactness | BVR | Connected-region count MAE |
|---|---:|---:|---:|---:|---:|
| U-Net baseline | 0.375276 | 0.190705 | 0.507777 | 0.000000 | 2.792920 |
| U-Net + morphology | **0.381116** | **0.193860** | 0.629373 | 0.000565 | 2.955752 |
| U-Net + hill-climbing | 0.291573 | 0.192182 | 0.550903 | 0.135012 | 3.725664 |
| cGAN baseline | 0.362964 | 0.191560 | 0.507818 | 0.000001 | **2.545133** |
| cGAN + morphology | 0.367379 | 0.189786 | **0.645706** | 0.000533 | 2.646018 |
| cGAN + hill-climbing | 0.280923 | 0.185532 | 0.558177 | 0.135018 | 3.392920 |

The authoritative per-sample files are stored in `evidence/metrics/`.

## Main Findings

The U-Net baseline achieved higher mIoU than the cGAN baseline, while the cGAN baseline achieved lower connected-region count MAE. Their baseline adjacency F1 and compactness values were similar.

Morphology produced the most balanced refinement outcome. U-Net with morphology achieved the highest mIoU and adjacency F1 across the six final configurations and substantially increased compactness while keeping BVR very low. However, morphology slightly increased connected-region count MAE for both models.

cGAN with morphology achieved the highest compactness. Compactness-based hill-climbing increased compactness but reduced mIoU, substantially increased boundary violation and increased connected-region count MAE under the corrected final metric.

These results show why floor-plan generation should be evaluated using several complementary measures rather than a single semantic score.

## Condition Sensitivity

A supplementary test applied encoded counts from 3 to 7 to the same support mask. Every count produced a different semantic prediction, showing that the numerical condition influences the output. The predicted connected-region count did not change monotonically and did not reliably match the supplied value. The count is therefore treated as a soft conditioning signal rather than a hard architectural constraint.

Repeated inference with the same input was deterministic under the tested setup.

## Documentation

```text
RUNNING_PROJECT.md                       - commands for running the final project
EXPERIMENTS.md                           - development history and final experiment record
preprocessing_evidence_audit.md          - verified preprocessing evidence
condition_sensitivity_evidence_audit.md  - condition, determinism and runtime audit
evidence/README.md                       - authoritative evidence index
```

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
- corrected held-out count evaluation
- final six-configuration evaluation
- preprocessing evidence audit
- condition-sensitivity and determinism checks
- manual semantic floor-plan generation

## Future Work

Future work may include richer room-type and adjacency conditioning, instance-aware room representations, building-grouped dataset splits, repeated training across random seeds, boundary-aware or multi-objective refinement, higher-resolution prediction, raster-to-vector conversion and professional architectural assessment.

## License

This repository is intended for academic research purposes. CubiCasa5K remains subject to its own dataset licence and attribution requirements.

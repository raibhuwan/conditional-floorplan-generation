# Experiment Log

This document records the main training experiments, evaluation results and refinement observations for the conditional semantic floor plan generation framework. The project focuses on generating two-dimensional semantic floor plan masks from a building outline and room-count condition, then evaluating the generated layouts using both semantic and spatial metrics.

The experiments cover:
- conditional U-Net baseline training,
- Pix2Pix-style conditional GAN training,
- dataset scaling behaviour,
- held-out test evaluation,
- morphology and hill-climbing refinement,
- and final manual generation from boundary and room-count input.

## Metric Note

Validation IoU values reported during training represent epoch-based validation performance used for checkpoint selection.

Final mIoU values are obtained from the post-training held-out test evaluation pipeline. These values are therefore not directly equivalent to training-stage validation IoU because the final evaluation is performed on unseen test samples and includes additional spatial metrics.

## Experiment Summary

| Experiment | Purpose | Clean Dataset Size | Best Validation IoU |
|---|---|---:|---:|
| EXP-01 | Initial cGAN training | 492 | 0.269 |
| EXP-02 | Stabilised cGAN training | 492 | 0.278 |
| EXP-03 | Increased dataset training | 1004 | 0.319 |
| EXP-04 | Development scaling run | 1966 | 0.373 |
| EXP-05 | Final full high_quality_architectural run | 3761 | 0.4706 U-Net / 0.3860 cGAN |

The early experiments were used to test preprocessing, adversarial stability and dataset scaling. The final results reported for the project are based on EXP-05, which uses the filtered full high_quality_architectural subset of CubiCasa5K.

---

## EXP-01 — Initial cGAN Training

### Configuration

- Requested source SVG samples: 500
- Clean dataset size: 492 samples
- Generator: U-Net
- Discriminator: PatchGAN
- Batch size: 2
- Epochs: 30
- Generator learning rate: 1e-4
- Discriminator learning rate: 5e-5
- Cross-entropy loss weight: 20.0
- Adversarial loss weight: 0.1

### Results

- Best validation IoU: 0.269
- Final training IoU: 0.624
- Final validation IoU: 0.235

### Observations

The discriminator rapidly became overconfident during training. Discriminator loss collapsed close to zero and generator adversarial loss increased, indicating unstable adversarial optimisation. The model showed early semantic learning but had poor generalisation.

---

## EXP-02 — Stabilised cGAN Training

### Configuration

- Requested source SVG samples: 500
- Clean dataset size: 492 samples
- Generator: U-Net
- Discriminator: PatchGAN
- Batch size: 2
- Epochs: 30
- Generator learning rate: 1e-4
- Discriminator learning rate: 1e-5
- Cross-entropy loss weight: 30.0
- Adversarial loss weight: 0.05
- Real label smoothing: 0.9

### Results

- Best validation IoU: 0.278
- Final training IoU: 0.660
- Final validation IoU: 0.243

### Observations

Lowering the discriminator learning rate, increasing the cross-entropy weight and applying label smoothing improved adversarial stability. Validation IoU improved slightly, although overfitting remained visible.

---

## EXP-03 — Increased Dataset Training

### Configuration

- Requested source SVG samples: 1000
- Clean dataset size: 1004 samples
- Generator: U-Net
- Discriminator: PatchGAN
- Batch size: 2
- Epochs: 30
- Generator learning rate: 1e-4
- Discriminator learning rate: 1e-5
- Cross-entropy loss weight: 30.0
- Adversarial loss weight: 0.05
- Real label smoothing: 0.9

### Results

- Best validation IoU: 0.319
- Final training IoU: 0.740
- Final validation IoU: 0.313

### Observations

Increasing the dataset size improved validation performance and generalisation. Training became more stable than the earlier 492-sample runs, although overfitting was still present.

---

## EXP-04 — Development Scaling Run on 1966 Samples

EXP-04 was used as a controlled development run to test training stability, evaluation scripts and refinement methods before moving to the full high_quality_architectural subset.

### Dataset and Split

- Requested source SVG samples: 2000
- Clean dataset size after filtering: 1966 samples
- Split: train 1376 / validation 294 / test 296
- MAX_COUNT: 18

### Training Configuration

- Generator: U-Net
- Discriminator: PatchGAN
- Batch size: 2
- Epochs: 30
- Generator learning rate: 1e-4
- Discriminator learning rate: 1e-5
- Cross-entropy loss weight: 30.0
- Adversarial loss weight: 0.05
- Real label smoothing: 0.9

### Training Results

- Best validation IoU: 0.373
- Best checkpoint epoch: 21
- Final training IoU: 0.753
- Final validation IoU: 0.360

### Corrected Held-Out Test Evaluation

| Method | mIoU | Adj F1 | Compactness | BVR | RC-MAE |
|---|---:|---:|---:|---:|---:|
| U-Net baseline | 0.344 | 0.196 | 0.508 | 0.000 | 8.814 |
| U-Net + morphology | 0.348 | 0.204 | 0.629 | 0.001 | 7.872 |
| U-Net + hill-climbing | 0.267 | 0.188 | 0.555 | 0.130 | 4.973 |
| cGAN baseline | 0.330 | 0.200 | 0.519 | 0.000 | 9.209 |
| cGAN + morphology | 0.336 | 0.200 | 0.650 | 0.001 | 8.412 |
| cGAN + hill-climbing | 0.257 | 0.197 | 0.565 | 0.129 | 5.358 |

### Observations

The 1966-sample run showed that morphology improved compactness and slightly improved semantic performance for both U-Net and cGAN outputs. Hill-climbing reduced room-count error but caused a large drop in mIoU and increased boundary violation. These findings helped define the final evaluation approach used in EXP-05.

---

## EXP-05 — Final Full high_quality_architectural Dataset Run

EXP-05 is the final experiment used for the main project results. It uses the filtered full high_quality_architectural subset of CubiCasa5K.

### Dataset Preprocessing

- Dataset source: CubiCasa5K high_quality_architectural category
- Processed folder: `data/processed_npz_full`
- Total processed NPZ files: 4566
- Stored keys: `sem`, `outline`, `room_count`, `sample_id`
- Semantic mask shape: 256 x 256
- Outline mask shape: 256 x 256

### Filtering

- Filtering rule: keep samples with `room_count >= 3`
- Additional checks: valid semantic mask, valid outline mask, valid class range and non-empty masks
- Clean folder: `data/processed_npz_clean_full`
- Total clean samples: 3761
- Dropped samples: 805
- Drop reason: `room_count_below_min`

### Room-Count Statistics

- Minimum room count: 3
- Maximum room count: 32
- Mean room count: 6.254985376229726
- MAX_COUNT used for conditioning: 32

### Train / Validation / Test Split

- Split file: `outputs/splits/split_seed42_full.json`
- Total samples: 3761
- Training samples: 2632
- Validation samples: 564
- Test samples: 565
- Seed: 42

The training set was used for model learning, the validation set was used for checkpoint selection and tuning, and the held-out test set was reserved for final evaluation on unseen floor plans.

---

## EXP-05 — U-Net Training

### Command

```bash
python -m train_unet \
  --data_dir data/processed_npz_clean_full \
  --split_path outputs/splits/split_seed42_full.json \
  --max_count 32 \
  --batch_size 4 \
  --epochs 30 \
  --lr 1e-3 \
  --seed 42 \
  --checkpoint_name unet_base16_best.pt
```

### Configuration

- Model: U-Net baseline
- Device: MPS
- Dataset: `data/processed_npz_clean_full`
- Split file: `outputs/splits/split_seed42_full.json`
- Batch size: 4
- Epochs: 30
- Learning rate: 1e-3
- Seed: 42
- Checkpoint: `outputs/checkpoints/unet_base16_best.pt`

### Results

- Best validation IoU: 0.4706
- Best epoch: 20
- Final epoch training loss: 0.1088
- Final epoch training IoU: 0.766
- Final epoch validation loss: 0.7020
- Final epoch validation IoU: 0.439

### Observation

The U-Net achieved the strongest validation performance. Training IoU continued to increase after epoch 20 while validation IoU decreased, so the epoch 20 checkpoint was used for final evaluation.

---

## EXP-05 — cGAN Training

### Command

```bash
python -m train_cgan \
  --data_dir data/processed_npz_clean_full \
  --split_path outputs/splits/split_seed42_full.json \
  --max_count 32 \
  --batch_size 4 \
  --epochs 30 \
  --lr_g 1e-4 \
  --lr_d 1e-5 \
  --lambda_ce 30.0 \
  --lambda_gan 0.05 \
  --seed 42 \
  --checkpoint_name cgan_unet_patchgan_best.pt
```

### Configuration

- Model: Pix2Pix-style conditional GAN
- Generator: U-Net
- Discriminator: PatchGAN
- Device: MPS
- Dataset: `data/processed_npz_clean_full`
- Split file: `outputs/splits/split_seed42_full.json`
- Batch size: 4
- Epochs: 30
- Generator learning rate: 1e-4
- Discriminator learning rate: 1e-5
- Cross-entropy loss weight: 30.0
- Adversarial loss weight: 0.05
- Seed: 42
- Checkpoint: `outputs/checkpoints/cgan_unet_patchgan_best.pt`

### Results

- Best validation IoU: 0.3860
- Best epoch: 20
- Final epoch discriminator loss: 0.1728
- Final epoch generator loss: 3.4959
- Final epoch cross-entropy loss: 0.1042
- Final epoch GAN loss: 7.3808
- Final epoch training IoU: 0.749
- Final epoch validation CE: 0.7073
- Final epoch validation IoU: 0.351

### Observation

The cGAN trained stably with the reduced discriminator learning rate and low adversarial loss weight. However, its best validation IoU remained lower than the U-Net baseline, indicating that the adversarial component did not improve semantic validation performance in the final full-dataset run.

---

## EXP-05 — Final Held-Out Test Results

The following table reports final performance on the 565-sample held-out test set.

| Method | mIoU | Adj F1 | Compactness | BVR | RC-MAE |
|---|---:|---:|---:|---:|---:|
| U-Net baseline | 0.375 | 0.191 | 0.508 | 0.000 | 8.287 |
| U-Net + morphology | 0.381 | 0.194 | 0.629 | 0.001 | 7.156 |
| U-Net + hill-climbing | 0.292 | 0.192 | 0.551 | 0.135 | 4.361 |
| cGAN baseline | 0.363 | 0.192 | 0.508 | 0.000 | 9.377 |
| cGAN + morphology | 0.367 | 0.190 | 0.646 | 0.001 | 8.550 |
| cGAN + hill-climbing | 0.281 | 0.186 | 0.558 | 0.135 | 5.513 |

### Output CSV Files

- U-Net baseline: `outputs/metrics_unet_full.csv`
- U-Net + morphology: `outputs/metrics_unet_morphology_full.csv`
- U-Net + hill-climbing: `outputs/metrics_unet_hillclimb_full.csv`
- cGAN baseline: `outputs/metrics_cgan_full.csv`
- cGAN + morphology: `outputs/metrics_cgan_morphology_full.csv`
- cGAN + hill-climbing: `outputs/metrics_cgan_hillclimb_full.csv`

---

## Final Findings

### U-Net vs cGAN

The U-Net baseline performed better than the cGAN on mIoU and room-count error in the final held-out test evaluation. The cGAN had almost identical adjacency and compactness values to the U-Net baseline, but it did not outperform the simpler U-Net model overall.

### Morphology Refinement

Morphology refinement improved both U-Net and cGAN outputs. It increased compactness substantially and improved room-count MAE while preserving semantic performance reasonably well. U-Net + morphology achieved the best overall balance, with the highest mIoU, highest adjacency F1, strong compactness and very low boundary violation.

### Hill-Climbing Refinement

Hill-climbing reduced room-count MAE for both models, but this came at the cost of semantic accuracy and boundary consistency. Both hill-climbing results had lower mIoU and much higher boundary violation rates. This shows that improving one spatial metric can degrade other aspects of layout quality.

### Multi-Metric Evaluation

The final results show why multi-metric evaluation is needed. If only room-count error were considered, hill-climbing would appear successful. However, mIoU and boundary violation rate show that this improvement is achieved through a loss of semantic and spatial consistency.

### Final Selected Method

The best overall method for final generation is U-Net with morphology refinement. This method is used in the manual generation script because it provides the strongest overall balance across semantic and spatial metrics.

---

## Manual Generation Script

The final user-facing generation script is:

```text
scripts/generate_floorplan.py
```

It generates a colourised 2D semantic floor plan mask from a boundary image and a room-count condition.

### Example Command

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

A helper script is also provided to export a valid boundary image from the processed dataset:

```text
scripts/export_boundary_sample.py
```

### Output Scope

The generated output is a semantic floor plan mask, not a complete architectural drawing. It does not include detailed architectural symbols such as doors, windows, dimensions or room labels. Producing complete architectural drawings is outside the scope of this project and should be treated as future work.

---

## Report Use Notes

- EXP-01 to EXP-04 should be described as development and scaling experiments.
- EXP-05 should be used as the final experiment for the main results chapter.
- The corrected 1966-sample held-out table should be used if EXP-04 is discussed.
- Older inconsistent results where cGAN mIoU was recorded as 0.546 and cGAN + morphology as 0.567 should not be used in the report.
- The final project conclusion should state that boundary and room-count conditioning can generate semantic layout masks, but this conditioning alone is not sufficient to guarantee architecturally complete or fully coherent floor plans.
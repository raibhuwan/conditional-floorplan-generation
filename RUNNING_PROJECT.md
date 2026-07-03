

# Running the Project

This file records the main commands used to run the conditional semantic floor plan generation project. The project generates two-dimensional semantic floor plan masks from a building outline and room-count condition. The final selected generation route is the U-Net model with optional morphology refinement.

## 1. Project Assumptions

Run all commands from the project root folder:

```bash
cd conditional-floorplan-generation
```

The final experiment uses the filtered `high_quality_architectural` subset of CubiCasa5K.

Main final paths:

```text
Processed full dataset: data/processed_npz_full
Clean filtered dataset: data/processed_npz_clean_full
Split file: outputs/splits/split_seed42_full.json
U-Net checkpoint: outputs/checkpoints/unet_base16_best.pt
cGAN checkpoint: outputs/checkpoints/cgan_unet_patchgan_best.pt
MAX_COUNT: 32
```

## 2. Dataset Preprocessing

The full high-quality CubiCasa5K subset was preprocessed into NPZ files containing the semantic mask, outline mask, room-count value and sample identifier.

Final processed output:

```text
data/processed_npz_full
```

The full preprocessing command depends on the local CubiCasa5K folder path. Use the preprocessing command configured for the local dataset location.

## 3. Dataset Filtering

Filter the processed samples and keep only layouts with at least three rooms:

```bash
python scripts/filter_dataset.py \
  --src_dir data/processed_npz_full \
  --dst_dir data/processed_npz_clean_full \
  --min_rooms 3 \
  --clear
```

Final filtering result:

```text
Total processed samples: 4566
Clean samples kept: 3761
Dropped samples: 805
Drop reason: room_count_below_min
```

## 4. Compute Maximum Room Count

Compute the maximum room count in the filtered dataset. This value is used to normalise the room-count conditioning channel.

```bash
python scripts/compute_max_count.py \
  --data_dir data/processed_npz_clean_full
```

Final value:

```text
MAX_COUNT = 32
```

## 5. Create Train / Validation / Test Split

Create the fixed split used in the final experiment:

```bash
python -m scripts.create_split \
  --data_dir data/processed_npz_clean_full \
  --out outputs/splits/split_seed42_full.json
```

Final split:

```text
Total: 3761
Train: 2632
Validation: 564
Test: 565
Seed: 42
```

The training set is used for model learning, the validation set is used for checkpoint selection and tuning, and the held-out test set is used only for final evaluation.

## 6. Train the U-Net Baseline

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

Final U-Net result:

```text
Best validation IoU: 0.4706
Best epoch: 20
Checkpoint: outputs/checkpoints/unet_base16_best.pt
```

## 7. Train the Pix2Pix-Style cGAN

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

Final cGAN result:

```text
Best validation IoU: 0.3860
Best epoch: 20
Checkpoint: outputs/checkpoints/cgan_unet_patchgan_best.pt
```

## 8. Evaluate Final Methods

All final evaluations are run on the held-out test set of 565 unseen samples.

### 8.1 U-Net Baseline

```bash
python -m scripts.evaluate_metrics \
  --data_dir data/processed_npz_clean_full \
  --split_path outputs/splits/split_seed42_full.json \
  --ckpt_path outputs/checkpoints/unet_base16_best.pt \
  --max_count 32 \
  --out_csv outputs/metrics_unet_full.csv
```

### 8.2 cGAN Baseline

```bash
python -m scripts.evaluate_cgan \
  --data_dir data/processed_npz_clean_full \
  --split_path outputs/splits/split_seed42_full.json \
  --ckpt_path outputs/checkpoints/cgan_unet_patchgan_best.pt \
  --max_count 32 \
  --out_csv outputs/metrics_cgan_full.csv
```

### 8.3 U-Net + Morphology

```bash
python -m scripts.evaluate_unet_morphology \
  --data_dir data/processed_npz_clean_full \
  --split_path outputs/splits/split_seed42_full.json \
  --ckpt_path outputs/checkpoints/unet_base16_best.pt \
  --max_count 32 \
  --out_csv outputs/metrics_unet_morphology_full.csv \
  --kernel_size 3 \
  --min_area 30
```

### 8.4 cGAN + Morphology

```bash
python -m scripts.evaluate_cgan_morphology \
  --data_dir data/processed_npz_clean_full \
  --split_path outputs/splits/split_seed42_full.json \
  --ckpt_path outputs/checkpoints/cgan_unet_patchgan_best.pt \
  --max_count 32 \
  --out_csv outputs/metrics_cgan_morphology_full.csv \
  --kernel_size 3 \
  --min_area 30
```

### 8.5 U-Net + Hill-Climbing

```bash
python -m scripts.evaluate_unet_hillclimb \
  --data_dir data/processed_npz_clean_full \
  --split_path outputs/splits/split_seed42_full.json \
  --ckpt_path outputs/checkpoints/unet_base16_best.pt \
  --max_count 32 \
  --out_csv outputs/metrics_unet_hillclimb_full.csv \
  --kernel_size 3 \
  --iterations 3
```

### 8.6 cGAN + Hill-Climbing

```bash
python -m scripts.evaluate_cgan_hillclimb \
  --data_dir data/processed_npz_clean_full \
  --split_path outputs/splits/split_seed42_full.json \
  --ckpt_path outputs/checkpoints/cgan_unet_patchgan_best.pt \
  --max_count 32 \
  --out_csv outputs/metrics_cgan_hillclimb_full.csv \
  --kernel_size 3 \
  --iterations 3
```

## 9. Final Held-Out Test Results

| Method | mIoU | Adj F1 | Compactness | BVR | RC-MAE |
|---|---:|---:|---:|---:|---:|
| U-Net baseline | 0.375 | 0.191 | 0.508 | 0.000 | 8.287 |
| U-Net + morphology | 0.381 | 0.194 | 0.629 | 0.001 | 7.156 |
| U-Net + hill-climbing | 0.292 | 0.192 | 0.551 | 0.135 | 4.361 |
| cGAN baseline | 0.363 | 0.192 | 0.508 | 0.000 | 9.377 |
| cGAN + morphology | 0.367 | 0.190 | 0.646 | 0.001 | 8.550 |
| cGAN + hill-climbing | 0.281 | 0.186 | 0.558 | 0.135 | 5.513 |

The best overall method is U-Net with morphology refinement because it gives the strongest balance across semantic accuracy, adjacency similarity, compactness and boundary consistency.

## 10. Export a Boundary Image for Manual Generation

A boundary image can be exported from an existing test sample using:

```bash
python -m scripts.export_boundary_sample \
  --data_dir data/processed_npz_clean_full \
  --split_path outputs/splits/split_seed42_full.json \
  --split test \
  --sample_index 0 \
  --max_count 32 \
  --out_path inputs/boundary.png
```

This creates:

```text
inputs/boundary.png
```

## 11. Generate a Semantic Floor Plan from Boundary + Room Count

Use the final selected model route, U-Net with morphology refinement:

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

Output:

```text
outputs/generated/floorplan_6rooms.png
outputs/generated/floorplan_6rooms.npy
```

The PNG file is a colourised semantic floor plan mask. The NPY file stores the raw class-id mask.

## 12. Output Scope

The project generates semantic floor plan masks rather than complete architectural drawings. The output does not include detailed architectural symbols such as doors, windows, dimensions or room labels.

Producing a clean architectural drawing from the generated semantic mask would require additional rendering, vectorisation or symbol-generation stages, which are outside the scope of this project.
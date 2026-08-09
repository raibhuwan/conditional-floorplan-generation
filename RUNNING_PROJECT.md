# Running the Project

This file records the main commands used to run the final conditional semantic floor-plan generation project. The implemented model receives a filled binary floor-plan support mask and an encoded connected-region count, then predicts a 256 x 256 semantic floor-plan mask.

Historical implementation names such as `outline` and `room_count` are retained in filenames and command-line arguments. In the final dissertation, these are described more precisely as the **binary support condition** and **encoded connected-region count condition**.

## 1. Project Assumptions

Run all commands from the project root:

```bash
cd conditional-floorplan-generation
```

Final experiment paths:

```text
Processed full dataset: data/processed_npz_full
Clean filtered dataset: data/processed_npz_clean_full
Split file: outputs/splits/split_seed42_full.json
U-Net checkpoint: outputs/checkpoints/unet_base16_best.pt
cGAN checkpoint: outputs/checkpoints/cgan_unet_patchgan_best.pt
MAX_COUNT: 32
```

Verified final evidence is stored under:

```text
evidence/training/
evidence/metrics/
evidence/preprocessing/
evidence/condition_sensitivity/
evidence/splits/
```

## 2. Dataset Preprocessing

The full CubiCasa5K `high_quality_architectural` subset was preprocessed into NPZ files containing:

```text
sem
outline
room_count
sample_id
```

In the final terminology:

- `sem` is the nine-class semantic target mask;
- `outline` stores the filled binary floor-plan support mask;
- `room_count` stores the encoded connected-region count; and
- `sample_id` preserves sample traceability.

Final processed output:

```text
data/processed_npz_full
```

Verified preprocessing totals:

```text
Successfully processed samples: 4566
Retained after filtering: 3761
Excluded: 805
```

## 3. Dataset Filtering

Retain samples with an encoded connected-region count of at least three:

```bash
python scripts/filter_dataset.py \
  --src_dir data/processed_npz_full \
  --dst_dir data/processed_npz_clean_full \
  --min_rooms 3 \
  --clear
```

The argument is named `--min_rooms` for historical compatibility, but the stored value is an encoded connected-region count rather than a manually verified architectural room count.

Final exclusion distribution:

```text
Count 0: 243
Count 1: 187
Count 2: 375
Total excluded: 805
Total retained: 3761
```

## 4. Compute Maximum Encoded Count

```bash
python scripts/compute_max_count.py \
  --data_dir data/processed_npz_clean_full
```

Final value:

```text
MAX_COUNT = 32
```

This value is used to normalise the count-conditioning channel.

## 5. Create Train / Validation / Test Split

```bash
python -m scripts.create_split \
  --data_dir data/processed_npz_clean_full \
  --out outputs/splits/split_seed42_full.json \
  --seed 42
```

Final split:

```text
Total: 3761
Training: 2632
Validation: 564
Test: 565
Seed: 42
```

Training data are used for parameter learning, validation data for checkpoint selection, and the test set only for final held-out evaluation.

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

Final selected checkpoint evidence:

```text
Selected epoch: 20
Common sample-level validation mIoU: 0.399831
Training mIoU at epoch 20: 0.621130
Final training mIoU: 0.736845
Final validation mIoU: 0.369020
Approximate total training time: 54.0 minutes
```

The training history also contains an older validation aggregation that produced a higher logged value around 0.4706 at the selected checkpoint. The final dissertation comparison uses the common sample-level validation procedure stored in `evidence/training/` so that U-Net and cGAN are compared consistently.

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

Final selected checkpoint evidence:

```text
Selected epoch: 20
Common sample-level validation mIoU: 0.386000
Training mIoU at epoch 20: 0.659419
Final training mIoU: 0.748921
Final validation mIoU: 0.351045
Approximate total training time: 101.7 minutes
```

## 8. Evaluate the Final Six Configurations

All final evaluations use the same 565 held-out test samples. The output filenames below use the historical `room_count_fixed` label because those files correspond directly to the authoritative evidence copied into `evidence/metrics/`.

### 8.1 U-Net Baseline

```bash
python -m scripts.evaluate_metrics \
  --data_dir data/processed_npz_clean_full \
  --split_path outputs/splits/split_seed42_full.json \
  --ckpt_path outputs/checkpoints/unet_base16_best.pt \
  --max_count 32 \
  --out_csv outputs/metrics_unet_room_count_fixed.csv
```

### 8.2 cGAN Baseline

```bash
python -m scripts.evaluate_cgan_metrics \
  --data_dir data/processed_npz_clean_full \
  --split_path outputs/splits/split_seed42_full.json \
  --ckpt_path outputs/checkpoints/cgan_unet_patchgan_best.pt \
  --max_count 32 \
  --out_csv outputs/metrics_cgan_room_count_fixed.csv
```

### 8.3 U-Net + Morphology

```bash
python -m scripts.evaluate_unet_morphology \
  --data_dir data/processed_npz_clean_full \
  --split_path outputs/splits/split_seed42_full.json \
  --ckpt_path outputs/checkpoints/unet_base16_best.pt \
  --max_count 32 \
  --out_csv outputs/metrics_unet_morphology_room_count_fixed.csv \
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
  --out_csv outputs/metrics_cgan_morphology_room_count_fixed.csv \
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
  --out_csv outputs/metrics_unet_hillclimb_room_count_fixed.csv \
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
  --out_csv outputs/metrics_cgan_hillclimb_room_count_fixed.csv \
  --kernel_size 3 \
  --iterations 3
```

## 9. Final Held-Out Test Results

| Method | mIoU | Adj F1 | Compactness | BVR | Connected-region count MAE |
|---|---:|---:|---:|---:|---:|
| U-Net baseline | 0.375276 | 0.190705 | 0.507777 | 0.000000 | 2.792920 |
| U-Net + morphology | **0.381116** | **0.193860** | 0.629373 | 0.000565 | 2.955752 |
| U-Net + hill-climbing | 0.291573 | 0.192182 | 0.550903 | 0.135012 | 3.725664 |
| cGAN baseline | 0.362964 | 0.191560 | 0.507818 | 0.000001 | **2.545133** |
| cGAN + morphology | 0.367379 | 0.189786 | **0.645706** | 0.000533 | 2.646018 |
| cGAN + hill-climbing | 0.280923 | 0.185532 | 0.558177 | 0.135018 | 3.392920 |

The final selected generation route is U-Net with morphology because it achieved the highest mIoU and adjacency F1, substantially improved compactness relative to the U-Net baseline, and kept boundary violation very low. Its connected-region count MAE is slightly worse than the U-Net baseline, so the selection is based on the overall metric balance rather than count agreement alone.

## 10. Export a Support Image for Manual Generation

A support image can be exported from an existing test sample using:

```bash
python -m scripts.export_boundary_sample \
  --data_dir data/processed_npz_clean_full \
  --split_path outputs/splits/split_seed42_full.json \
  --split test \
  --sample_index 0 \
  --max_count 32 \
  --out_path inputs/boundary.png
```

The historical script and output filename use the word `boundary`, but the exported image used by the final model is the filled binary floor-plan support condition.

## 11. Generate a Semantic Floor Plan

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

The command-line argument `--room_count` supplies the encoded connected-region count condition. It should not be interpreted as guaranteed architectural room-count control.

The manual generation script clips the saved output to the supplied support mask. This differs from the final evaluation pipeline, where predictions remain unclipped so that boundary violation can be measured.

## 12. Supplementary Condition-Sensitivity Test

The evidence script under `scripts/evidence/` tests the same support condition with encoded counts from 3 to 7, repeated inference and local runtime. Verified results are stored under:

```text
evidence/condition_sensitivity/
```

The test shows that changing the encoded count changes the semantic prediction, but the predicted connected-region count is not monotonic and does not reliably equal the requested value.

## 13. Output Scope

The system generates semantic floor-plan masks rather than complete architectural drawings. Outputs do not include verified room instances, doors, windows, room labels, dimensions, furniture, structural calculations, building-service information or CAD-ready geometry.

For final dissertation numbers, use the files under `evidence/` rather than older metric CSVs retained in `outputs/` for development history.

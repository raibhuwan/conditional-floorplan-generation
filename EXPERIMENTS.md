# Experiment Log

This document records the main development and final experiments for the conditional semantic floor-plan generation framework. The implemented system generates two-dimensional semantic masks from two lightweight conditions: a filled binary floor-plan support mask and an encoded connected-region count. The implementation retains historical variable names such as `outline` and `room_count`, but the final dissertation uses the more precise terminology because the count is derived from connected semantic regions rather than manually labelled architectural room instances.

The experiments cover:

- conditional U-Net baseline training;
- Pix2Pix-style conditional GAN training;
- dataset scaling behaviour;
- held-out test evaluation;
- morphology and hill-climbing refinement;
- condition-sensitivity checks; and
- final manual generation from a support image and encoded count condition.

## Authoritative Evidence Note

The final dissertation results are based on EXP-05 and the verified files under `evidence/`. The six files in `evidence/metrics/` are the authoritative held-out test results for all 565 test samples.

The metric filenames retain the historical term `room_count_fixed` for traceability. In the dissertation, this measure is described as **connected-region count Mean Absolute Error (MAE)**. It compares the supplied encoded count condition with the connected-component count obtained from the combined non-background, non-wall prediction mask using eight-connectivity.

Development experiments EXP-01 to EXP-04 are retained for project history. Their logged validation values and earlier count metrics should not be substituted for the final EXP-05 evidence.

## Experiment Summary

| Experiment | Purpose | Clean Dataset Size | Validation evidence |
|---|---|---:|---:|
| EXP-01 | Initial cGAN training | 492 | Logged best validation IoU: 0.269 |
| EXP-02 | Stabilised cGAN training | 492 | Logged best validation IoU: 0.278 |
| EXP-03 | Increased dataset training | 1,004 | Logged best validation IoU: 0.319 |
| EXP-04 | Development scaling run | 1,966 | Logged best validation IoU: 0.373 |
| EXP-05 | Final full `high_quality_architectural` run | 3,761 | Common validation mIoU: 0.399831 U-Net / 0.386000 cGAN |

The early experiments were used to test preprocessing, adversarial stability and dataset scaling. The final dissertation results are based on EXP-05.

---

## EXP-01 - Initial cGAN Training

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

- Best logged validation IoU: 0.269
- Final training IoU: 0.624
- Final validation IoU: 0.235

### Observation

The discriminator became overconfident early in training. Discriminator loss moved close to zero while the generator adversarial loss increased, indicating unstable adversarial optimisation. The run showed early semantic learning but weak validation performance.

---

## EXP-02 - Stabilised cGAN Training

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

- Best logged validation IoU: 0.278
- Final training IoU: 0.660
- Final validation IoU: 0.243

### Observation

Reducing the discriminator learning rate, increasing the cross-entropy weight and applying one-sided label smoothing improved adversarial stability. Validation performance improved slightly, although overfitting remained visible.

---

## EXP-03 - Increased Dataset Training

### Configuration

- Requested source SVG samples: 1,000
- Clean dataset size: 1,004 samples
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

- Best logged validation IoU: 0.319
- Final training IoU: 0.740
- Final validation IoU: 0.313

### Observation

Increasing the dataset size improved validation performance and supported the decision to move to a larger final dataset.

---

## EXP-04 - Development Scaling Run on 1,966 Samples

EXP-04 was a development run used to test training stability, evaluation scripts and refinement methods before the final full-dataset experiment.

### Dataset and Split

- Requested source SVG samples: 2,000
- Clean dataset size after filtering: 1,966 samples
- Split: 1,376 training / 294 validation / 296 test
- `MAX_COUNT`: 18

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

### Logged Training Results

- Best validation IoU: 0.373
- Best checkpoint epoch: 21
- Final training IoU: 0.753
- Final validation IoU: 0.360

### Historical Held-Out Evaluation

The following values are retained only as development evidence. The connected-count calculation used during this stage predates the final corrected EXP-05 metric implementation, so the RC-MAE values below should **not** be compared directly with the final dissertation results.

| Method | mIoU | Adj F1 | Compactness | BVR | Historical RC-MAE |
|---|---:|---:|---:|---:|---:|
| U-Net baseline | 0.344 | 0.196 | 0.508 | 0.000 | 8.814 |
| U-Net + morphology | 0.348 | 0.204 | 0.629 | 0.001 | 7.872 |
| U-Net + hill-climbing | 0.267 | 0.188 | 0.555 | 0.130 | 4.973 |
| cGAN baseline | 0.330 | 0.200 | 0.519 | 0.000 | 9.209 |
| cGAN + morphology | 0.336 | 0.200 | 0.650 | 0.001 | 8.412 |
| cGAN + hill-climbing | 0.257 | 0.197 | 0.565 | 0.129 | 5.358 |

### Observation

The development run showed that refinement could change semantic and geometric metrics in different directions. This motivated the final multi-metric evaluation used in EXP-05. Final conclusions should be drawn from EXP-05 rather than these development-stage count values.

---

## EXP-05 - Final Full `high_quality_architectural` Dataset Run

EXP-05 is the final experiment used in the dissertation.

### Dataset Preprocessing

- Dataset source: CubiCasa5K `high_quality_architectural` category
- Processed folder: `data/processed_npz_full`
- Successfully processed floor-level samples: 4,566
- Stored keys: `sem`, `outline`, `room_count`, `sample_id`
- Semantic mask shape: 256 x 256
- Support-mask shape: 256 x 256

The stored key `outline` contains the filled binary floor-plan support mask used by the final model. The stored key `room_count` contains the encoded connected-region count.

### Filtering

- Filtering rule: retain samples with encoded connected-region count >= 3
- Clean folder: `data/processed_npz_clean_full`
- Retained samples: 3,761
- Excluded samples: 805
- Excluded with count 0: 243
- Excluded with count 1: 187
- Excluded with count 2: 375

The threshold is an operational preprocessing heuristic rather than an architectural rule.

### Encoded Count Statistics

- Minimum retained encoded count: 3
- Maximum encoded count: 32
- `MAX_COUNT` used for conditioning: 32

### Train / Validation / Test Split

- Split file: `outputs/splits/split_seed42_full.json`
- Total retained samples: 3,761
- Training samples: 2,632
- Validation samples: 564
- Test samples: 565
- Seed: 42

The training set was used for model learning, the validation set for checkpoint selection, and the held-out test set for final evaluation.

---

## EXP-05 - U-Net Training

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

### Final Evidence

- Selected checkpoint epoch: 20
- Common sample-level validation mIoU at epoch 20: 0.399831
- Training mIoU at epoch 20: 0.621130
- Final epoch training mIoU: 0.736845
- Final epoch validation mIoU: 0.369020
- Approximate 30-epoch training time: 54.0 minutes

The training script historically logged a different validation aggregation and produced a value of approximately 0.4706 at the selected checkpoint. For the dissertation, both models were re-evaluated using the same common sample-level validation mIoU procedure. The common values in `evidence/training/training_checkpoint_summary.csv` are therefore the values used for the final comparison.

---

## EXP-05 - Pix2Pix-Style cGAN Training

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

### Final Evidence

- Selected checkpoint epoch: 20
- Common sample-level validation mIoU at epoch 20: 0.386000
- Training mIoU at epoch 20: 0.659419
- Final epoch training mIoU: 0.748921
- Final epoch validation mIoU: 0.351045
- Approximate 30-epoch training time: 101.7 minutes

The cGAN remained below the U-Net on the common validation mIoU comparison despite achieving higher training mIoU.

---

## EXP-05 - Final Held-Out Test Results

The following table contains the authoritative mean results from the six files in `evidence/metrics/`. All configurations were evaluated on the same 565 held-out test samples.

| Method | mIoU | Adj F1 | Compactness | BVR | Connected-region count MAE |
|---|---:|---:|---:|---:|---:|
| U-Net baseline | 0.375276 | 0.190705 | 0.507777 | 0.000000 | 2.792920 |
| U-Net + morphology | **0.381116** | **0.193860** | 0.629373 | 0.000565 | 2.955752 |
| U-Net + hill-climbing | 0.291573 | 0.192182 | 0.550903 | 0.135012 | 3.725664 |
| cGAN baseline | 0.362964 | 0.191560 | 0.507818 | 0.000001 | **2.545133** |
| cGAN + morphology | 0.367379 | 0.189786 | **0.645706** | 0.000533 | 2.646018 |
| cGAN + hill-climbing | 0.280923 | 0.185532 | 0.558177 | 0.135018 | 3.392920 |

### Authoritative Metric Files

- `evidence/metrics/metrics_unet_room_count_fixed.csv`
- `evidence/metrics/metrics_unet_morphology_room_count_fixed.csv`
- `evidence/metrics/metrics_unet_hillclimb_room_count_fixed.csv`
- `evidence/metrics/metrics_cgan_room_count_fixed.csv`
- `evidence/metrics/metrics_cgan_morphology_room_count_fixed.csv`
- `evidence/metrics/metrics_cgan_hillclimb_room_count_fixed.csv`

---

## Final Findings

### U-Net vs cGAN

The U-Net baseline achieved higher semantic overlap than the cGAN baseline, with mIoU values of 0.375276 and 0.362964 respectively. The cGAN baseline achieved the lower connected-region count MAE, 2.545133 compared with 2.792920 for U-Net. Baseline adjacency F1 and compactness were similar. The two models therefore showed different strengths rather than one model being strongest on every metric.

### Morphology Refinement

Morphology produced the most balanced refinement outcome. For U-Net it increased mIoU, adjacency F1 and compactness, while connected-region count MAE increased from 2.792920 to 2.955752 and BVR increased slightly from 0.000000 to 0.000565. For the cGAN it increased mIoU and compactness, while adjacency F1 decreased slightly, count MAE increased from 2.545133 to 2.646018, and BVR remained very low.

U-Net with morphology achieved the highest mIoU and adjacency F1 across the six final configurations. cGAN with morphology achieved the highest compactness.

### Hill-Climbing Refinement

Compactness-based hill-climbing increased compactness relative to each baseline, but it substantially reduced mIoU and increased boundary violation. Under the corrected connected-region count metric, it also increased count MAE for both models. This demonstrates that optimising compactness alone can weaken other semantic and spatial properties.

### Multi-Metric Evaluation

The final results support multi-metric evaluation. mIoU alone would favour U-Net with morphology, whereas compactness favours cGAN with morphology and connected-region count MAE favours the cGAN baseline. Reporting the metrics separately makes these trade-offs visible.

### Final Selected Generation Route

U-Net with morphology is used as the final manual generation route because it provides the strongest overall balance for the project objective: highest mIoU, highest adjacency F1, substantially improved compactness and very low boundary violation. It is **not** selected because of connected-region count MAE, which is slightly worse than the U-Net baseline and higher than the cGAN baseline.

---

## Supplementary Condition-Sensitivity Evidence

The selected U-Net checkpoint was evaluated using the same support mask with encoded counts from 3 to 7. Every requested count produced a different output, showing that the model responds to the count channel. However, the predicted connected-region count did not change monotonically with the requested condition. This supports the interpretation that the encoded count acts as a conditioning signal rather than a hard architectural room-count constraint.

Repeated inference with the same input produced identical outputs under the tested setup. Runtime evidence is recorded separately in `condition_sensitivity_evidence_audit.md`.

---

## Manual Generation Script

The final generation script is:

```text
scripts/generate_floorplan.py
```

Example:

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

The argument names `--outline_path` and `--room_count` are retained for implementation compatibility. In the final dissertation they correspond to a filled binary support mask and an encoded connected-region count condition.

### Output Scope

The generated output is a semantic floor-plan mask, not a complete architectural drawing. It does not include verified room instances, doors, windows, dimensions, structural calculations, utilities or construction documentation.

---

## Report Use Notes

- EXP-01 to EXP-04 are development history and should not replace final EXP-05 evidence.
- EXP-05 is the source for the main dissertation results.
- Use the common sample-level validation mIoU values 0.399831 for U-Net and 0.386000 for cGAN when describing the selected epoch-20 checkpoints.
- Use the six `evidence/metrics/*room_count_fixed.csv` files for final held-out test results.
- Describe the final count measure as **connected-region count MAE**, not architectural room-count accuracy.
- Do not use the older EXP-05 count-error values around 7-9 from the earlier evaluation files.
- Do not state that morphology or hill-climbing improved final connected-region count MAE; both increased it relative to their corresponding final baselines.
- The final conclusion should state that the support and encoded-count conditions can guide semantic mask generation, but they do not guarantee an exact number of architectural rooms or a construction-ready layout.

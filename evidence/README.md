# Final Dissertation Evidence

This directory contains the verified evidence used for the final evaluation of the conditional semantic floor-plan generation project. Where older development outputs conflict with these files, the evidence directory should be treated as the final source for dissertation numbers.

## Terminology Note

Historical implementation names and filenames use terms such as `outline`, `room_count` and `room_count_fixed`.

In the final dissertation:

- `outline` is described as the **filled binary floor-plan support mask**;
- `room_count` is described as the **encoded connected-region count**; and
- `room_count_error` is reported as **connected-region count Mean Absolute Error (MAE)**.

The count is derived from connected semantic regions and is not a verified architectural room-instance count.

## Dataset Split

`evidence/splits/split_seed42_full.json` contains the fixed seed-42 split:

- training: 2,632 samples;
- validation: 564 samples; and
- test: 565 samples.

The split was created at floor-sample level. It was not grouped by original building, which is reported as a project limitation.

## Training Evidence

`evidence/training/` contains:

- the 30-epoch U-Net training history;
- the 30-epoch Pix2Pix-style cGAN training history;
- the selected-checkpoint summary; and
- the validation mIoU values recorded during checkpoint selection.

The selected checkpoint for both models was epoch 20.

| Model | Selected epoch | Validation mIoU at selected epoch | Training mIoU at epoch 20 | Final training mIoU | Final validation mIoU |
|---|---:|---:|---:|---:|---:|
| U-Net | 20 | 0.399831 | 0.621130 | 0.736845 | 0.369020 |
| Pix2Pix-style cGAN | 20 | 0.386000 | 0.659419 | 0.748921 | 0.351045 |

Approximate total training time from the epoch histories was 54.0 minutes for U-Net and 101.7 minutes for the cGAN.

Both selected checkpoints were obtained at epoch 20 using the validation mIoU calculation recorded during training. This procedure calculates mIoU for each validation batch and averages the batch-level values across the validation partition. The same procedure was used for both final models.

Model checkpoints are not included in this evidence package because checkpoint files are excluded from the review archive.

## Final Held-Out Evaluation

`evidence/metrics/` contains the six authoritative per-sample evaluation files:

- U-Net baseline;
- U-Net with morphology;
- U-Net with hill-climbing;
- cGAN baseline;
- cGAN with morphology; and
- cGAN with hill-climbing.

Each file contains results for all 565 held-out test samples.

Mean results are:

| Method | mIoU | Adj F1 | Compactness | BVR | Connected-region count MAE |
|---|---:|---:|---:|---:|---:|
| U-Net baseline | 0.375276 | 0.190705 | 0.507777 | 0.000000 | 2.792920 |
| U-Net + morphology | **0.381116** | **0.193860** | 0.629373 | 0.000565 | 2.955752 |
| U-Net + hill-climbing | 0.291573 | 0.192182 | 0.550903 | 0.135012 | 3.725664 |
| cGAN baseline | 0.362964 | 0.191560 | 0.507818 | 0.000001 | **2.545133** |
| cGAN + morphology | 0.367379 | 0.189786 | **0.645706** | 0.000533 | 2.646018 |
| cGAN + hill-climbing | 0.280923 | 0.185532 | 0.558177 | 0.135018 | 3.392920 |

The filenames retain the historical term `room_count_fixed` for traceability. The corrected count measure compares the encoded count condition with connected components calculated from one combined non-background, non-wall prediction mask using eight-connectivity.

Class-specific connected components with a minimum area of 30 pixels are used separately for adjacency and compactness evaluation.

### Bootstrap Uncertainty Analysis

`evidence/metrics/bootstrap_uncertainty_summary.csv` records the supplementary paired 95% building-cluster bootstrap intervals reported in Chapter 5. The analysis uses 20,000 resamples across the 548 buildings represented by the 565 held-out test floors.

The analysis is reproduced using:

```text
scripts/evidence/bootstrap_uncertainty.py

## Final Result Interpretation

- U-Net baseline has higher mIoU than the cGAN baseline.
- cGAN baseline has the lowest connected-region count MAE.
- U-Net with morphology has the highest mIoU and adjacency F1.
- cGAN with morphology has the highest compactness.
- Morphology slightly worsens connected-region count MAE for both models.
- Hill-climbing increases compactness but substantially reduces mIoU, increases BVR and increases connected-region count MAE relative to the corresponding baselines.

U-Net with morphology was used for the final manual generation example because it achieved the highest mIoU and adjacency F1, substantially improved compactness relative to the U-Net baseline and retained a very low boundary violation rate. This was a pragmatic project choice rather than evidence that one configuration was universally best across all evaluation measures.

## Preprocessing Evidence

`evidence/preprocessing/` records:

- all 4,566 processed samples;
- the 3,761 retained samples;
- the 805 excluded samples;
- the encoded connected-region count distribution; and
- representative excluded examples.

The exclusion threshold of three connected regions was a practical preprocessing heuristic and is not presented as an architectural standard.

## Condition-Sensitivity Evidence

`evidence/condition_sensitivity/` contains the supplementary test in which the same binary support mask was evaluated with encoded counts from 3 to 7.

The outputs changed when the encoded count changed, but the resulting connected-region count did not change monotonically. Repeated inference with the same input was deterministic. The evidence therefore supports condition sensitivity but not exact architectural room-count control.

## Supporting Scripts

Scripts used to construct supplementary evidence, including the building-cluster bootstrap uncertainty analysis, are stored in:

```text
scripts/evidence/
```

For the final dissertation, use the evidence in this directory rather than older development metric files retained under `outputs/`.

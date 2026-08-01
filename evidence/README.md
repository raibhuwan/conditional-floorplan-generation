# Final Dissertation Evidence

This directory contains the verified evidence used for the final evaluation
of the conditional semantic floor-plan generation project.

## Dataset split

`evidence/splits/split_seed42_full.json` contains the fixed split created
with random seed 42:

- training: 2,632 samples
- validation: 564 samples
- test: 565 samples

The split was created at floor-sample index level. It was not grouped by
original building, which is reported as a project limitation.

## Training evidence

`evidence/training/` contains:

- the 30-epoch U-Net training history;
- the 30-epoch Pix2Pix-style cGAN training history;
- the selected-checkpoint summary;
- a common sample-level validation mIoU comparison.

The selected checkpoint for both models was epoch 20. The common validation
mIoU was 0.399831 for U-Net and 0.38599955 for the cGAN generator.

Model checkpoints are not committed because checkpoint files are excluded
through `.gitignore`.

## Final held-out evaluation

`evidence/metrics/` contains the six authoritative evaluation files:

- U-Net baseline;
- U-Net with morphology;
- U-Net with hill-climbing;
- cGAN baseline;
- cGAN with morphology;
- cGAN with hill-climbing.

Each file contains results for all 565 held-out test samples.

The filenames retain the historical term `room_count_fixed` for traceability.
Within the final dissertation, this measure is described more precisely as
connected-region count Mean Absolute Error. It compares the encoded count
condition with connected components calculated from one combined
non-background, non-wall prediction mask using eight-connectivity.

Class-specific connected components with a minimum area of 30 pixels are
used separately for adjacency and compactness evaluation.

## Preprocessing evidence

`evidence/preprocessing/` records:

- all 4,566 processed samples;
- the 3,761 retained samples;
- the 805 excluded samples;
- the connected-region count distribution;
- representative excluded examples.

The exclusion threshold of three connected regions was a practical
preprocessing heuristic and is not presented as an architectural standard.

## Condition-sensitivity evidence

`evidence/condition_sensitivity/` contains the supplementary test in which
the same binary support mask was evaluated with encoded counts from 3 to 7.

The outputs changed when the encoded count changed, but the resulting
connected-region count did not change monotonically. Repeated inference with
the same input was deterministic. The evidence therefore supports
condition sensitivity but not exact architectural room-count control.

## Supporting scripts

The scripts used to construct supplementary evidence are stored in
`scripts/evidence/`.

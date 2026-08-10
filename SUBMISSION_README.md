# Applied Research Project – Supporting Code and Experimental Evidence

Project: Conditional Semantic Floor Plan Generation Using Deep Learning and Spatial Evaluation

Student: Bhuwan Rai
Student ID: 250176478

## Purpose

This archive contains the source code, experiment documentation and retained evidence supporting the Applied Research Project report.

The dissertation PDF is the primary assessed document. This archive is provided as supplementary implementation and reproducibility evidence.

## Project Documentation

- `README.md` – project overview and final results
- `RUNNING_PROJECT.md` – instructions for preprocessing, training, evaluation and generation
- `EXPERIMENTS.md` – development-stage and final experiment record
- `requirements.txt` – Python package requirements
- `evidence/README.md` – guide to the verified evidence used in the dissertation

## Final Evidence

The `evidence/` directory should be treated as the authoritative source for the final dissertation results.

It contains:

- preprocessing and filtering records;
- the fixed seed-42 train/validation/test split;
- training histories and checkpoint-selection evidence;
- per-sample held-out evaluation results;
- condition-sensitivity evidence; and
- the building-cluster bootstrap uncertainty summary.

The final held-out test set contains 565 floor-level samples.

## Reproduction

The principal commands required to reproduce the final workflow are also provided in Appendix B of the dissertation.

Detailed execution instructions are available in:

`RUNNING_PROJECT.md`

The supplementary bootstrap analysis can be reproduced using:

python scripts/evidence/bootstrap_uncertainty.py

## Terminology

Some historical code arguments and filenames use terms such as `outline`, `room_count` and `room_count_fixed`.

In the final dissertation:

- `outline` refers to the filled binary support mask;
- `room_count` refers to the encoded connected-region count; and
- `room_count_error` is reported as connected-region count Mean Absolute Error.

The encoded count is derived from connected semantic regions and is not a verified architectural room-instance count.

## Large Files

Raw CubiCasa5K data, processed NPZ datasets and model checkpoint files may be excluded from this submission archive because of file-size limitations.

The retained experiment logs, metric files, configurations and scripts provide the evidence used to report the dissertation results.

## Repository

The maintained project repository is:

https://github.com/raibhuwan/conditional-floorplan-generation
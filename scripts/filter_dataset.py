import argparse
import glob
import os
import shutil

import numpy as np


# Check whether one processed NPZ sample satisfies the dataset quality filters.
def is_valid_sample(path, min_rooms):
    # Load the processed sample and allow access to its stored metadata values.
    d = np.load(path, allow_pickle=True)

    # Require all fields needed by model training and later traceability.
    required_keys = {"sem", "outline", "room_count", "sample_id"}
    if not required_keys.issubset(set(d.files)):
        return False, "missing_required_keys"

    # Load the semantic target, filled support mask and encoded
    # connected-region count stored during preprocessing.
    sem = d["sem"]
    outline = d["outline"]
    room_count = int(d["room_count"])
    uniq = np.unique(sem)

    # Require the expected fixed spatial dimensions used by the models.
    if sem.shape != (256, 256):
        return False, "bad_sem_shape"
    if outline.shape != (256, 256):
        return False, "bad_outline_shape"

    # Despite the historical variable name, this threshold is applied
    # to the encoded connected-region count rather than verified room instances.
    if room_count < min_rooms:
        return False, "room_count_below_min"

    # Reject samples without any valid floor-plan support.
    if outline.sum() == 0:
        return False, "empty_outline"

    # Reject semantic masks containing only background.
    if np.count_nonzero(sem) == 0:
        return False, "empty_semantic_mask"

    # Require at least three distinct class IDs to avoid very limited
    # semantic masks in the retained modelling dataset.
    if len(uniq) < 3:
        return False, "too_few_classes"

    # Ensure every semantic value belongs to the defined nine-class range.
    if int(uniq.min()) < 0 or int(uniq.max()) > 8:
        return False, "class_out_of_range"

    return True, "kept"


# Filter all processed samples and copy valid NPZ files into the clean dataset.
def main():
    parser = argparse.ArgumentParser(description="Filter processed CubiCasa5K NPZ files into a clean dataset.")
    parser.add_argument("--src_dir", type=str, default="data/processed_npz", help="Folder containing processed NPZ files.")
    parser.add_argument("--dst_dir", type=str, default="data/processed_npz_clean", help="Output folder for clean NPZ files.")
    parser.add_argument("--min_rooms", type=int, default=3, help="Minimum detected room count to keep a sample.")
    parser.add_argument("--clear", action="store_true", help="Clear existing NPZ files in the destination folder before filtering.")
    args = parser.parse_args()

    # Ensure the clean-dataset destination exists before copying samples.
    os.makedirs(args.dst_dir, exist_ok=True)

    # Optionally remove previously generated NPZ files so the destination
    # contains only samples retained by the current filtering run.
    if args.clear:
        for old_file in glob.glob(os.path.join(args.dst_dir, "*.npz")):
            os.remove(old_file)

    # Sort source filenames so filtering is performed in a consistent order.
    files = sorted(glob.glob(os.path.join(args.src_dir, "*.npz")))

    kept = 0
    dropped = 0
    reasons = {}

    # Apply the same validation rules independently to every processed sample.
    for f in files:
        valid, reason = is_valid_sample(f, args.min_rooms)

        if valid:
            # Preserve the original NPZ contents and file metadata when
            # copying a retained sample into the clean dataset.
            shutil.copy2(f, os.path.join(args.dst_dir, os.path.basename(f)))
            kept += 1
        else:
            # Record the reason for exclusion so the filtering process
            # can be audited after completion.
            dropped += 1
            reasons[reason] = reasons.get(reason, 0) + 1

    # Report the final retained and excluded sample counts.
    print("Source folder:", args.src_dir)
    print("Clean dataset folder:", args.dst_dir)
    print("Total:", len(files))
    print("Kept:", kept)
    print("Dropped:", dropped)

    # Summarise the number of exclusions associated with each filtering rule.
    if reasons:
        print("Drop reasons:")
        for reason, count in sorted(reasons.items()):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
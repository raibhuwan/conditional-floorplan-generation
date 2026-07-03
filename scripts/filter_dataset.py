import argparse
import glob
import os
import shutil

import numpy as np


def is_valid_sample(path, min_rooms):
    d = np.load(path, allow_pickle=True)

    required_keys = {"sem", "outline", "room_count", "sample_id"}
    if not required_keys.issubset(set(d.files)):
        return False, "missing_required_keys"

    sem = d["sem"]
    outline = d["outline"]
    room_count = int(d["room_count"])
    uniq = np.unique(sem)

    if sem.shape != (256, 256):
        return False, "bad_sem_shape"
    if outline.shape != (256, 256):
        return False, "bad_outline_shape"
    if room_count < min_rooms:
        return False, "room_count_below_min"
    if outline.sum() == 0:
        return False, "empty_outline"
    if np.count_nonzero(sem) == 0:
        return False, "empty_semantic_mask"
    if len(uniq) < 3:
        return False, "too_few_classes"
    if int(uniq.min()) < 0 or int(uniq.max()) > 8:
        return False, "class_out_of_range"

    return True, "kept"


def main():
    parser = argparse.ArgumentParser(description="Filter processed CubiCasa5K NPZ files into a clean dataset.")
    parser.add_argument("--src_dir", type=str, default="data/processed_npz", help="Folder containing processed NPZ files.")
    parser.add_argument("--dst_dir", type=str, default="data/processed_npz_clean", help="Output folder for clean NPZ files.")
    parser.add_argument("--min_rooms", type=int, default=3, help="Minimum detected room count to keep a sample.")
    parser.add_argument("--clear", action="store_true", help="Clear existing NPZ files in the destination folder before filtering.")
    args = parser.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)

    if args.clear:
        for old_file in glob.glob(os.path.join(args.dst_dir, "*.npz")):
            os.remove(old_file)

    files = sorted(glob.glob(os.path.join(args.src_dir, "*.npz")))

    kept = 0
    dropped = 0
    reasons = {}

    for f in files:
        valid, reason = is_valid_sample(f, args.min_rooms)

        if valid:
            shutil.copy2(f, os.path.join(args.dst_dir, os.path.basename(f)))
            kept += 1
        else:
            dropped += 1
            reasons[reason] = reasons.get(reason, 0) + 1

    print("Source folder:", args.src_dir)
    print("Clean dataset folder:", args.dst_dir)
    print("Total:", len(files))
    print("Kept:", kept)
    print("Dropped:", dropped)

    if reasons:
        print("Drop reasons:")
        for reason, count in sorted(reasons.items()):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
import argparse
import glob
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compute room-count statistics for a processed NPZ dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed_npz_clean",
        help="Folder containing clean processed NPZ files.",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))

    print("Data folder:", args.data_dir)
    print("Files found:", len(files))
    print("First files:", files[:5])

    counts = []

    for f in files:
        d = np.load(f, allow_pickle=True)
        counts.append(int(d["room_count"]))

    if len(counts) == 0:
        print("No samples found. Check the data path.")
    else:
        print("Num samples:", len(counts))
        print("Min room_count:", min(counts))
        print("Max room_count:", max(counts))
        print("Mean room_count:", sum(counts) / len(counts))


if __name__ == "__main__":
    main()
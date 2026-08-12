import argparse
import glob
import os

import numpy as np


# Compute summary statistics for the encoded connected-region count
# stored in the processed NPZ dataset.
def main():
    parser = argparse.ArgumentParser(description="Compute room-count statistics for a processed NPZ dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed_npz_clean",
        help="Folder containing clean processed NPZ files.",
    )
    args = parser.parse_args()

    # Locate processed samples in a consistent filename order.
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))

    print("Data folder:", args.data_dir)
    print("Files found:", len(files))
    print("First files:", files[:5])

    counts = []

    # Read the stored room_count field from every sample. Despite the
    # historical field name, this stores the encoded connected-region count
    # produced during preprocessing.
    for f in files:
        d = np.load(f, allow_pickle=True)
        counts.append(int(d["room_count"]))

    # Report whether the selected directory contained any processed samples.
    if len(counts) == 0:
        print("No samples found. Check the data path.")
    else:
        # Summarise the count distribution so an appropriate normalisation
        # limit can be identified for the conditional count channel.
        print("Num samples:", len(counts))
        print("Min room_count:", min(counts))
        print("Max room_count:", max(counts))
        print("Mean room_count:", sum(counts) / len(counts))


if __name__ == "__main__":
    main()
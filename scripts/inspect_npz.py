import argparse
import glob
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Inspect generated CubiCasa5K NPZ files.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed_npz",
        help="Folder containing processed NPZ files.",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    print("Found", len(files), "npz files")

    if not files:
        print("No NPZ files found in", args.data_dir)
        return

    print("First file:", files[0])

    d = np.load(files[0], allow_pickle=True)

    print("\nKeys inside npz:", d.files)

    if "sample_id" in d.files:
        print("sample_id:", d["sample_id"])

    if "room_count" in d.files:
        print("room_count:", int(d["room_count"]))

    print("\nShapes / dtypes")
    if "sem" in d.files:
        print("sem:", d["sem"].shape, d["sem"].dtype)
    if "outline" in d.files:
        print("outline:", d["outline"].shape, d["outline"].dtype)

    if "sem" in d.files:
        unique = np.unique(d["sem"])
        print("\nUnique class ids in sem:", unique)


if __name__ == "__main__":
    main()
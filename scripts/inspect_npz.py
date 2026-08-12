import argparse
import glob
import os

import numpy as np


# Inspect the structure and key values stored in one processed NPZ floor sample.
def main():
    parser = argparse.ArgumentParser(
        description="Inspect generated CubiCasa5K NPZ files."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed_npz",
        help="Folder containing processed NPZ files.",
    )
    args = parser.parse_args()

    # Locate processed samples in a consistent filename order.
    files = sorted(
        glob.glob(
            os.path.join(
                args.data_dir,
                "*.npz",
            )
        )
    )

    print(
        "Found",
        len(files),
        "npz files",
    )

    # Stop cleanly when the selected directory contains no processed samples.
    if not files:
        print(
            "No NPZ files found in",
            args.data_dir,
        )
        return

    # Inspect the first sample in the sorted processed-data directory.
    print(
        "First file:",
        files[0],
    )

    d = np.load(
        files[0],
        allow_pickle=True,
    )

    # Display the stored fields so the processed sample structure can be verified.
    print(
        "\nKeys inside npz:",
        d.files,
    )

    if "sample_id" in d.files:
        print(
            "sample_id:",
            d["sample_id"],
        )

    # The historical room_count field stores the encoded
    # connected-region count produced during preprocessing.
    if "room_count" in d.files:
        print(
            "room_count:",
            int(d["room_count"]),
        )

    print("\nShapes / dtypes")

    # Report semantic-mask and support-mask dimensions and data types.
    if "sem" in d.files:
        print(
            "sem:",
            d["sem"].shape,
            d["sem"].dtype,
        )

    if "outline" in d.files:
        print(
            "outline:",
            d["outline"].shape,
            d["outline"].dtype,
        )

    # List the semantic class IDs represented in the inspected sample.
    if "sem" in d.files:
        unique = np.unique(
            d["sem"]
        )

        print(
            "\nUnique class ids in sem:",
            unique,
        )


if __name__ == "__main__":
    main()
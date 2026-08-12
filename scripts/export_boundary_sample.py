import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import FloorplanNPZDataset
from src.data.splits import load_split


# Parse the dataset, split and output settings for exporting one support condition.
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export a binary support condition from a processed "
            "floor-plan sample."
        )
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed_npz_clean_full",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        default="outputs/splits/split_seed42_full.json",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=2,
        help="Position inside the selected split.",
    )
    parser.add_argument(
        "--max_count",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=(
            "outputs/figures/"
            "figure_5_6_binary_support_condition.png"
        ),
    )

    return parser.parse_args()


# Export the binary support condition from one selected processed floor sample.
def main():
    args = parse_args()

    # Load the processed dataset using the same count normalisation
    # configuration used by the trained models.
    dataset = FloorplanNPZDataset(
        args.data_dir,
        max_count=args.max_count,
    )

    # Load the fixed partition and select the requested subset.
    split = load_split(args.split_path)
    split_indices = split[args.split]

    # Ensure the requested position exists within the selected split.
    if args.sample_index < 0 or args.sample_index >= len(split_indices):
        raise IndexError(
            f"sample_index={args.sample_index} is outside the "
            f"{args.split} split range 0 to "
            f"{len(split_indices) - 1}."
        )

    # Convert the position within the split back to its original dataset index.
    dataset_index = int(
        split_indices[args.sample_index]
    )

    # Load the two-channel condition for the selected floor sample.
    x, _ = dataset[dataset_index]

    # Recover the filled binary support mask from the first input channel.
    support = (
        x[0]
        .detach()
        .cpu()
        .numpy()
        > 0.5
    ).astype(np.uint8)

    # Recover the spatially repeated normalised count from the second channel.
    normalised_count = float(
        x[1]
        .detach()
        .cpu()
        .numpy()
        .max()
    )

    # Reverse the normalisation to recover the encoded connected-region count.
    encoded_count = int(
        round(normalised_count * args.max_count)
    )

    output_directory = os.path.dirname(args.out_path)

    # Create the output folder when one is included in the requested path.
    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    # Convert the binary support mask to an 8-bit image for PNG export.
    support_image = support * 255

    saved = cv2.imwrite(
        args.out_path,
        support_image,
    )

    if not saved:
        raise IOError(
            f"Could not save PNG image: {args.out_path}"
        )

    # Save the same binary support condition as a PDF for figure use.
    pdf_path = os.path.splitext(
        args.out_path
    )[0] + ".pdf"

    figure = plt.figure(
        figsize=(5, 5)
    )

    plt.imshow(
        support,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    plt.axis("off")
    plt.tight_layout(pad=0)

    figure.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close(figure)

    # Report the exact source sample and exported conditioning information.
    print("Data folder:", args.data_dir)
    print("Split file:", args.split_path)
    print("Split:", args.split)
    print("Split sample position:", args.sample_index)
    print("Dataset index:", dataset_index)
    print(
        "Encoded connected-region count:",
        encoded_count,
    )
    print("Saved PNG:", args.out_path)
    print("Saved PDF:", pdf_path)


if __name__ == "__main__":
    main()
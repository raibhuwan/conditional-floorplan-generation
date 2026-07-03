import argparse
import os

import cv2
import numpy as np

from src.data.dataset import FloorplanNPZDataset
from src.data.splits import load_split


def parse_args():
    parser = argparse.ArgumentParser(description="Export a boundary/outline image from a processed NPZ sample.")
    parser.add_argument("--data_dir", type=str, default="data/processed_npz_clean_full")
    parser.add_argument("--split_path", type=str, default="outputs/splits/split_seed42_full.json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample_index", type=int, default=0, help="Index inside the selected split.")
    parser.add_argument("--max_count", type=int, default=32)
    parser.add_argument("--out_path", type=str, default="inputs/boundary.png")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = FloorplanNPZDataset(args.data_dir, max_count=args.max_count)
    split = load_split(args.split_path)

    if args.sample_index < 0 or args.sample_index >= len(split[args.split]):
        raise IndexError(
            f"sample_index={args.sample_index} is outside the {args.split} split range "
            f"0 to {len(split[args.split]) - 1}."
        )

    dataset_index = split[args.split][args.sample_index]
    x, y = dataset[dataset_index]

    outline = x[0].numpy()
    outline_img = (outline > 0.5).astype(np.uint8) * 255

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    cv2.imwrite(args.out_path, outline_img)

    print("Data folder:", args.data_dir)
    print("Split file:", args.split_path)
    print("Split:", args.split)
    print("Split sample index:", args.sample_index)
    print("Dataset index:", dataset_index)
    print("Saved boundary image:", args.out_path)


if __name__ == "__main__":
    main()

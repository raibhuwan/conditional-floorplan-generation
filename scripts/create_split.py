import argparse
import glob
import os

from src.data.splits import make_split_indices, save_split, split_lengths


def main():
    parser = argparse.ArgumentParser(
        description="Create a fixed train/validation/test split for processed NPZ floor plans."
    )
    parser.add_argument("--data_dir", type=str, default="data/processed_npz_clean")
    parser.add_argument("--out", type=str, default="outputs/splits/split_seed42.json")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    n = len(files)

    if n == 0:
        raise RuntimeError(f"No .npz files found in {args.data_dir}")

    split = make_split_indices(
        n=n,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    metadata = {
        "data_dir": args.data_dir,
        "num_samples": n,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
    }

    save_split(split, args.out, metadata=metadata)
    train_n, val_n, test_n = split_lengths(split)

    print("Created fixed split:")
    print(f"  data_dir: {args.data_dir}")
    print(f"  total: {n}")
    print(f"  train: {train_n}")
    print(f"  val: {val_n}")
    print(f"  test: {test_n}")
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
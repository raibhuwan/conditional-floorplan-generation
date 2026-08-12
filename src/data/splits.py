import json
import os
import random
from typing import Dict, List, Tuple


def make_split_indices(
    n: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Create fixed train/validation/test index splits.

    The ratios must add to 1.0. The returned indices are shuffled using
    the supplied seed so that repeated runs use the same split.
    """

    # Require at least one sample before attempting to construct a split.
    if n <= 0:
        raise ValueError("Dataset size must be greater than zero.")

    # Check that the requested partition ratios describe the complete dataset.
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Shuffle sample indices using a dedicated seeded random generator so the
    # same seed produces the same partition ordering on repeated runs.
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    # Calculate the boundaries used to separate the shuffled indices.
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    # Assign non-overlapping index ranges to training, validation and test sets.
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


# Save a generated split and optional metadata to a JSON file.
def save_split(split: Dict[str, List[int]], path: str, metadata: Dict = None) -> None:

    # Create the destination directory before writing the split file.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Store metadata separately from the actual partition indices.
    payload = {
        "metadata": metadata or {},
        "split": split,
    }

    # Write the split in a human-readable JSON format for later reuse.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# Load previously saved train, validation and test indices from JSON.
def load_split(path: str) -> Dict[str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Return only the partition indices required by the training and evaluation code.
    return payload["split"]


# Return the number of samples contained in each saved partition.
def split_lengths(split: Dict[str, List[int]]) -> Tuple[int, int, int]:
    return len(split["train"]), len(split["val"]), len(split["test"])
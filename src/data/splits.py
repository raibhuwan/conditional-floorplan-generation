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
    if n <= 0:
        raise ValueError("Dataset size must be greater than zero.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


def save_split(split: Dict[str, List[int]], path: str, metadata: Dict = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "split": split,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_split(path: str) -> Dict[str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["split"]


def split_lengths(split: Dict[str, List[int]]) -> Tuple[int, int, int]:
    return len(split["train"]), len(split["val"]), len(split["test"])
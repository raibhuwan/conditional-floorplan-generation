import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PREPROCESSING_CSV = ROOT / "evidence/preprocessing/all_processed_samples.csv"
SPLIT_JSON = ROOT / "evidence/splits/split_seed42_full.json"
METRICS_DIR = ROOT / "evidence/metrics"
OUTPUT_CSV = ROOT / "evidence/metrics/bootstrap_uncertainty_summary.csv"

N_BOOTSTRAP = 20000
BOOTSTRAP_SEED = 20260810


METRIC_FILES = {
    "unet": "metrics_unet_room_count_fixed.csv",
    "cgan": "metrics_cgan_room_count_fixed.csv",
    "unet_morphology": "metrics_unet_morphology_room_count_fixed.csv",
    "cgan_morphology": "metrics_cgan_morphology_room_count_fixed.csv",
    "unet_hillclimb": "metrics_unet_hillclimb_room_count_fixed.csv",
    "cgan_hillclimb": "metrics_cgan_hillclimb_room_count_fixed.csv",
}


def load_test_buildings():
    processed = pd.read_csv(PREPROCESSING_CSV)

    retained = processed[
        processed["present_in_clean_folder"]
        .astype(str)
        .str.lower()
        .eq("true")
    ].copy()

    # FloorplanNPZDataset uses sorted filenames, so this reconstructs
    # the dataset-index ordering used by the saved split.
    retained = retained.sort_values("filename").reset_index(drop=True)

    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    test_indices = split_data["split"]["test"]

    test_meta = retained.iloc[test_indices][
        ["filename", "sample_id"]
    ].copy().reset_index(drop=True)

    test_meta["dataset_index"] = test_indices

    # sample_id format:
    # high_quality_architectural/<building_id>/Floor-*
    test_meta["building_id"] = (
        test_meta["sample_id"].astype(str).str.split("/").str[1]
    )

    return test_meta, test_indices


def load_metrics(test_indices):
    metrics = {}

    for name, filename in METRIC_FILES.items():
        df = pd.read_csv(METRICS_DIR / filename)

        if len(df) != len(test_indices):
            raise ValueError(
                f"{filename}: expected {len(test_indices)} rows, "
                f"found {len(df)}."
            )

        if df["dataset_index"].tolist() != test_indices:
            raise ValueError(
                f"{filename}: dataset-index order does not match "
                "the held-out test split."
            )

        metrics[name] = df

    return metrics


def cluster_bootstrap_difference(
    values_a,
    values_b,
    building_ids,
    n_bootstrap=N_BOOTSTRAP,
    seed=BOOTSTRAP_SEED,
):
    difference = (
        np.asarray(values_a, dtype=float)
        - np.asarray(values_b, dtype=float)
    )

    unique_buildings = np.unique(building_ids)
    n_buildings = len(unique_buildings)

    cluster_sums = np.zeros(n_buildings, dtype=float)
    cluster_sizes = np.zeros(n_buildings, dtype=int)

    for i, building in enumerate(unique_buildings):
        mask = building_ids == building
        cluster_sums[i] = difference[mask].sum()
        cluster_sizes[i] = mask.sum()

    rng = np.random.default_rng(seed)

    # Resample buildings with replacement. Multinomial counts are
    # equivalent to drawing n_buildings clusters with replacement.
    bootstrap_counts = rng.multinomial(
        n_buildings,
        np.full(n_buildings, 1.0 / n_buildings),
        size=n_bootstrap,
    )

    bootstrap_sums = bootstrap_counts @ cluster_sums
    bootstrap_sizes = bootstrap_counts @ cluster_sizes

    bootstrap_means = bootstrap_sums / bootstrap_sizes

    lower, upper = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return {
        "mean_difference": float(difference.mean()),
        "ci_lower_95": float(lower),
        "ci_upper_95": float(upper),
    }


def main():
    test_meta, test_indices = load_test_buildings()
    metrics = load_metrics(test_indices)

    building_ids = test_meta["building_id"].to_numpy()

    comparisons = [
        (
            "U-Net baseline - cGAN baseline mIoU",
            metrics["unet"]["miou_no_bg"],
            metrics["cgan"]["miou_no_bg"],
        ),
        (
            "U-Net baseline - cGAN baseline count MAE",
            metrics["unet"]["room_count_error"],
            metrics["cgan"]["room_count_error"],
        ),
        (
            "U-Net morphology - U-Net baseline mIoU",
            metrics["unet_morphology"]["miou_no_bg"],
            metrics["unet"]["miou_no_bg"],
        ),
        (
            "cGAN morphology - cGAN baseline mIoU",
            metrics["cgan_morphology"]["miou_no_bg"],
            metrics["cgan"]["miou_no_bg"],
        ),
        (
            "U-Net hill-climbing - U-Net baseline mIoU",
            metrics["unet_hillclimb"]["miou_no_bg"],
            metrics["unet"]["miou_no_bg"],
        ),
        (
            "cGAN hill-climbing - cGAN baseline mIoU",
            metrics["cgan_hillclimb"]["miou_no_bg"],
            metrics["cgan"]["miou_no_bg"],
        ),
    ]

    rows = []

    for comparison, values_a, values_b in comparisons:
        result = cluster_bootstrap_difference(
            values_a,
            values_b,
            building_ids,
        )

        rows.append(
            {
                "comparison": comparison,
                "mean_difference": result["mean_difference"],
                "ci_lower_95": result["ci_lower_95"],
                "ci_upper_95": result["ci_upper_95"],
                "bootstrap_resamples": N_BOOTSTRAP,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "test_floors": len(test_meta),
                "test_buildings": test_meta["building_id"].nunique(),
            }
        )

    output = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False)

    print(f"Test floors: {len(test_meta)}")
    print(
        f"Unique test buildings: "
        f"{test_meta['building_id'].nunique()}"
    )
    print(f"Bootstrap resamples: {N_BOOTSTRAP}")
    print(f"Bootstrap seed: {BOOTSTRAP_SEED}")
    print()

    for _, row in output.iterrows():
        print(row["comparison"])
        print(
            f"  mean difference: "
            f"{row['mean_difference']:.6f}"
        )
        print(
            f"  95% bootstrap interval: "
            f"{row['ci_lower_95']:.6f} to "
            f"{row['ci_upper_95']:.6f}"
        )

    print()
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

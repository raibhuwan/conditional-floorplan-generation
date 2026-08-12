import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

META_CSV = ROOT / "evidence/preprocessing/all_processed_samples.csv"
SPLIT_JSON = ROOT / "evidence/splits/split_seed42_full.json"
METRICS_DIR = ROOT / "evidence/metrics"

SUMMARY_OUT = METRICS_DIR / "unseen_building_metrics_summary.csv"
BOOTSTRAP_OUT = METRICS_DIR / "unseen_building_bootstrap_summary.csv"
SAMPLES_OUT = METRICS_DIR / "unseen_building_test_samples.csv"

N_BOOTSTRAP = 20000
BOOTSTRAP_SEED = 20260810


# Associate each evaluated method with its previously generated
# sample-level metric file.
METRIC_FILES = {
    "U-Net baseline": "metrics_unet_room_count_fixed.csv",
    "U-Net + morphology": "metrics_unet_morphology_room_count_fixed.csv",
    "U-Net + hill-climbing": "metrics_unet_hillclimb_room_count_fixed.csv",
    "cGAN baseline": "metrics_cgan_room_count_fixed.csv",
    "cGAN + morphology": "metrics_cgan_morphology_room_count_fixed.csv",
    "cGAN + hill-climbing": "metrics_cgan_hillclimb_room_count_fixed.csv",
}


# Allow equivalent metric column names produced by different evaluation scripts.
METRIC_COLUMNS = {
    "miou": ["miou_no_bg", "miou"],
    "adjacency_f1": ["adjacency_f1", "adj_f1"],
    "compactness": ["compact_pred", "compactness"],
    "bvr": ["boundary_violation_rate", "bvr"],
    "count_mae": [
        "room_count_error",
        "connected_region_count_error",
        "count_error",
    ],
}


# Find the first available CSV column corresponding to the requested metric.
def find_column(df, candidates, metric_name):
    for column in candidates:
        if column in df.columns:
            return column

    # Stop when none of the accepted column names exists rather than
    # silently evaluating the wrong quantity.
    raise ValueError(
        f"Could not find column for {metric_name}. "
        f"Available columns: {list(df.columns)}"
    )


# Recover the original building identifier from a processed floor sample ID.
def building_id_from_sample_id(value):
    # Normalise path separators so IDs are handled consistently across platforms.
    parts = str(value).replace("\\", "/").split("/")

    if len(parts) < 2:
        raise ValueError(
            f"Cannot extract building ID from sample_id: {value}"
        )

    # The building identifier is the path component immediately before the floor ID.
    return parts[-2]


# Reconstruct dataset ordering and identify test floors whose buildings
# are absent from both the original training and validation partitions.
def load_metadata():
    processed = pd.read_csv(META_CSV)

    # Retain only samples that survived preprocessing/filtering and were
    # therefore available to the model dataset.
    retained = processed[
        processed["present_in_clean_folder"]
        .astype(str)
        .str.lower()
        .eq("true")
    ].copy()

    # Reconstruct the ordering used by FloorplanNPZDataset.
    retained = retained.sort_values("filename").reset_index(drop=True)
    retained["dataset_index"] = np.arange(len(retained))

    # Associate every floor-level sample with its source building.
    retained["building_id"] = retained["sample_id"].apply(
        building_id_from_sample_id
    )

    # Load the original fixed floor-sample-level train/validation/test split.
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split_data = json.load(f)["split"]

    train_idx = split_data["train"]

    # Support either validation key used by saved split files.
    if "val" in split_data:
        val_idx = split_data["val"]
    else:
        val_idx = split_data["validation"]

    test_idx = split_data["test"]

    # Identify every building represented during model development.
    train_buildings = set(
        retained.iloc[train_idx]["building_id"]
    )

    val_buildings = set(
        retained.iloc[val_idx]["building_id"]
    )

    development_buildings = train_buildings | val_buildings

    # Preserve metadata for all floors in the existing test partition.
    test_meta = retained.iloc[test_idx][
        ["dataset_index", "filename", "sample_id", "building_id"]
    ].copy()

    # Form a supplementary subset by removing test floors whose source
    # building also appears in training or validation. This does not create
    # or replace the original floor-level train/validation/test split.
    unseen_test = test_meta[
        ~test_meta["building_id"].isin(development_buildings)
    ].copy()

    return test_meta, unseen_test


# Load one method's sample-level metrics and align them with the strictly
# unseen-building subset using the original dataset index.
def load_metric_file(filename, unseen_meta):
    path = METRICS_DIR / filename
    df = pd.read_csv(path)

    # Dataset indices are required to align identical held-out floors
    # across model and refinement configurations.
    if "dataset_index" not in df.columns:
        raise ValueError(
            f"{filename} does not contain dataset_index."
        )

    # Require one metric record per dataset sample.
    if df["dataset_index"].duplicated().any():
        raise ValueError(
            f"{filename} contains duplicate dataset_index values."
        )

    # Keep only the strictly unseen-building floors while attaching
    # their building identifiers for cluster-aware analysis.
    joined = unseen_meta[
        ["dataset_index", "building_id"]
    ].merge(
        df,
        on="dataset_index",
        how="left",
        validate="one_to_one",
    )

    # Confirm that filtering and metric alignment did not lose any unseen floors.
    if len(joined) != len(unseen_meta):
        raise ValueError(
            f"{filename}: unseen subset row-count mismatch."
        )

    return df, joined


# Estimate a paired metric difference and its bootstrap interval while
# resampling source buildings rather than treating every floor as independent.
def cluster_bootstrap_difference(
    pair_df,
    value_a,
    value_b,
):
    # Calculate the paired per-floor difference between two methods
    # evaluated on exactly the same held-out samples.
    difference = (
        pair_df[value_a].to_numpy(dtype=float)
        - pair_df[value_b].to_numpy(dtype=float)
    )

    building_ids = pair_df["building_id"].astype(str).to_numpy()
    unique_buildings = np.unique(building_ids)

    # Aggregate the paired differences and floor counts within each building.
    # This keeps multiple floors from the same source building together.
    cluster_sums = np.zeros(len(unique_buildings), dtype=float)
    cluster_sizes = np.zeros(len(unique_buildings), dtype=int)

    for i, building in enumerate(unique_buildings):
        mask = building_ids == building
        cluster_sums[i] = difference[mask].sum()
        cluster_sizes[i] = mask.sum()

    # Use a fixed bootstrap seed so the reported interval is reproducible.
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # Multinomial counts represent repeated sampling of building clusters
    # with replacement for each bootstrap replicate.
    counts = rng.multinomial(
        len(unique_buildings),
        np.full(
            len(unique_buildings),
            1.0 / len(unique_buildings),
        ),
        size=N_BOOTSTRAP,
    )

    # Preserve all floors belonging to a sampled building when calculating
    # each bootstrap replicate of the floor-level mean difference.
    bootstrap_means = (
        counts @ cluster_sums
    ) / (
        counts @ cluster_sizes
    )

    # Use percentile bounds to form the 95% bootstrap interval.
    lower, upper = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return (
        float(difference.mean()),
        float(lower),
        float(upper),
    )


# Run the strictly unseen-building evaluation and supplementary
# building-cluster bootstrap comparisons.
def main():
    test_meta, unseen_meta = load_metadata()

    # Report the relationship between the full existing test partition
    # and its strictly unseen-building subset.
    print(f"Total test floors: {len(test_meta)}")
    print(
        "Total unique test buildings: "
        f"{test_meta['building_id'].nunique()}"
    )
    print(
        "Strictly unseen-building test floors: "
        f"{len(unseen_meta)}"
    )
    print(
        "Strictly unseen test buildings: "
        f"{unseen_meta['building_id'].nunique()}"
    )
    print(
        "Strictly unseen proportion: "
        f"{100 * len(unseen_meta) / len(test_meta):.2f}%"
    )
    print()

    # Save the exact floor samples included in the supplementary analysis.
    unseen_meta.to_csv(SAMPLES_OUT, index=False)

    loaded = {}
    summary_rows = []

    # Calculate the same evaluation metrics for each method using only
    # floors whose buildings are absent from training and validation.
    for method, filename in METRIC_FILES.items():
        full_df, unseen_df = load_metric_file(
            filename,
            unseen_meta,
        )

        # Resolve the appropriate source column for each required metric.
        columns = {
            name: find_column(
                unseen_df,
                candidates,
                name,
            )
            for name, candidates in METRIC_COLUMNS.items()
        }

        # Retain aligned data and column names for the paired comparisons below.
        loaded[method] = {
            "full": full_df,
            "unseen": unseen_df,
            "columns": columns,
        }

        # Aggregate sample-level values across the strictly unseen-building subset.
        summary_rows.append(
            {
                "method": method,
                "test_floors": len(unseen_df),
                "test_buildings": (
                    unseen_df["building_id"].nunique()
                ),
                "miou": unseen_df[
                    columns["miou"]
                ].mean(),
                "adjacency_f1": unseen_df[
                    columns["adjacency_f1"]
                ].mean(),
                "compactness": unseen_df[
                    columns["compactness"]
                ].mean(),
                "bvr": unseen_df[
                    columns["bvr"]
                ].mean(),
                "count_mae": unseen_df[
                    columns["count_mae"]
                ].mean(),
            }
        )

    # Save the strictly unseen-building metric summary for all six configurations.
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_OUT, index=False)

    print("STRICTLY UNSEEN-BUILDING METRICS")
    print(
        summary[
            [
                "method",
                "miou",
                "adjacency_f1",
                "compactness",
                "bvr",
                "count_mae",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )
    print()

    # Align two methods on identical unseen test floors before computing
    # their paired metric difference.
    def aligned_pair(method_a, method_b, metric):
        a = loaded[method_a]
        b = loaded[method_b]

        col_a = a["columns"][metric]
        col_b = b["columns"][metric]

        left = a["unseen"][
            ["dataset_index", "building_id", col_a]
        ].copy()

        right = b["unseen"][
            ["dataset_index", col_b]
        ].copy()

        # Standardise metric names so the same bootstrap function can
        # process every model/refinement comparison.
        left = left.rename(
            columns={col_a: "value_a"}
        )

        right = right.rename(
            columns={col_b: "value_b"}
        )

        # The one-to-one merge ensures both values correspond to the same floor.
        pair = left.merge(
            right,
            on="dataset_index",
            how="inner",
            validate="one_to_one",
        )

        return pair

    # Define the model and refinement comparisons used in the bootstrap analysis.
    comparisons = [
        (
            "U-Net baseline - cGAN baseline mIoU",
            "U-Net baseline",
            "cGAN baseline",
            "miou",
        ),
        (
            "U-Net baseline - cGAN baseline count MAE",
            "U-Net baseline",
            "cGAN baseline",
            "count_mae",
        ),
        (
            "U-Net morphology - U-Net baseline mIoU",
            "U-Net + morphology",
            "U-Net baseline",
            "miou",
        ),
        (
            "cGAN morphology - cGAN baseline mIoU",
            "cGAN + morphology",
            "cGAN baseline",
            "miou",
        ),
        (
            "U-Net hill-climbing - U-Net baseline mIoU",
            "U-Net + hill-climbing",
            "U-Net baseline",
            "miou",
        ),
        (
            "cGAN hill-climbing - cGAN baseline mIoU",
            "cGAN + hill-climbing",
            "cGAN baseline",
            "miou",
        ),
    ]

    bootstrap_rows = []

    print("STRICTLY UNSEEN-BUILDING BOOTSTRAP")
    print(
        f"Resamples: {N_BOOTSTRAP}, "
        f"seed: {BOOTSTRAP_SEED}"
    )
    print()

    # Perform paired building-cluster bootstrap analysis for each comparison.
    for (
        label,
        method_a,
        method_b,
        metric,
    ) in comparisons:
        pair = aligned_pair(
            method_a,
            method_b,
            metric,
        )

        mean_diff, lower, upper = (
            cluster_bootstrap_difference(
                pair,
                "value_a",
                "value_b",
            )
        )

        # Store the observed paired difference and its bootstrap interval.
        bootstrap_rows.append(
            {
                "comparison": label,
                "mean_difference": mean_diff,
                "ci_lower_95": lower,
                "ci_upper_95": upper,
                "bootstrap_resamples": N_BOOTSTRAP,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "test_floors": len(pair),
                "test_buildings": (
                    pair["building_id"].nunique()
                ),
            }
        )

        print(label)
        print(
            f"  mean difference: {mean_diff:.6f}"
        )
        print(
            "  95% bootstrap interval: "
            f"{lower:.6f} to {upper:.6f}"
        )
        print()

    # Save all supplementary bootstrap comparisons to a separate CSV file.
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(
        BOOTSTRAP_OUT,
        index=False,
    )

    print(f"Saved: {SUMMARY_OUT}")
    print(f"Saved: {BOOTSTRAP_OUT}")
    print(f"Saved: {SAMPLES_OUT}")


if __name__ == "__main__":
    main()
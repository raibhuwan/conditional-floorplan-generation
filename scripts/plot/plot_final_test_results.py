from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------
# File locations
# ---------------------------------------------------------

RESULT_FILES = {
    "U-Net baseline": Path(
        "outputs/metrics_unet_room_count_fixed.csv"
    ),
    "U-Net + morphology": Path(
        "outputs/metrics_unet_morphology_room_count_fixed.csv"
    ),
    "U-Net + hill-climbing": Path(
        "outputs/metrics_unet_hillclimb_room_count_fixed.csv"
    ),
    "cGAN baseline": Path(
        "outputs/metrics_cgan_room_count_fixed.csv"
    ),
    "cGAN + morphology": Path(
        "outputs/metrics_cgan_morphology_room_count_fixed.csv"
    ),
    "cGAN + hill-climbing": Path(
        "outputs/metrics_cgan_hillclimb_room_count_fixed.csv"
    ),
}

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load and validate results
# ---------------------------------------------------------

summary_rows = []

for configuration, csv_path in RESULT_FILES.items():
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Result file not found: {csv_path}"
        )

    results = pd.read_csv(csv_path)

    required_columns = {
        "miou_no_bg",
        "room_count_error",
    }

    missing_columns = required_columns.difference(
        results.columns
    )

    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(results) != 565:
        raise ValueError(
            f"{csv_path} contains {len(results)} rows; "
            "565 held-out test samples were expected."
        )

    summary_rows.append(
        {
            "configuration": configuration,
            "mean_miou": results["miou_no_bg"].mean(),
            "room_count_mae": results[
                "room_count_error"
            ].mean(),
        }
    )

summary = pd.DataFrame(summary_rows)

print("\nCalculated figure values:")
print(summary.to_string(index=False))


# ---------------------------------------------------------
# Figure 5.2: Test-set mIoU
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5.5))

bars = ax.barh(
    summary["configuration"],
    summary["mean_miou"],
)

ax.invert_yaxis()
ax.set_xlabel("Mean intersection over union (mIoU)")
ax.set_ylabel("Model configuration")
ax.set_xlim(0, 0.45)
ax.grid(
    axis="x",
    linestyle="--",
    linewidth=0.7,
    alpha=0.5,
)
ax.set_axisbelow(True)

for bar, value in zip(
    bars,
    summary["mean_miou"],
):
    ax.text(
        value + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
        fontsize=10,
    )

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "figure_5_2_test_miou.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_DIR / "figure_5_2_test_miou.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------
# Figure 5.3: Room-count MAE
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5.5))

bars = ax.barh(
    summary["configuration"],
    summary["room_count_mae"],
)

ax.invert_yaxis()
ax.set_xlabel("Connected-region count Mean Absolute Error")
ax.set_ylabel("Model configuration")
ax.set_xlim(0, 4.2)
ax.grid(
    axis="x",
    linestyle="--",
    linewidth=0.7,
    alpha=0.5,
)
ax.set_axisbelow(True)

for bar, value in zip(
    bars,
    summary["room_count_mae"],
):
    ax.text(
        value + 0.05,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
        fontsize=10,
    )

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "figure_5_3_connected_region_count_mae.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_DIR / "figure_5_3_connected_region_count_mae.pdf",
    bbox_inches="tight",
)

plt.close(fig)


print("\nFigures saved successfully:")
print(
    OUTPUT_DIR / "figure_5_2_test_miou.png"
)
print(
    OUTPUT_DIR / "figure_5_3_room_count_mae.png"
)
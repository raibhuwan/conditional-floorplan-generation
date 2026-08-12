from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


UNET_LOG = Path("outputs/logs/unet_training_history.csv")
CGAN_LOG = Path("outputs/logs/cgan_training_history.csv")

FIGURE_OUTPUT = Path(
    "outputs/figures/figure_5_1_training_validation_miou.png"
)

PDF_OUTPUT = Path(
    "outputs/figures/figure_5_1_training_validation_miou.pdf"
)

SUMMARY_OUTPUT = Path(
    "outputs/logs/training_checkpoint_summary.csv"
)


# Validate that a training log contains the expected columns and complete epoch sequence.
def validate_log(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    model_name: str,
) -> None:
    # Check that all values required for plotting and checkpoint analysis are present.
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{model_name} log is missing columns: "
            f"{sorted(missing_columns)}"
        )

    # The retained experiments were trained for 30 epochs.
    if len(dataframe) != 30:
        raise ValueError(
            f"{model_name} log contains {len(dataframe)} rows; "
            "30 epoch rows were expected."
        )

    expected_epochs = list(range(1, 31))

    # Require the complete ordered epoch sequence to avoid plotting
    # incomplete or incorrectly ordered training histories.
    if dataframe["epoch"].tolist() != expected_epochs:
        raise ValueError(
            f"{model_name} log does not contain epochs 1 to 30 "
            "in the expected order."
        )


# Generate the training/validation mIoU figure and checkpoint summary.
def main() -> None:
    # Require both retained training logs before attempting comparison.
    if not UNET_LOG.exists():
        raise FileNotFoundError(
            f"U-Net log was not found: {UNET_LOG}"
        )

    if not CGAN_LOG.exists():
        raise FileNotFoundError(
            f"cGAN log was not found: {CGAN_LOG}"
        )

    # Load the recorded epoch-level histories for both model configurations.
    unet = pd.read_csv(UNET_LOG)
    cgan = pd.read_csv(CGAN_LOG)

    # Validate the U-Net log before using it for plotting or checkpoint analysis.
    validate_log(
        unet,
        {
            "epoch",
            "train_miou",
            "validation_miou",
            "checkpoint_saved",
        },
        "U-Net",
    )

    # Apply the same validation checks to the cGAN training history.
    validate_log(
        cgan,
        {
            "epoch",
            "train_miou",
            "validation_miou",
            "checkpoint_saved",
        },
        "cGAN",
    )

    # Locate the epoch with the highest logged validation mIoU for each model.
    unet_best_index = unet["validation_miou"].idxmax()
    cgan_best_index = cgan["validation_miou"].idxmax()

    unet_best = unet.loc[unet_best_index]
    cgan_best = cgan.loc[cgan_best_index]

    # Summarise the selected validation result together with the corresponding
    # training mIoU and final-epoch values for each model.
    summary = pd.DataFrame(
        [
            {
                "model": "U-Net",
                "best_epoch": int(unet_best["epoch"]),
                "best_validation_miou": float(
                    unet_best["validation_miou"]
                ),
                "training_miou_at_best_epoch": float(
                    unet_best["train_miou"]
                ),
                "final_training_miou": float(
                    unet.iloc[-1]["train_miou"]
                ),
                "final_validation_miou": float(
                    unet.iloc[-1]["validation_miou"]
                ),
            },
            {
                "model": "Pix2Pix-style cGAN",
                "best_epoch": int(cgan_best["epoch"]),
                "best_validation_miou": float(
                    cgan_best["validation_miou"]
                ),
                "training_miou_at_best_epoch": float(
                    cgan_best["train_miou"]
                ),
                "final_training_miou": float(
                    cgan.iloc[-1]["train_miou"]
                ),
                "final_validation_miou": float(
                    cgan.iloc[-1]["validation_miou"]
                ),
            },
        ]
    )

    # Save the checkpoint summary as a separate CSV for traceability.
    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    # Create the figure output directory before plotting.
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Plot U-Net training and validation mIoU across all recorded epochs.
    plt.plot(
        unet["epoch"],
        unet["train_miou"],
        label="U-Net training mIoU",
        linewidth=2,
    )

    plt.plot(
        unet["epoch"],
        unet["validation_miou"],
        label="U-Net validation mIoU",
        linewidth=2,
    )

    # Plot the corresponding cGAN generator training and validation mIoU.
    plt.plot(
        cgan["epoch"],
        cgan["train_miou"],
        label="cGAN training mIoU",
        linewidth=2,
    )

    plt.plot(
        cgan["epoch"],
        cgan["validation_miou"],
        label="cGAN validation mIoU",
        linewidth=2,
    )

    # Mark the retained checkpoint epoch used by both final model configurations.
    plt.axvline(
        x=20,
        linestyle="--",
        linewidth=1.5,
        label="Selected checkpoint epoch",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Mean Intersection over Union")
    plt.title(
        "Training and Validation mIoU Across 30 Epochs"
    )

    plt.xticks(range(0, 31, 5))
    plt.ylim(0.0, 0.8)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save a high-resolution PNG version of the training-history figure.
    plt.savefig(
        FIGURE_OUTPUT,
        dpi=300,
        bbox_inches="tight",
    )

    # Save the same figure as a PDF.
    plt.savefig(
        PDF_OUTPUT,
        format="pdf",
        bbox_inches="tight",
    )

    plt.close()

    # Report all generated evidence files and the checkpoint summary.
    print(f"Saved figure to: {FIGURE_OUTPUT}")
    print(f"Saved PDF figure to: {PDF_OUTPUT}")
    print(f"Saved summary to: {SUMMARY_OUTPUT}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
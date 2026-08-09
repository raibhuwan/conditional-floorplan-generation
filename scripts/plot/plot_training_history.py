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


def validate_log(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    model_name: str,
) -> None:
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{model_name} log is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(dataframe) != 30:
        raise ValueError(
            f"{model_name} log contains {len(dataframe)} rows; "
            "30 epoch rows were expected."
        )

    expected_epochs = list(range(1, 31))

    if dataframe["epoch"].tolist() != expected_epochs:
        raise ValueError(
            f"{model_name} log does not contain epochs 1 to 30 "
            "in the expected order."
        )


def main() -> None:
    if not UNET_LOG.exists():
        raise FileNotFoundError(
            f"U-Net log was not found: {UNET_LOG}"
        )

    if not CGAN_LOG.exists():
        raise FileNotFoundError(
            f"cGAN log was not found: {CGAN_LOG}"
        )

    unet = pd.read_csv(UNET_LOG)
    cgan = pd.read_csv(CGAN_LOG)

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

    unet_best_index = unet["validation_miou"].idxmax()
    cgan_best_index = cgan["validation_miou"].idxmax()

    unet_best = unet.loc[unet_best_index]
    cgan_best = cgan.loc[cgan_best_index]

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

    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

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

    plt.savefig(
        FIGURE_OUTPUT,
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        PDF_OUTPUT,
        format="pdf",
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved figure to: {FIGURE_OUTPUT}")
    print(f"Saved PDF figure to: {PDF_OUTPUT}")
    print(f"Saved summary to: {SUMMARY_OUTPUT}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
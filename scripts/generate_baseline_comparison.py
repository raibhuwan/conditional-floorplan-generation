import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap

from src.data.dataset import FloorplanNPZDataset
from src.data.splits import load_split
from src.models.unet import UNet


DATA_DIR = "data/processed_npz_clean_full"
SPLIT_PATH = "outputs/splits/split_seed42_full.json"

UNET_CKPT = "outputs/checkpoints/unet_base16_best.pt"
CGAN_CKPT = "outputs/checkpoints/cgan_unet_patchgan_best.pt"

UNET_METRICS = "outputs/metrics_unet_room_count_fixed.csv"
CGAN_METRICS = "outputs/metrics_cgan_room_count_fixed.csv"

OUTPUT_PATH = (
    "outputs/figures/"
    "figure_5_5_unet_cgan_comparison.png"
)

MAX_COUNT = 32
NUM_CLASSES = 9

# The same semantic palette used in Figure 5.4.
CLASS_COLOURS = [
    "#2d1238",  # 0: background
    "#25d9ad",  # 1
    "#38a8e8",  # 2
    "#4964da",  # 3
    "#f45a0a",  # 4
    "#75fa39",  # 5
    "#d7ee00",  # 6
    "#f28c16",  # 7
    "#d3360a",  # 8: wall / structure
]

SEMANTIC_CMAP = ListedColormap(CLASS_COLOURS)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the retained U-Net and cGAN checkpoints "
            "on one selected held-out test sample."
        )
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR,
    )
    parser.add_argument(
        "--split_path",
        type=str,
        default=SPLIT_PATH,
    )
    parser.add_argument(
        "--unet_ckpt",
        type=str,
        default=UNET_CKPT,
    )
    parser.add_argument(
        "--cgan_ckpt",
        type=str,
        default=CGAN_CKPT,
    )
    parser.add_argument(
        "--unet_metrics",
        type=str,
        default=UNET_METRICS,
    )
    parser.add_argument(
        "--cgan_metrics",
        type=str,
        default=CGAN_METRICS,
    )
    parser.add_argument(
        "--test_position",
        type=int,
        default=1,
        help=(
            "Position in the held-out test split. Figure 5.4 uses "
            "position 0, so position 1 is used here by default "
            "to avoid repeating the same sample."
        ),
    )
    parser.add_argument(
        "--max_count",
        type=int,
        default=MAX_COUNT,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_PATH,
    )

    return parser.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model(
    checkpoint_path,
    device,
    state_key,
):
    model = UNet(
        in_channels=2,
        out_channels=NUM_CLASSES,
        base=16,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if state_key not in checkpoint:
        raise KeyError(
            f"Checkpoint {checkpoint_path} does not contain "
            f"the key '{state_key}'."
        )

    model.load_state_dict(
        checkpoint[state_key]
    )
    model.eval()

    return model, checkpoint


def predict_mask(model, inputs):
    with torch.no_grad():
        logits = model(inputs)
        prediction = torch.argmax(
            logits,
            dim=1,
        )

    return (
        prediction[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


def decode_connected_region_count(
    inputs,
    max_count,
):
    count_channel = (
        inputs[0, 1]
        .detach()
        .cpu()
        .numpy()
    )

    normalised_count = float(
        count_channel.max()
    )

    return int(
        round(normalised_count * max_count)
    )


def load_sample_metric(
    csv_path,
    test_position,
    expected_dataset_index,
):
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Metric file not found: {path}"
        )

    frame = pd.read_csv(path)

    required_columns = {
        "idx",
        "dataset_index",
        "miou_no_bg",
        "room_count_error",
    }

    missing = required_columns.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            f"{path} is missing columns: "
            f"{sorted(missing)}"
        )

    selected = frame.loc[
        frame["idx"] == test_position
    ]

    if len(selected) != 1:
        raise ValueError(
            f"{path} does not contain exactly one row "
            f"for test position {test_position}."
        )

    row = selected.iloc[0]

    if int(row["dataset_index"]) != int(
        expected_dataset_index
    ):
        raise ValueError(
            f"Dataset-index mismatch in {path}: "
            f"expected {expected_dataset_index}, "
            f"found {int(row['dataset_index'])}."
        )

    return row


def show_semantic(
    axis,
    mask,
    title,
    subtitle=None,
):
    axis.imshow(
        mask,
        cmap=SEMANTIC_CMAP,
        vmin=0,
        vmax=NUM_CLASSES - 1,
        interpolation="nearest",
    )

    axis.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    if subtitle:
        axis.text(
            0.5,
            -0.08,
            subtitle,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )

    axis.axis("off")


def metric_subtitle(row):
    return (
        f"mIoU {row['miou_no_bg']:.3f} | "
        f"Count error {row['room_count_error']:.0f}"
    )


def main():
    args = parse_args()
    device = get_device()

    dataset = FloorplanNPZDataset(
        args.data_dir,
        max_count=args.max_count,
    )

    split = load_split(
        args.split_path
    )

    test_indices = split["test"]

    if not 0 <= args.test_position < len(test_indices):
        raise IndexError(
            f"test_position must be between 0 and "
            f"{len(test_indices) - 1}."
        )

    dataset_index = int(
        test_indices[args.test_position]
    )

    inputs, target = dataset[dataset_index]

    target_mask = (
        target.detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    inputs = inputs.unsqueeze(0).to(device)

    support_mask = (
        inputs[0, 0]
        .detach()
        .cpu()
        .numpy()
        > 0.5
    ).astype(np.uint8)

    encoded_count = decode_connected_region_count(
        inputs,
        args.max_count,
    )

    unet, unet_checkpoint = load_model(
        args.unet_ckpt,
        device,
        state_key="model_state",
    )

    cgan, cgan_checkpoint = load_model(
        args.cgan_ckpt,
        device,
        state_key="generator_state",
    )

    unet_prediction = predict_mask(
        unet,
        inputs,
    )

    cgan_prediction = predict_mask(
        cgan,
        inputs,
    )

    unet_metric = load_sample_metric(
        args.unet_metrics,
        args.test_position,
        dataset_index,
    )

    cgan_metric = load_sample_metric(
        args.cgan_metrics,
        args.test_position,
        dataset_index,
    )

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(14, 5),
    )

    axes[0].imshow(
        support_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[0].set_title(
        "Binary support condition",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    axes[0].axis("off")

    show_semantic(
        axes[1],
        target_mask,
        "Ground truth",
    )

    show_semantic(
        axes[2],
        unet_prediction,
        "U-Net baseline",
        metric_subtitle(unet_metric),
    )

    show_semantic(
        axes[3],
        cgan_prediction,
        "Pix2Pix-style cGAN",
        metric_subtitle(cgan_metric),
    )

    figure.suptitle(
        (
            "Selected held-out test sample "
            f"(encoded connected-region count: {encoded_count})"
        ),
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    figure.tight_layout(
        rect=(0.01, 0.08, 0.99, 0.92)
    )

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    pdf_path = output_path.with_suffix(
        ".pdf"
    )

    figure.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Device:", device)
    print(
        "U-Net checkpoint epoch:",
        unet_checkpoint.get(
            "epoch",
            "unknown",
        ),
    )
    print(
        "cGAN checkpoint epoch:",
        cgan_checkpoint.get(
            "epoch",
            "unknown",
        ),
    )
    print(
        "Held-out test position:",
        args.test_position,
    )
    print(
        "Dataset index:",
        dataset_index,
    )
    print(
        "Encoded connected-region count:",
        encoded_count,
    )
    print(
        "U-Net sample mIoU:",
        f"{unet_metric['miou_no_bg']:.6f}",
    )
    print(
        "cGAN sample mIoU:",
        f"{cgan_metric['miou_no_bg']:.6f}",
    )
    print("Saved:", output_path)
    print("Saved:", pdf_path)


if __name__ == "__main__":
    main()
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
from src.refinement.hillclimb import refine_semantic_mask_hillclimb
from src.refinement.morphology import refine_semantic_mask_morphology


DATA_DIR = "data/processed_npz_clean_full"
SPLIT_PATH = "outputs/splits/split_seed42_full.json"
UNET_CKPT = "outputs/checkpoints/unet_base16_best.pt"
CGAN_CKPT = "outputs/checkpoints/cgan_unet_patchgan_best.pt"

MAX_COUNT = 32
NUM_CLASSES = 9
BG = 0

OUTPUT_PATH = (
    "outputs/figures/"
    "figure_5_7_refinement_comparison.png"
)

METRIC_FILES = {
    "U-Net baseline": (
        "outputs/metrics_unet_room_count_fixed.csv"
    ),
    "U-Net + morphology": (
        "outputs/"
        "metrics_unet_morphology_room_count_fixed.csv"
    ),
    "U-Net + hill-climbing": (
        "outputs/"
        "metrics_unet_hillclimb_room_count_fixed.csv"
    ),
    "cGAN baseline": (
        "outputs/metrics_cgan_room_count_fixed.csv"
    ),
    "cGAN + morphology": (
        "outputs/"
        "metrics_cgan_morphology_room_count_fixed.csv"
    ),
    "cGAN + hill-climbing": (
        "outputs/"
        "metrics_cgan_hillclimb_room_count_fixed.csv"
    ),
}

# The colours follow the vivid semantic style used in the existing
# qualitative figures. Replace these values with the palette from the
# existing visualisation script if exact colour matching is required.
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
            "Create a qualitative comparison of baseline, morphology "
            "and hill-climbing outputs for one held-out test sample."
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
        "--max_count",
        type=int,
        default=MAX_COUNT,
    )
    parser.add_argument(
        "--test_position",
        type=int,
        default=0,
        help=(
            "Position within the held-out test split. Position 0 is "
            "the same sample used for the first baseline comparison."
        ),
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
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


def load_unet(device, checkpoint_path):
    model = UNet(
        in_channels=2,
        out_channels=NUM_CLASSES,
        base=16,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state" not in checkpoint:
        raise KeyError(
            "The U-Net checkpoint does not contain 'model_state'."
        )

    model.load_state_dict(
        checkpoint["model_state"]
    )
    model.eval()
    return model, checkpoint


def load_cgan_generator(device, checkpoint_path):
    model = UNet(
        in_channels=2,
        out_channels=NUM_CLASSES,
        base=16,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "generator_state" not in checkpoint:
        raise KeyError(
            "The cGAN checkpoint does not contain "
            "'generator_state'."
        )

    model.load_state_dict(
        checkpoint["generator_state"]
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


def expected_room_count(inputs, max_count):
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


def load_sample_metrics(
    test_position,
    expected_dataset_index,
):
    metrics = {}

    for configuration, csv_path in METRIC_FILES.items():
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
            "compact_pred",
            "boundary_violation_rate",
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

        metrics[configuration] = row

    return metrics


def metric_subtitle(row):
    return (
        f"mIoU {row['miou_no_bg']:.3f} | "
        f"Comp. {row['compact_pred']:.3f} | "
        f"BVR {row['boundary_violation_rate']:.3f}"
    )


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
        fontsize=11,
        fontweight="bold",
        pad=7,
    )

    if subtitle:
        axis.text(
            0.5,
            -0.06,
            subtitle,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )

    axis.axis("off")


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

    inputs = inputs.unsqueeze(0).to(device)
    target_mask = (
        target.detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    support_mask = (
        inputs[0, 0]
        .detach()
        .cpu()
        .numpy()
        > 0.5
    ).astype(np.uint8)

    requested_count = expected_room_count(
        inputs,
        args.max_count,
    )

    unet, unet_checkpoint = load_unet(
        device,
        args.unet_ckpt,
    )
    cgan, cgan_checkpoint = load_cgan_generator(
        device,
        args.cgan_ckpt,
    )

    unet_baseline = predict_mask(
        unet,
        inputs,
    )
    cgan_baseline = predict_mask(
        cgan,
        inputs,
    )

    unet_morphology = (
        refine_semantic_mask_morphology(
            unet_baseline,
            num_classes=NUM_CLASSES,
            kernel_size=args.kernel_size,
            min_area=args.min_area,
        ).astype(np.uint8)
    )
    cgan_morphology = (
        refine_semantic_mask_morphology(
            cgan_baseline,
            num_classes=NUM_CLASSES,
            kernel_size=args.kernel_size,
            min_area=args.min_area,
        ).astype(np.uint8)
    )

    unet_hillclimb = (
        refine_semantic_mask_hillclimb(
            unet_baseline,
            num_classes=NUM_CLASSES,
            ignore_classes=(BG,),
            kernel_size=args.kernel_size,
            iterations=args.iterations,
        ).astype(np.uint8)
    )
    cgan_hillclimb = (
        refine_semantic_mask_hillclimb(
            cgan_baseline,
            num_classes=NUM_CLASSES,
            ignore_classes=(BG,),
            kernel_size=args.kernel_size,
            iterations=args.iterations,
        ).astype(np.uint8)
    )

    sample_metrics = load_sample_metrics(
        args.test_position,
        dataset_index,
    )

    figure, axes = plt.subplots(
        3,
        3,
        figsize=(11, 10),
    )

    axes[0, 0].imshow(
        support_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[0, 0].set_title(
        "Binary support condition",
        fontsize=11,
        fontweight="bold",
        pad=7,
    )
    axes[0, 0].axis("off")

    show_semantic(
        axes[0, 1],
        target_mask,
        "Ground truth",
    )

    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.5,
        0.62,
        "Selected held-out sample",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 2].text(
        0.5,
        0.50,
        f"Encoded connected-region count: {requested_count}",
        ha="center",
        va="center",
        fontsize=11,
    )

    show_semantic(
        axes[1, 0],
        unet_baseline,
        "U-Net baseline",
        metric_subtitle(
            sample_metrics["U-Net baseline"]
        ),
    )
    show_semantic(
        axes[1, 1],
        unet_morphology,
        "U-Net + morphology",
        metric_subtitle(
            sample_metrics["U-Net + morphology"]
        ),
    )
    show_semantic(
        axes[1, 2],
        unet_hillclimb,
        "U-Net + hill-climbing",
        metric_subtitle(
            sample_metrics[
                "U-Net + hill-climbing"
            ]
        ),
    )

    show_semantic(
        axes[2, 0],
        cgan_baseline,
        "cGAN baseline",
        metric_subtitle(
            sample_metrics["cGAN baseline"]
        ),
    )
    show_semantic(
        axes[2, 1],
        cgan_morphology,
        "cGAN + morphology",
        metric_subtitle(
            sample_metrics["cGAN + morphology"]
        ),
    )
    show_semantic(
        axes[2, 2],
        cgan_hillclimb,
        "cGAN + hill-climbing",
        metric_subtitle(
            sample_metrics[
                "cGAN + hill-climbing"
            ]
        ),
    )

    figure.text(
        0.5,
        0.015,
        (
            "Comp. = compactness; "
            "BVR = boundary violation rate."
        ),
        ha="center",
        fontsize=9,
    )

    figure.tight_layout(
        rect=(0.02, 0.04, 0.98, 0.98)
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
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Device:", device)
    print(
        "U-Net checkpoint:",
        f"epoch={unet_checkpoint.get('epoch', 'unknown')}",
    )
    print(
        "cGAN checkpoint:",
        f"epoch={cgan_checkpoint.get('epoch', 'unknown')}",
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
        "Requested room count:",
        requested_count,
    )
    print("Saved:", output_path)
    print("Saved:", pdf_path)


if __name__ == "__main__":
    main()

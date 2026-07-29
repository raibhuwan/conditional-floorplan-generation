import argparse
import csv
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset import FloorplanNPZDataset
from src.data.splits import load_split
from src.models.unet import UNet


DEFAULT_DATA_DIR = "data/processed_npz_clean_full"
DEFAULT_SPLIT_PATH = "outputs/splits/split_seed42_full.json"
DEFAULT_MAX_COUNT = 32
NUM_CLASSES = 9

DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-3
DEFAULT_SEED = 42

OUT_SAMPLES = "outputs/train_samples"
OUT_CKPT = "outputs/checkpoints"
OUT_LOGS = "outputs/logs"

os.makedirs(OUT_SAMPLES, exist_ok=True)
os.makedirs(OUT_CKPT, exist_ok=True)
os.makedirs(OUT_LOGS, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the U-Net baseline for semantic floor plan generation."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Folder containing clean processed NPZ files.",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        default=DEFAULT_SPLIT_PATH,
        help="Path to the fixed train/validation/test split JSON file.",
    )
    parser.add_argument(
        "--max_count",
        type=int,
        default=DEFAULT_MAX_COUNT,
        help="Maximum room count used to normalise the room-count condition channel.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="unet_base16_best.pt",
        help="Filename for the best checkpoint saved in outputs/checkpoints.",
    )
    parser.add_argument(
        "--log_csv",
        type=str,
        default="outputs/logs/unet_training_history.csv",
        help="CSV file used to record the U-Net training history.",
    )
    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mean_iou(pred, target, num_classes=NUM_CLASSES, ignore_index=0):
    """
    Calculate mean IoU across classes present in either prediction or target.

    Args:
        pred: Tensor of shape [B, H, W] containing predicted class IDs.
        target: Tensor of shape [B, H, W] containing target class IDs.
        num_classes: Total number of semantic classes.
        ignore_index: Class ID excluded from the calculation. Class 0 is
            excluded so that background does not dominate the metric.
    """
    ious = []

    for class_id in range(num_classes):
        if ignore_index is not None and class_id == ignore_index:
            continue

        pred_class = pred == class_id
        target_class = target == class_id

        intersection = (pred_class & target_class).sum().item()
        union = (pred_class | target_class).sum().item()

        if union == 0:
            continue

        ious.append(intersection / union)

    return float(sum(ious) / len(ious)) if ious else 0.0


def colorize_mask(mask):
    """
    Convert a semantic mask of shape [H, W] into a colour image.
    """
    mapped = ((mask.astype(np.int32) * 29) % 255).astype(np.uint8)
    return cv2.applyColorMap(mapped, cv2.COLORMAP_TURBO)


def save_sample_batch(x, y, pred, epoch, prefix="train"):
    """
    Save up to four qualitative samples containing:
    input support mask, ground-truth mask and predicted mask.
    """
    batch_size = min(4, x.shape[0])

    for sample_index in range(batch_size):
        support_mask = (
            x[sample_index, 0].detach().cpu().numpy() * 255
        ).astype(np.uint8)
        ground_truth = (
            y[sample_index].detach().cpu().numpy().astype(np.uint8)
        )
        prediction = (
            pred[sample_index].detach().cpu().numpy().astype(np.uint8)
        )

        support_rgb = cv2.cvtColor(support_mask, cv2.COLOR_GRAY2BGR)
        ground_truth_rgb = colorize_mask(ground_truth)
        prediction_rgb = colorize_mask(prediction)

        combined = np.concatenate(
            [support_rgb, ground_truth_rgb, prediction_rgb],
            axis=1,
        )

        output_path = os.path.join(
            OUT_SAMPLES,
            f"{prefix}_epoch{epoch:03d}_sample{sample_index}.png",
        )
        cv2.imwrite(output_path, combined)


def initialise_log(log_path):
    """
    Create a new CSV training log and write the header row.
    """
    log_directory = os.path.dirname(log_path)
    if log_directory:
        os.makedirs(log_directory, exist_ok=True)

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            [
                "epoch",
                "epoch_time_seconds",
                "train_loss",
                "train_miou",
                "validation_loss",
                "validation_miou",
                "checkpoint_saved",
            ]
        )


def append_log_row(
    log_path,
    epoch,
    epoch_time,
    train_loss,
    train_iou,
    val_loss,
    val_iou,
    checkpoint_saved,
):
    """
    Append one epoch of training results to the CSV log.
    """
    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            [
                epoch,
                f"{epoch_time:.6f}",
                f"{train_loss:.8f}",
                f"{train_iou:.8f}",
                f"{val_loss:.8f}",
                f"{val_iou:.8f}",
                int(checkpoint_saved),
            ]
        )


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print("Device:", device)

    dataset = FloorplanNPZDataset(
        args.data_dir,
        max_count=args.max_count,
    )

    split = load_split(args.split_path)

    train_dataset = Subset(dataset, split["train"])
    validation_dataset = Subset(dataset, split["val"])
    test_count = len(split["test"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Data folder: {args.data_dir}")
    print(f"Split file: {args.split_path}")
    print(f"MAX_COUNT: {args.max_count}")
    print(
        "Samples: "
        f"total={len(dataset)}, "
        f"train={len(train_dataset)}, "
        f"val={len(validation_dataset)}, "
        f"test={test_count}"
    )

    model = UNet(
        in_channels=2,
        out_channels=NUM_CLASSES,
        base=16,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss()

    best_validation_iou = -1.0
    initialise_log(args.log_csv)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # -------------------------
        # Training
        # -------------------------
        model.train()

        total_train_loss = 0.0
        total_train_iou = 0.0
        train_steps = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)

                total_train_iou += mean_iou(
                    predictions,
                    targets,
                    num_classes=NUM_CLASSES,
                    ignore_index=0,
                )
                total_train_loss += loss.item()
                train_steps += 1

        train_loss = total_train_loss / max(1, train_steps)
        train_iou = total_train_iou / max(1, train_steps)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()

        total_validation_loss = 0.0
        total_validation_iou = 0.0
        validation_steps = 0

        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                logits = model(inputs)
                loss = criterion(logits, targets)
                predictions = torch.argmax(logits, dim=1)

                total_validation_iou += mean_iou(
                    predictions,
                    targets,
                    num_classes=NUM_CLASSES,
                    ignore_index=0,
                )
                total_validation_loss += loss.item()
                validation_steps += 1

        validation_loss = (
            total_validation_loss / max(1, validation_steps)
        )
        validation_iou = (
            total_validation_iou / max(1, validation_steps)
        )

        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch {epoch:03d} time: {epoch_time:.1f}s | "
            f"train loss={train_loss:.4f} iou={train_iou:.3f} | "
            f"val loss={validation_loss:.4f} "
            f"iou={validation_iou:.3f}"
        )

        # Save qualitative validation samples.
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs = inputs.to(device)

                logits = model(inputs)
                predictions = torch.argmax(logits, dim=1)

                save_sample_batch(
                    inputs,
                    targets,
                    predictions,
                    epoch,
                    prefix="val",
                )
                break

        checkpoint_saved = validation_iou > best_validation_iou

        if checkpoint_saved:
            best_validation_iou = validation_iou

            checkpoint_path = os.path.join(
                OUT_CKPT,
                args.checkpoint_name,
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_iou": best_validation_iou,
                    "config": {
                        "data_dir": args.data_dir,
                        "split_path": args.split_path,
                        "max_count": args.max_count,
                        "num_classes": NUM_CLASSES,
                        "batch_size": args.batch_size,
                        "epochs": args.epochs,
                        "lr": args.lr,
                        "seed": args.seed,
                        "validation_ignore_index": 0,
                        "log_csv": args.log_csv,
                    },
                },
                checkpoint_path,
            )

            print(
                "Saved best U-Net checkpoint with "
                f"val IoU={best_validation_iou:.3f}"
            )

        append_log_row(
            args.log_csv,
            epoch,
            epoch_time,
            train_loss,
            train_iou,
            validation_loss,
            validation_iou,
            checkpoint_saved,
        )

    print("Training finished.")
    print("Best validation IoU:", best_validation_iou)
    print("Training log saved to:", args.log_csv)


if __name__ == "__main__":
    main()

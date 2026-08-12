import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.dataset import FloorplanNPZDataset
from src.data.splits import load_split
from src.models.patchgan import PatchDiscriminator
from src.models.unet import UNet


DEFAULT_DATA_DIR = "data/processed_npz_clean_full"
DEFAULT_SPLIT_PATH = "outputs/splits/split_seed42_full.json"

OUT_CKPT = "outputs/checkpoints"
OUT_LOGS = "outputs/logs"

os.makedirs(OUT_CKPT, exist_ok=True)
os.makedirs(OUT_LOGS, exist_ok=True)

DEFAULT_MAX_COUNT = 32
NUM_CLASSES = 9

DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 30
DEFAULT_LR_G = 1e-4
DEFAULT_LR_D = 1e-5
DEFAULT_LAMBDA_CE = 30.0
DEFAULT_LAMBDA_GAN = 0.05
DEFAULT_SEED = 42


# Parse command-line options used to configure cGAN training.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Pix2Pix-style cGAN for semantic floor plan generation."
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
        help="Maximum encoded connected-region used to normalise the room-count condition channel.",
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
        "--lr_g",
        type=float,
        default=DEFAULT_LR_G,
        help="Generator learning rate.",
    )
    parser.add_argument(
        "--lr_d",
        type=float,
        default=DEFAULT_LR_D,
        help="Discriminator learning rate.",
    )
    parser.add_argument(
        "--lambda_ce",
        type=float,
        default=DEFAULT_LAMBDA_CE,
        help="Weight for the generator cross-entropy loss.",
    )
    parser.add_argument(
        "--lambda_gan",
        type=float,
        default=DEFAULT_LAMBDA_GAN,
        help="Weight for the generator adversarial loss.",
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
        default="cgan_unet_patchgan_best.pt",
        help="Filename for the best checkpoint saved in outputs/checkpoints.",
    )
    parser.add_argument(
        "--log_csv",
        type=str,
        default="outputs/logs/cgan_training_history.csv",
        help="CSV file used to record the cGAN training history.",
    )
    return parser.parse_args()


# Set Python, NumPy and PyTorch random seeds for reproducible execution.
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Select Apple MPS acceleration when available, otherwise use the CPU.
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_one_hot(mask, num_classes):
    """
    Convert a class-ID mask [B, H, W] to one-hot format [B, C, H, W].
    """

    # Move the class dimension before the spatial dimensions so the result
    # can be concatenated with the conditional channels for the discriminator.
    return (
        F.one_hot(mask, num_classes=num_classes)
        .permute(0, 3, 1, 2)
        .float()
    )


def mean_iou(pred, target, num_classes=NUM_CLASSES, ignore_index=0):
    """
    Calculate mean IoU while excluding background class 0.
    """
    ious = []

    # Calculate IoU independently for each semantic class.
    for class_id in range(num_classes):
        if ignore_index is not None and class_id == ignore_index:
            continue

        pred_class = pred == class_id
        target_class = target == class_id

        intersection = (pred_class & target_class).sum().item()
        union = (pred_class | target_class).sum().item()

        # Skip classes that are absent from both prediction and target.
        if union == 0:
            continue

        ious.append(intersection / union)

    return float(sum(ious) / len(ious)) if ious else 0.0


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
                "discriminator_loss",
                "generator_loss",
                "cross_entropy_loss",
                "adversarial_loss",
                "train_miou",
                "validation_cross_entropy",
                "validation_miou",
                "checkpoint_saved",
            ]
        )


def append_log_row(
    log_path,
    epoch,
    epoch_time,
    discriminator_loss,
    generator_loss,
    cross_entropy_loss,
    adversarial_loss,
    train_iou,
    validation_cross_entropy,
    validation_iou,
    checkpoint_saved,
):
    """
    Append one epoch of cGAN training results to the CSV log.
    """
    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            [
                epoch,
                f"{epoch_time:.6f}",
                f"{discriminator_loss:.8f}",
                f"{generator_loss:.8f}",
                f"{cross_entropy_loss:.8f}",
                f"{adversarial_loss:.8f}",
                f"{train_iou:.8f}",
                f"{validation_cross_entropy:.8f}",
                f"{validation_iou:.8f}",
                int(checkpoint_saved),
            ]
        )


# Run the complete cGAN training and validation workflow.
def main():
    args = parse_args()

    # Apply the selected random seed before model and data-loader setup.
    set_seed(args.seed)

    device = get_device()
    print("Device:", device)

    # Load the processed samples used to construct conditional inputs and targets.
    dataset = FloorplanNPZDataset(
        args.data_dir,
        max_count=args.max_count,
    )

    # Load the fixed train, validation and test partition.
    split = load_split(args.split_path)

    # Use training samples for parameter learning and validation samples
    # for checkpoint selection. The test partition remains separate.
    train_dataset = Subset(dataset, split["train"])
    validation_dataset = Subset(dataset, split["val"])
    test_count = len(split["test"])

    # Shuffle the training batches while keeping validation order fixed.
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
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"LR_G: {args.lr_g}")
    print(f"LR_D: {args.lr_d}")
    print(f"LAMBDA_CE: {args.lambda_ce}")
    print(f"LAMBDA_GAN: {args.lambda_gan}")
    print(
        "Samples: "
        f"total={len(dataset)}, "
        f"train={len(train_dataset)}, "
        f"val={len(validation_dataset)}, "
        f"test={test_count}"
    )

    # Use the same U-Net structure as the supervised model to generate
    # nine-class semantic predictions from the two-channel condition.
    generator = UNet(
        in_channels=2,
        out_channels=NUM_CLASSES,
        base=16,
    ).to(device)

    # Create the PatchGAN discriminator that evaluates the condition
    # together with a nine-channel semantic representation.
    discriminator = PatchDiscriminator(
        condition_channels=2,
        mask_channels=NUM_CLASSES,
        base=32,
    ).to(device)

    # Use separate Adam optimisers because generator and discriminator
    # parameters are updated independently during adversarial training.
    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr_g,
        betas=(0.5, 0.999),
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr_d,
        betas=(0.5, 0.999),
    )

    # Cross-entropy measures semantic prediction error, while binary
    # cross-entropy provides the adversarial real/fake training objective.
    cross_entropy_loss = nn.CrossEntropyLoss()
    adversarial_loss = nn.BCEWithLogitsLoss()

    # Track the strongest validation mIoU for checkpoint selection.
    best_validation_iou = -1.0
    initialise_log(args.log_csv)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # Enable training behaviour for both networks.
        generator.train()
        discriminator.train()

        total_generator_loss = 0.0
        total_discriminator_loss = 0.0
        total_cross_entropy = 0.0
        total_adversarial = 0.0
        total_train_iou = 0.0
        train_steps = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # -------------------------
            # 1. Train discriminator
            # -------------------------

            # Generate predictions without retaining generator gradients because
            # this stage updates only the discriminator.
            with torch.no_grad():
                detached_fake_logits = generator(inputs)
                detached_fake_probabilities = torch.softmax(
                    detached_fake_logits,
                    dim=1,
                )

            # Convert ground-truth class IDs into nine one-hot semantic channels.
            real_one_hot = to_one_hot(
                targets,
                NUM_CLASSES,
            ).to(device)

            # Evaluate real condition-target pairs and generated condition-output pairs.
            discriminator_real_logits = discriminator(
                inputs,
                real_one_hot,
            )
            discriminator_fake_logits = discriminator(
                inputs,
                detached_fake_probabilities.detach(),
            )

            # Use one-sided label smoothing for real pairs and zero targets for
            # generated pairs during discriminator training.
            real_targets = torch.full_like(
                discriminator_real_logits,
                0.9,
            )
            fake_targets = torch.zeros_like(
                discriminator_fake_logits
            )

            # Calculate discriminator loss for both real and generated pairs.
            discriminator_real_loss = adversarial_loss(
                discriminator_real_logits,
                real_targets,
            )
            discriminator_fake_loss = adversarial_loss(
                discriminator_fake_logits,
                fake_targets,
            )
            discriminator_loss = 0.5 * (
                discriminator_real_loss
                + discriminator_fake_loss
            )

            # Update discriminator parameters only.
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # -------------------------
            # 2. Train generator
            # -------------------------

            # Generate a new prediction with gradients enabled for generator training.
            fake_logits = generator(inputs)

            # Convert generator logits into semantic probabilities before
            # supplying the generated layout to the discriminator.
            fake_probabilities = torch.softmax(
                fake_logits,
                dim=1,
            )

            # Ask the discriminator to evaluate the generated condition-layout pair.
            discriminator_fake_for_generator = discriminator(
                inputs,
                fake_probabilities,
            )

            # Encourage generated pairs to be classified as real by the discriminator.
            generator_adversarial_loss = adversarial_loss(
                discriminator_fake_for_generator,
                torch.ones_like(
                    discriminator_fake_for_generator
                ),
            )

            # Measure pixel-level semantic disagreement with the ground-truth mask.
            generator_cross_entropy_loss = cross_entropy_loss(
                fake_logits,
                targets,
            )

            # Combine the weighted supervised and adversarial objectives.
            generator_loss = (
                args.lambda_ce * generator_cross_entropy_loss
                + args.lambda_gan * generator_adversarial_loss
            )

            # Back-propagate the combined objective through the generator.
            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            # Convert logits to class predictions for training-metric monitoring.
            with torch.no_grad():
                predictions = torch.argmax(
                    fake_logits,
                    dim=1,
                )

                batch_iou = mean_iou(
                    predictions,
                    targets,
                    num_classes=NUM_CLASSES,
                    ignore_index=0,
                )

            # Accumulate batch-level losses and mIoU for epoch-level reporting.
            total_generator_loss += generator_loss.item()
            total_discriminator_loss += discriminator_loss.item()
            total_cross_entropy += generator_cross_entropy_loss.item()
            total_adversarial += generator_adversarial_loss.item()
            total_train_iou += batch_iou
            train_steps += 1

        # Average the recorded training values over all batches in the epoch.
        train_generator_loss = (
            total_generator_loss / max(1, train_steps)
        )
        train_discriminator_loss = (
            total_discriminator_loss / max(1, train_steps)
        )
        train_cross_entropy = (
            total_cross_entropy / max(1, train_steps)
        )
        train_adversarial = (
            total_adversarial / max(1, train_steps)
        )
        train_iou = (
            total_train_iou / max(1, train_steps)
        )

        # -------------------------
        # Validation
        # -------------------------

        # Only the generator is required for semantic validation.
        generator.eval()

        total_validation_iou = 0.0
        total_validation_cross_entropy = 0.0
        validation_steps = 0

        # Validation does not update generator or discriminator parameters.
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Generate validation predictions from the fixed condition.
                logits = generator(inputs)
                predictions = torch.argmax(
                    logits,
                    dim=1,
                )

                # Record supervised validation loss for monitoring.
                total_validation_cross_entropy += (
                    cross_entropy_loss(
                        logits,
                        targets,
                    ).item()
                )

                # Calculate validation mIoU using the same class handling
                # applied during training monitoring.
                total_validation_iou += mean_iou(
                    predictions,
                    targets,
                    num_classes=NUM_CLASSES,
                    ignore_index=0,
                )

                validation_steps += 1

        # Average validation values across all validation batches.
        validation_iou = (
            total_validation_iou / max(1, validation_steps)
        )
        validation_cross_entropy = (
            total_validation_cross_entropy
            / max(1, validation_steps)
        )

        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch {epoch:03d} | "
            f"time={epoch_time:.1f}s | "
            f"D={train_discriminator_loss:.4f} | "
            f"G={train_generator_loss:.4f} | "
            f"CE={train_cross_entropy:.4f} | "
            f"GAN={train_adversarial:.4f} | "
            f"train IoU={train_iou:.3f} | "
            f"val CE={validation_cross_entropy:.4f} | "
            f"val IoU={validation_iou:.3f}"
        )

        # Retain a checkpoint only when validation mIoU improves.
        # The held-out test set is not used for checkpoint selection.
        checkpoint_saved = (
            validation_iou > best_validation_iou
        )

        if checkpoint_saved:
            best_validation_iou = validation_iou

            checkpoint_path = os.path.join(
                OUT_CKPT,
                args.checkpoint_name,
            )

            # Save both network states and the configuration needed to
            # identify the selected cGAN training setup.
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state": generator.state_dict(),
                    "discriminator_state": discriminator.state_dict(),
                    "val_iou": best_validation_iou,
                    "config": {
                        "data_dir": args.data_dir,
                        "split_path": args.split_path,
                        "max_count": args.max_count,
                        "num_classes": NUM_CLASSES,
                        "batch_size": args.batch_size,
                        "epochs": args.epochs,
                        "lr_g": args.lr_g,
                        "lr_d": args.lr_d,
                        "lambda_ce": args.lambda_ce,
                        "lambda_gan": args.lambda_gan,
                        "seed": args.seed,
                        "validation_ignore_index": 0,
                        "log_csv": args.log_csv,
                    },
                },
                checkpoint_path,
            )

            print(
                "Saved best cGAN checkpoint with "
                f"val IoU={best_validation_iou:.3f}"
            )

        # Record the current epoch for later training-history analysis.
        append_log_row(
            args.log_csv,
            epoch,
            epoch_time,
            train_discriminator_loss,
            train_generator_loss,
            train_cross_entropy,
            train_adversarial,
            train_iou,
            validation_cross_entropy,
            validation_iou,
            checkpoint_saved,
        )

    print("Training finished.")
    print("Best validation IoU:", best_validation_iou)
    print("Training log saved to:", args.log_csv)


if __name__ == "__main__":
    main()
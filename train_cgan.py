import argparse
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
from src.models.unet import UNet
from src.models.patchgan import PatchDiscriminator


# -----------------------------
# Config
# -----------------------------
DEFAULT_DATA_DIR = "data/processed_npz_clean_full"
DEFAULT_SPLIT_PATH = "outputs/splits/split_seed42_full.json"
OUT_CKPT = "outputs/checkpoints"
os.makedirs(OUT_CKPT, exist_ok=True)

DEFAULT_MAX_COUNT = 32
NUM_CLASSES = 9

DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 30
DEFAULT_LR_G = 1e-4

DEFAULT_SEED = 42

# Default cGAN configuration used for the full high_quality_architectural run.
DEFAULT_LR_D = 1e-5
DEFAULT_LAMBDA_CE = 30.0
DEFAULT_LAMBDA_GAN = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Pix2Pix-style cGAN for semantic floor plan generation.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Folder containing clean processed NPZ files.")
    parser.add_argument("--split_path", type=str, default=DEFAULT_SPLIT_PATH, help="Path to the fixed train/val/test split JSON file.")
    parser.add_argument("--max_count", type=int, default=DEFAULT_MAX_COUNT, help="Maximum room count used to normalise the room-count condition channel.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--lr_g", type=float, default=DEFAULT_LR_G, help="Generator learning rate.")
    parser.add_argument("--lr_d", type=float, default=DEFAULT_LR_D, help="Discriminator learning rate.")
    parser.add_argument("--lambda_ce", type=float, default=DEFAULT_LAMBDA_CE, help="Weight for the generator cross-entropy loss.")
    parser.add_argument("--lambda_gan", type=float, default=DEFAULT_LAMBDA_GAN, help="Weight for the generator adversarial loss.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility.")
    parser.add_argument("--checkpoint_name", type=str, default="cgan_unet_patchgan_best.pt", help="Filename for the best checkpoint saved in outputs/checkpoints.")
    return parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_one_hot(mask, num_classes):
    """
    mask: [B,H,W]
    output: [B,C,H,W]
    """
    return F.one_hot(mask, num_classes=num_classes).permute(0, 3, 1, 2).float()


def mean_iou(pred, target, num_classes=NUM_CLASSES, ignore_index=0):
    ious = []

    for c in range(num_classes):
        if c == ignore_index:
            continue

        pred_c = pred == c
        tgt_c = target == c

        inter = (pred_c & tgt_c).sum().item()
        union = (pred_c | tgt_c).sum().item()

        if union == 0:
            continue

        ious.append(inter / union)

    return float(sum(ious) / len(ious)) if ious else 0.0


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    dataset = FloorplanNPZDataset(args.data_dir, max_count=args.max_count)

    # fixed train/validation/test split
    n = len(dataset)
    split = load_split(args.split_path)

    train_ds = Subset(dataset, split["train"])
    val_ds = Subset(dataset, split["val"])
    test_count = len(split["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
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
    print(f"Samples: total={n}, train={len(train_ds)}, val={len(val_ds)}, test={test_count}")

    # Generator = your U-Net
    generator = UNet(
        in_channels=2,
        out_channels=NUM_CLASSES,
        base=16
    ).to(device)

    # Discriminator = PatchGAN
    discriminator = PatchDiscriminator(
        condition_channels=2,
        mask_channels=NUM_CLASSES,
        base=32
    ).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    ce_loss = nn.CrossEntropyLoss()
    adv_loss = nn.BCEWithLogitsLoss()

    best_val_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        generator.train()
        discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0
        total_ce = 0.0
        total_gan = 0.0
        total_iou = 0.0
        steps = 0

        for x, y in train_loader:
            x = x.to(device)  # condition [B,2,H,W]
            y = y.to(device)  # real labels [B,H,W]

            # -------------------------
            # 1) Train Discriminator
            # -------------------------
            with torch.no_grad():
                fake_logits = generator(x)
                fake_probs = torch.softmax(fake_logits, dim=1)

            real_onehot = to_one_hot(y, NUM_CLASSES).to(device)

            d_real_logits = discriminator(x, real_onehot)
            d_fake_logits = discriminator(x, fake_probs.detach())

            # real_targets = torch.ones_like(d_real_logits)
            # fake_targets = torch.zeros_like(d_fake_logits)
            real_targets = torch.full_like(d_real_logits, 0.9)
            fake_targets = torch.zeros_like(d_fake_logits)

            d_real_loss = adv_loss(d_real_logits, real_targets)
            d_fake_loss = adv_loss(d_fake_logits, fake_targets)
            d_loss = 0.5 * (d_real_loss + d_fake_loss)

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # -------------------------
            # 2) Train Generator
            # -------------------------
            fake_logits = generator(x)
            fake_probs = torch.softmax(fake_logits, dim=1)

            d_fake_for_g = discriminator(x, fake_probs)
            g_adv = adv_loss(d_fake_for_g, torch.ones_like(d_fake_for_g))
            g_ce = ce_loss(fake_logits, y)

            g_loss = (args.lambda_ce * g_ce) + (args.lambda_gan * g_adv)

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            with torch.no_grad():
                pred = torch.argmax(fake_logits, dim=1)
                batch_iou = mean_iou(pred, y)

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            total_ce += g_ce.item()
            total_gan += g_adv.item()
            total_iou += batch_iou
            steps += 1

        train_g_loss = total_g_loss / max(1, steps)
        train_d_loss = total_d_loss / max(1, steps)
        train_ce = total_ce / max(1, steps)
        train_gan = total_gan / max(1, steps)
        train_iou = total_iou / max(1, steps)

        # -------------------------
        # Validation
        # -------------------------
        generator.eval()
        val_iou = 0.0
        val_ce = 0.0
        vsteps = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = generator(x)
                pred = torch.argmax(logits, dim=1)

                val_ce += ce_loss(logits, y).item()
                val_iou += mean_iou(pred, y)
                vsteps += 1

        val_iou /= max(1, vsteps)
        val_ce /= max(1, vsteps)

        print(
            f"Epoch {epoch:03d} | "
            f"time={time.time() - t0:.1f}s | "
            f"D={train_d_loss:.4f} | "
            f"G={train_g_loss:.4f} | "
            f"CE={train_ce:.4f} | "
            f"GAN={train_gan:.4f} | "
            f"train IoU={train_iou:.3f} | "
            f"val CE={val_ce:.4f} | "
            f"val IoU={val_iou:.3f}"
        )

        # Save best generator
        if val_iou > best_val_iou:
            best_val_iou = val_iou

            torch.save(
                {
                    "epoch": epoch,
                    "generator_state": generator.state_dict(),
                    "discriminator_state": discriminator.state_dict(),
                    "val_iou": best_val_iou,
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
                    },
                },
                os.path.join(OUT_CKPT, args.checkpoint_name)
            )

            print(f"Saved best cGAN checkpoint with val IoU={best_val_iou:.3f}")

    print("Training finished.")
    print("Best validation IoU:", best_val_iou)


if __name__ == "__main__":
    main()
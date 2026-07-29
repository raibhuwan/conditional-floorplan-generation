import argparse
import csv
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import FloorplanNPZDataset
from src.data.splits import load_split
from src.models.unet import UNet

NUM_CLASSES = 9
BACKGROUND_CLASS = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the selected U-Net and cGAN generator checkpoints on the same "
            "validation split using the same no-background mIoU definition."
        )
    )
    parser.add_argument("--data_dir", default="data/processed_npz_clean_full")
    parser.add_argument("--split_path", default="outputs/splits/split_seed42_full.json")
    parser.add_argument("--unet_ckpt", default="outputs/checkpoints/unet_base16_best.pt")
    parser.add_argument("--cgan_ckpt", default="outputs/checkpoints/cgan_unet_patchgan_best.pt")
    parser.add_argument("--max_count", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_csv", default="outputs/validation_common_miou.csv")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sample_miou_no_background(pred: np.ndarray, target: np.ndarray) -> float:
    ious = []
    for class_id in range(NUM_CLASSES):
        if class_id == BACKGROUND_CLASS:
            continue
        pred_class = pred == class_id
        target_class = target == class_id
        union = np.logical_or(pred_class, target_class).sum()
        if union == 0:
            continue
        intersection = np.logical_and(pred_class, target_class).sum()
        ious.append(float(intersection / union))
    return float(np.mean(ious)) if ious else 0.0


def load_unet_checkpoint(path: str, key: str, device: torch.device) -> tuple[UNet, Dict]:
    checkpoint = torch.load(path, map_location=device)
    if key not in checkpoint:
        raise KeyError(f"Checkpoint {path!r} does not contain key {key!r}.")
    model = UNet(in_channels=2, out_channels=NUM_CLASSES, base=16).to(device)
    model.load_state_dict(checkpoint[key])
    model.eval()
    return model, checkpoint


def main() -> None:
    args = parse_args()
    device = get_device()

    dataset = FloorplanNPZDataset(args.data_dir, max_count=args.max_count)
    split = load_split(args.split_path)
    validation_dataset = Subset(dataset, split["val"])
    loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    unet, unet_checkpoint = load_unet_checkpoint(
        args.unet_ckpt, "model_state", device
    )
    cgan_generator, cgan_checkpoint = load_unet_checkpoint(
        args.cgan_ckpt, "generator_state", device
    )

    rows = []
    running_index = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            unet_predictions = torch.argmax(unet(inputs), dim=1)
            cgan_predictions = torch.argmax(cgan_generator(inputs), dim=1)

            for batch_index in range(inputs.shape[0]):
                target = targets[batch_index].cpu().numpy().astype(np.uint8)
                unet_pred = unet_predictions[batch_index].cpu().numpy().astype(np.uint8)
                cgan_pred = cgan_predictions[batch_index].cpu().numpy().astype(np.uint8)

                unet_miou = sample_miou_no_background(unet_pred, target)
                cgan_miou = sample_miou_no_background(cgan_pred, target)

                rows.append(
                    {
                        "validation_position": running_index,
                        "dataset_index": int(split["val"][running_index]),
                        "unet_miou_no_bg": unet_miou,
                        "cgan_miou_no_bg": cgan_miou,
                        "unet_minus_cgan": unet_miou - cgan_miou,
                    }
                )
                running_index += 1

    if not rows:
        raise RuntimeError("No validation samples were evaluated.")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    unet_mean = float(np.mean([row["unet_miou_no_bg"] for row in rows]))
    cgan_mean = float(np.mean([row["cgan_miou_no_bg"] for row in rows]))
    difference_mean = float(np.mean([row["unet_minus_cgan"] for row in rows]))

    print(f"Device: {device}")
    print(f"Validation samples: {len(rows)}")
    print(
        "Selected U-Net checkpoint: "
        f"epoch={unet_checkpoint.get('epoch', 'unknown')}, "
        f"stored_val_iou={unet_checkpoint.get('val_iou', 'unknown')}"
    )
    print(
        "Selected cGAN checkpoint: "
        f"epoch={cgan_checkpoint.get('epoch', 'unknown')}, "
        f"stored_val_iou={cgan_checkpoint.get('val_iou', 'unknown')}"
    )
    print(f"Common validation U-Net mIoU (no background): {unet_mean:.6f}")
    print(f"Common validation cGAN mIoU (no background): {cgan_mean:.6f}")
    print(f"Mean paired difference (U-Net - cGAN): {difference_mean:.6f}")
    print(f"Saved per-sample results to: {args.out_csv}")


if __name__ == "__main__":
    main()

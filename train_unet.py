import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import cv2

from src.data.dataset import FloorplanNPZDataset
from src.models.unet import UNet
import time

DATA_DIR = "data/processed_npz_clean"
MAX_COUNT = 17
NUM_CLASSES = 9

BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-3
SEED = 42

OUT_SAMPLES = "outputs/train_samples"
OUT_CKPT = "outputs/checkpoints"
os.makedirs(OUT_SAMPLES, exist_ok=True)
os.makedirs(OUT_CKPT, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mean_iou(pred, target, num_classes=NUM_CLASSES, ignore_index=None):
    """
    pred: [B,H,W] int
    target: [B,H,W] int
    Returns mean IoU across classes present in target (excluding background if you want).
    """
    ious = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        pred_c = (pred == c)
        tgt_c = (target == c)

        inter = (pred_c & tgt_c).sum().item()
        union = (pred_c | tgt_c).sum().item()
        if union == 0:
            continue
        ious.append(inter / union)
    if len(ious) == 0:
        return 0.0
    return float(sum(ious) / len(ious))


def colorize_mask(mask):
    """
    mask: [H,W] int -> color image for saving
    Simple deterministic colormap using OpenCV.
    """
    m = ((mask.astype(np.int32) * 29) % 255).astype(np.uint8)
    return cv2.applyColorMap(m, cv2.COLORMAP_TURBO)


def save_sample_batch(x, y, pred, epoch, prefix="train"):
    """
    Save a small grid of images for qualitative checks.
    We'll save 3 images per sample: outline, gt mask, pred mask.
    """
    b = min(4, x.shape[0])
    for i in range(b):
        outline = (x[i, 0].detach().cpu().numpy() * 255).astype(np.uint8)
        gt = y[i].detach().cpu().numpy().astype(np.uint8)
        pr = pred[i].detach().cpu().numpy().astype(np.uint8)

        outline_rgb = cv2.cvtColor(outline, cv2.COLOR_GRAY2BGR)
        gt_rgb = colorize_mask(gt)
        pr_rgb = colorize_mask(pr)

        combo = np.concatenate([outline_rgb, gt_rgb, pr_rgb], axis=1)
        out_path = os.path.join(OUT_SAMPLES, f"{prefix}_epoch{epoch:03d}_sample{i}.png")
        cv2.imwrite(out_path, combo)


def main():
    set_seed(SEED)
    device = get_device()
    print("Device:", device)

    ds = FloorplanNPZDataset(DATA_DIR, max_count=MAX_COUNT)

    # ---- train/val split ----
    n = len(ds)
    idxs = list(range(n))
    random.shuffle(idxs)

    val_size = max(1, int(0.2 * n))
    val_idxs = idxs[:val_size]
    train_idxs = idxs[val_size:]

    train_ds = Subset(ds, train_idxs)
    val_ds = Subset(ds, val_idxs)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Samples: total={n}, train={len(train_ds)}, val={len(val_ds)}")

    model = UNet(in_channels=2, out_channels=NUM_CLASSES, base=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0

    for epoch in range(1, EPOCHS + 1):

        t0 = time.time()

        # ---- train ----
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        steps = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)                 # [B,9,H,W]
            loss = criterion(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)  # [B,H,W]
                train_iou += mean_iou(pred, y, num_classes=NUM_CLASSES)
                train_loss += loss.item()
                steps += 1

        train_loss /= max(1, steps)
        train_iou /= max(1, steps)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        vsteps = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)
                pred = torch.argmax(logits, dim=1)

                val_iou += mean_iou(pred, y, num_classes=NUM_CLASSES)
                val_loss += loss.item()
                vsteps += 1

            val_loss /= max(1, vsteps)
            val_iou /= max(1, vsteps)

        print(f"Epoch {epoch:03d} time: {time.time()-t0:.1f}s | train loss={train_loss:.4f} iou={train_iou:.3f} | val loss={val_loss:.4f} iou={val_iou:.3f}")

        # Save sample images every epoch (use first val batch if available)
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1)
                save_sample_batch(x, y, pred, epoch, prefix="val")
                break

        # Save best checkpoint
        if val_iou > best_val:
            best_val = val_iou
            ckpt_path = os.path.join(OUT_CKPT, "unet_base16_best.pt")
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_iou": best_val},
                ckpt_path
            )

    print("Training finished. Best val IoU:", best_val)


if __name__ == "__main__":
    main()
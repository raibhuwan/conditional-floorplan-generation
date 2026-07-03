import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import FloorplanNPZDataset
from src.models.unet import UNet


DATA_DIR = "data/processed_npz_clean_full"

BASELINE_CKPT = "outputs/checkpoints/unet_base16_best.pt"
CGAN_CKPT = "outputs/checkpoints/cgan_unet_patchgan_best.pt"

OUT_DIR = "outputs/comparison_samples"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_COUNT = 32
NUM_CLASSES = 9
NUM_SAVE = 20


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def colorize(mask):
    m = ((mask.astype(np.int32) * 29) % 255).astype(np.uint8)
    return cv2.applyColorMap(m, cv2.COLORMAP_TURBO)


def load_unet_checkpoint(path, device, is_cgan=False):
    model = UNet(in_channels=2, out_channels=NUM_CLASSES, base=16).to(device)
    ckpt = torch.load(path, map_location=device)

    if is_cgan:
        model.load_state_dict(ckpt["generator_state"])
    else:
        model.load_state_dict(ckpt["model_state"])

    model.eval()
    return model


def add_title(img, title):
    out = img.copy()
    cv2.putText(
        out,
        title,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def main():
    device = get_device()
    print("Device:", device)

    ds = FloorplanNPZDataset(DATA_DIR, max_count=MAX_COUNT)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    baseline = load_unet_checkpoint(BASELINE_CKPT, device, is_cgan=False)
    cgan = load_unet_checkpoint(CGAN_CKPT, device, is_cgan=True)

    saved = 0

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            if saved >= NUM_SAVE:
                break

            x = x.to(device)

            gt = y[0].numpy().astype(np.uint8)
            outline = (x[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

            base_logits = baseline(x)
            cgan_logits = cgan(x)

            base_pred = torch.argmax(base_logits, dim=1)[0].cpu().numpy().astype(np.uint8)
            cgan_pred = torch.argmax(cgan_logits, dim=1)[0].cpu().numpy().astype(np.uint8)

            outline_rgb = cv2.cvtColor((outline * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            gt_rgb = colorize(gt)
            base_rgb = colorize(base_pred)
            cgan_rgb = colorize(cgan_pred)

            outline_rgb = add_title(outline_rgb, "Input outline")
            gt_rgb = add_title(gt_rgb, "Ground truth")
            base_rgb = add_title(base_rgb, "U-Net baseline")
            cgan_rgb = add_title(cgan_rgb, "cGAN")

            top = np.concatenate([outline_rgb, gt_rgb], axis=1)
            bottom = np.concatenate([base_rgb, cgan_rgb], axis=1)
            grid = np.concatenate([top, bottom], axis=0)

            out_path = os.path.join(OUT_DIR, f"comparison_{idx:04d}.png")
            cv2.imwrite(out_path, grid)

            saved += 1
            print("Saved:", out_path)

    print(f"\nSaved {saved} comparison images to:", OUT_DIR)


if __name__ == "__main__":
    main()
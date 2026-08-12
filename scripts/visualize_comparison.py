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


# Select Apple MPS acceleration when available, otherwise use the CPU.
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Convert a semantic class-ID mask into a colour image for visual comparison.
def colorize(mask):
    # Map class IDs to display values before applying the OpenCV colour map.
    m = ((mask.astype(np.int32) * 29) % 255).astype(np.uint8)
    return cv2.applyColorMap(m, cv2.COLORMAP_TURBO)


# Load the shared U-Net architecture from either a baseline or cGAN checkpoint.
def load_unet_checkpoint(path, device, is_cgan=False):
    model = UNet(in_channels=2, out_channels=NUM_CLASSES, base=16).to(device)
    ckpt = torch.load(path, map_location=device)

    # cGAN checkpoints store the semantic generator separately from
    # the discriminator, while baseline U-Net checkpoints use model_state.
    if is_cgan:
        model.load_state_dict(ckpt["generator_state"])
    else:
        model.load_state_dict(ckpt["model_state"])

    # Disable training-specific behaviour during visualisation.
    model.eval()
    return model


# Add a readable title directly onto one comparison image.
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


# Generate qualitative U-Net and cGAN comparison images from processed samples.
def main():
    device = get_device()
    print("Device:", device)

    # Load the complete processed dataset in fixed filename order.
    # This utility is not restricted to the held-out test partition.
    ds = FloorplanNPZDataset(
        DATA_DIR,
        max_count=MAX_COUNT,
    )

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # Load the retained U-Net baseline and the generator from the cGAN checkpoint.
    baseline = load_unet_checkpoint(
        BASELINE_CKPT,
        device,
        is_cgan=False,
    )

    cgan = load_unet_checkpoint(
        CGAN_CKPT,
        device,
        is_cgan=True,
    )

    saved = 0

    # No gradients are required because this script performs inference only.
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):

            # Stop after the requested number of qualitative examples.
            if saved >= NUM_SAVE:
                break

            x = x.to(device)

            # Recover the reference semantic mask and the filled binary
            # support condition from the loaded sample.
            gt = y[0].numpy().astype(np.uint8)
            outline = (
                x[0, 0].cpu().numpy() > 0.5
            ).astype(np.uint8)

            # Generate raw semantic logits from both retained models.
            base_logits = baseline(x)
            cgan_logits = cgan(x)

            # Convert class logits into one semantic class ID per pixel.
            base_pred = (
                torch.argmax(base_logits, dim=1)[0]
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            cgan_pred = (
                torch.argmax(cgan_logits, dim=1)[0]
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            # Convert the support, target and predictions into display images.
            outline_rgb = cv2.cvtColor(
                (outline * 255).astype(np.uint8),
                cv2.COLOR_GRAY2BGR,
            )

            gt_rgb = colorize(gt)
            base_rgb = colorize(base_pred)
            cgan_rgb = colorize(cgan_pred)

            # Label each panel before assembling the final comparison grid.
            outline_rgb = add_title(
                outline_rgb,
                "Input outline",
            )
            gt_rgb = add_title(
                gt_rgb,
                "Ground truth",
            )
            base_rgb = add_title(
                base_rgb,
                "U-Net baseline",
            )
            cgan_rgb = add_title(
                cgan_rgb,
                "cGAN",
            )

            # Arrange input and ground truth above the two model predictions.
            top = np.concatenate(
                [outline_rgb, gt_rgb],
                axis=1,
            )

            bottom = np.concatenate(
                [base_rgb, cgan_rgb],
                axis=1,
            )

            grid = np.concatenate(
                [top, bottom],
                axis=0,
            )

            # Save one combined qualitative comparison image per sample.
            out_path = os.path.join(
                OUT_DIR,
                f"comparison_{idx:04d}.png",
            )

            cv2.imwrite(
                out_path,
                grid,
            )

            saved += 1
            print("Saved:", out_path)

    print(
        f"\nSaved {saved} comparison images to:",
        OUT_DIR,
    )


if __name__ == "__main__":
    main()


import argparse
import os

import cv2
import numpy as np
import torch

from src.models.unet import UNet
from src.refinement.morphology import refine_semantic_mask_morphology


NUM_CLASSES = 9
BG = 0

# Simple colour palette for visualising semantic masks.
# The class ids follow the merged semantic ids used by the project.
PALETTE = {
    0: (255, 255, 255),  # background
    1: (230, 230, 250),
    2: (176, 224, 230),
    3: (152, 251, 152),
    4: (255, 228, 181),
    5: (255, 182, 193),
    6: (221, 160, 221),
    7: (240, 230, 140),
    8: (60, 60, 60),     # wall / structure
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a 2D semantic floor plan from a boundary mask and room-count condition."
    )
    parser.add_argument("--outline_path", type=str, required=True, help="Path to the input boundary/outline image.")
    parser.add_argument("--room_count", type=int, required=True, help="Target room-count conditioning value.")
    parser.add_argument("--ckpt_path", type=str, default="outputs/checkpoints/unet_base16_best.pt")
    parser.add_argument("--max_count", type=int, default=32)
    parser.add_argument("--out_path", type=str, default="outputs/generated/floorplan.png")
    parser.add_argument("--mask_out_path", type=str, default=None, help="Optional path for saving the raw class-id mask as .npy.")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--threshold", type=int, default=127, help="Threshold used to binarise the outline image.")
    parser.add_argument("--apply_morphology", action="store_true")
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--min_area", type=int, default=30)
    return parser.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_outline(path, size, threshold):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read outline image: {path}")

    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    outline = (image > threshold).astype(np.float32)

    # If the input is inverted, use the larger foreground assumption less aggressively.
    # The expected outline mask should contain non-zero pixels inside the building boundary.
    if outline.mean() < 0.05:
        outline = (image <= threshold).astype(np.float32)

    return outline


def build_input(outline, room_count, max_count):
    if room_count < 0:
        raise ValueError("room_count must be non-negative.")
    if room_count > max_count:
        raise ValueError(f"room_count={room_count} is larger than max_count={max_count}.")

    room_value = float(room_count) / float(max_count)
    count_channel = np.full_like(outline, room_value, dtype=np.float32)
    x = np.stack([outline.astype(np.float32), count_channel], axis=0)
    x = torch.from_numpy(x).unsqueeze(0)
    return x


def load_model(device, ckpt_path):
    model = UNet(in_channels=2, out_channels=NUM_CLASSES, base=16).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    if "generator_state" in ckpt:
        model.load_state_dict(ckpt["generator_state"])
    else:
        model.load_state_dict(ckpt["model_state"])

    model.eval()
    print("Loaded checkpoint:", ckpt_path)
    print("Loaded checkpoint epoch:", ckpt.get("epoch", "unknown"))
    print("Checkpoint val IoU:", ckpt.get("val_iou", "unknown"))
    return model


def colourise_mask(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, colour in PALETTE.items():
        rgb[mask == class_id] = colour
    return rgb


def save_outputs(mask, out_path, mask_out_path=None):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    rgb = colourise_mask(mask)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)

    if mask_out_path is not None:
        os.makedirs(os.path.dirname(mask_out_path) or ".", exist_ok=True)
        np.save(mask_out_path, mask.astype(np.uint8))


def main():
    args = parse_args()
    device = get_device()

    print("Device:", device)
    print("Outline path:", args.outline_path)
    print("Room count:", args.room_count)
    print("MAX_COUNT:", args.max_count)
    print("Checkpoint:", args.ckpt_path)
    print("Output image:", args.out_path)
    print("Apply morphology:", args.apply_morphology)

    outline = load_outline(args.outline_path, size=args.size, threshold=args.threshold)
    x = build_input(outline, room_count=args.room_count, max_count=args.max_count).to(device)

    model = load_model(device, args.ckpt_path)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

    # Keep predictions inside the supplied outline by setting outside pixels to background.
    pred[outline == 0] = BG

    if args.apply_morphology:
        pred = refine_semantic_mask_morphology(
            pred,
            num_classes=NUM_CLASSES,
            kernel_size=args.kernel_size,
            min_area=args.min_area,
        )
        pred[outline == 0] = BG

    save_outputs(pred, args.out_path, args.mask_out_path)
    print("Saved generated floor plan:", args.out_path)
    if args.mask_out_path is not None:
        print("Saved raw class-id mask:", args.mask_out_path)


if __name__ == "__main__":
    main()
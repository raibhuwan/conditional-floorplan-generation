import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.unet import UNet
from src.refinement.morphology import (
    refine_semantic_mask_morphology,
)


NUM_CLASSES = 9
BG = 0

# Colour palette for visualising the merged semantic classes.
PALETTE = {
    0: (255, 255, 255),  # background
    1: (230, 230, 250),  # kitchen
    2: (176, 224, 230),  # living
    3: (152, 251, 152),  # bedroom
    4: (255, 228, 181),  # bathroom
    5: (255, 182, 193),  # hallway
    6: (221, 160, 221),  # dining
    7: (240, 230, 140),  # utility
    8: (60, 60, 60),     # wall / structure
}


# Parse the support condition, encoded count, checkpoint and output settings.
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a semantic floor-plan mask from a binary "
            "support condition and an encoded connected-region "
            "count condition."
        )
    )

    parser.add_argument(
        "--outline_path",
        type=str,
        required=True,
        help=(
            "Path to the binary support-condition image. "
            "The argument name is retained for compatibility."
        ),
    )

    parser.add_argument(
        "--room_count",
        type=int,
        required=True,
        help=(
            "Encoded connected-region count used as the second "
            "conditioning channel. The argument name is retained "
            "for compatibility."
        ),
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=(
            "outputs/checkpoints/"
            "unet_base16_best.pt"
        ),
    )

    parser.add_argument(
        "--max_count",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--out_path",
        type=str,
        default=(
            "outputs/generated/"
            "floorplan.png"
        ),
        help=(
            "Path for the colourised PNG output. "
            "A PDF with the same filename stem is also saved."
        ),
    )

    parser.add_argument(
        "--mask_out_path",
        type=str,
        default=None,
        help=(
            "Optional path for saving the raw class-ID mask "
            "as a NumPy .npy file."
        ),
    )

    parser.add_argument(
        "--size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help=(
            "Threshold used to convert the support image "
            "into a binary mask."
        ),
    )

    parser.add_argument(
        "--apply_morphology",
        action="store_true",
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

    return parser.parse_args()


# Select the best available accelerator, with CPU as the final fallback.
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# Load an image and convert it into the binary support condition expected by the model.
def load_outline(
    path,
    size,
    threshold,
):
    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE,
    )

    # Stop if OpenCV cannot read the supplied support-condition image.
    if image is None:
        raise FileNotFoundError(
            f"Could not read support image: {path}"
        )

    # Resize using nearest-neighbour interpolation so binary boundaries
    # are not blurred by interpolation.
    image = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_NEAREST,
    )

    # Convert the grayscale image into a binary foreground/background support mask.
    support = (
        image > threshold
    ).astype(np.float32)

    # Handle an unexpectedly inverted image where only a very
    # small proportion of pixels are treated as foreground.
    if support.mean() < 0.05:
        support = (
            image <= threshold
        ).astype(np.float32)

    return support


# Construct the two-channel model input from support and encoded count conditions.
def build_input(
    support,
    encoded_count,
    max_count,
):
    # Reject encoded counts outside the range supported by the normalisation scheme.
    if encoded_count < 0:
        raise ValueError(
            "The encoded connected-region count must "
            "be non-negative."
        )

    if encoded_count > max_count:
        raise ValueError(
            f"encoded_count={encoded_count} is greater "
            f"than max_count={max_count}."
        )

    # Normalise the encoded count to the same [0, 1] scale used during training.
    count_value = (
        float(encoded_count)
        / float(max_count)
    )

    # Repeat the scalar count over the complete spatial field so it can
    # be supplied as the second convolutional input channel.
    count_channel = np.full_like(
        support,
        count_value,
        dtype=np.float32,
    )

    # Stack the binary support mask and normalised count channel.
    model_input = np.stack(
        [
            support.astype(np.float32),
            count_channel,
        ],
        axis=0,
    )

    # Add the batch dimension expected by the U-Net.
    return (
        torch.from_numpy(model_input)
        .unsqueeze(0)
    )


# Load either a retained U-Net checkpoint or the generator from a cGAN checkpoint.
def load_model(
    device,
    checkpoint_path,
):
    # Both supported checkpoints use the same U-Net architecture for prediction.
    model = UNet(
        in_channels=2,
        out_channels=NUM_CLASSES,
        base=16,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # Baseline U-Net checkpoints store their parameters under model_state.
    if "model_state" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state"]
        )
        checkpoint_type = "U-Net"

    # cGAN checkpoints store the semantic generator under generator_state.
    elif "generator_state" in checkpoint:
        model.load_state_dict(
            checkpoint["generator_state"]
        )
        checkpoint_type = "cGAN generator"

    else:
        raise KeyError(
            "The checkpoint does not contain either "
            "'model_state' or 'generator_state'."
        )

    # Disable training-specific behaviour during generation.
    model.eval()

    print(
        "Loaded checkpoint:",
        checkpoint_path,
    )
    print(
        "Checkpoint type:",
        checkpoint_type,
    )
    print(
        "Loaded checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )
    print(
        "Checkpoint validation IoU:",
        checkpoint.get(
            "val_iou",
            "unknown",
        ),
    )

    return model


# Convert a semantic class-ID mask into an RGB visualisation.
def colourise_mask(mask):
    height, width = mask.shape

    # Start with an empty RGB canvas matching the semantic mask dimensions.
    rgb = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    # Assign the predefined display colour to each semantic class.
    for class_id, colour in PALETTE.items():
        rgb[
            mask == class_id
        ] = colour

    return rgb


# Save the generated semantic layout as PNG, PDF and optionally a raw NumPy mask.
def save_outputs(
    mask,
    out_path,
    mask_out_path=None,
):
    output_directory = (
        os.path.dirname(out_path)
        or "."
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    # Convert class IDs into their RGB visual representation.
    rgb = colourise_mask(mask)

    # OpenCV writes BGR images, so convert the RGB visualisation before saving.
    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    saved = cv2.imwrite(
        out_path,
        bgr,
    )

    if not saved:
        raise IOError(
            f"Could not save PNG output: {out_path}"
        )

    # Save the same colourised semantic layout as a PDF.
    pdf_path = (
        os.path.splitext(out_path)[0]
        + ".pdf"
    )

    figure, axis = plt.subplots(
        figsize=(5, 5)
    )

    axis.imshow(
        rgb,
        interpolation="nearest",
    )

    axis.axis("off")

    figure.tight_layout(
        pad=0
    )

    figure.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close(figure)

    # Optionally preserve the raw class-ID mask for later quantitative analysis.
    if mask_out_path is not None:
        mask_directory = (
            os.path.dirname(mask_out_path)
            or "."
        )

        os.makedirs(
            mask_directory,
            exist_ok=True,
        )

        np.save(
            mask_out_path,
            mask.astype(np.uint8),
        )

    return pdf_path


# Run the complete conditional floor-plan generation workflow.
def main():
    args = parse_args()
    device = get_device()

    print("Device:", device)
    print(
        "Binary support path:",
        args.outline_path,
    )
    print(
        "Encoded connected-region count:",
        args.room_count,
    )
    print(
        "Maximum encoded count:",
        args.max_count,
    )
    print(
        "Checkpoint:",
        args.ckpt_path,
    )
    print(
        "Output image:",
        args.out_path,
    )
    print(
        "Apply morphology:",
        args.apply_morphology,
    )

    # Prepare the filled binary support condition from the supplied image.
    support = load_outline(
        args.outline_path,
        size=args.size,
        threshold=args.threshold,
    )

    # Combine the support mask and normalised encoded count into the
    # same two-channel representation used during model training.
    model_input = build_input(
        support,
        encoded_count=args.room_count,
        max_count=args.max_count,
    ).to(device)

    # Load the selected trained semantic prediction model.
    model = load_model(
        device,
        args.ckpt_path,
    )

    # Generate class logits without gradient calculation and convert
    # them into one semantic class ID per pixel.
    with torch.no_grad():
        logits = model(model_input)

        prediction = torch.argmax(
            logits,
            dim=1,
        )[0]

        prediction = (
            prediction.detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

    # Keep the generated semantic classes within the supplied
    # binary support condition.
    prediction[
        support == 0
    ] = BG

    # Optionally apply the fixed morphology post-processing procedure.
    if args.apply_morphology:
        prediction = (
            refine_semantic_mask_morphology(
                prediction,
                num_classes=NUM_CLASSES,
                kernel_size=args.kernel_size,
                min_area=args.min_area,
            )
            .astype(np.uint8)
        )

        # Reapply support clipping because morphological operations
        # may modify pixels close to the support boundary.
        prediction[
            support == 0
        ] = BG

    # Save the final semantic prediction in the requested output formats.
    pdf_path = save_outputs(
        prediction,
        args.out_path,
        args.mask_out_path,
    )

    print(
        "Saved generated semantic floor plan:",
        args.out_path,
    )
    print(
        "Saved PDF:",
        pdf_path,
    )

    if args.mask_out_path is not None:
        print(
            "Saved raw class-ID mask:",
            args.mask_out_path,
        )


if __name__ == "__main__":
    main()
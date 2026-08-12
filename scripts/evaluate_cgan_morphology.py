import argparse
import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import FloorplanNPZDataset
from src.data.splits import load_split
from src.models.unet import UNet
from src.refinement.morphology import refine_semantic_mask_morphology


DATA_DIR = "data/processed_npz_clean_full"
SPLIT_PATH = "outputs/splits/split_seed42_full.json"
CKPT_PATH = "outputs/checkpoints/cgan_unet_patchgan_best.pt"

MAX_COUNT = 32
NUM_CLASSES = 9

BG = 0
WALL = 8

INSTANCE_MIN_AREA = 30
OUT_CSV = "outputs/metrics_cgan_morphology_room_count_fixed.csv"

os.makedirs("outputs", exist_ok=True)


# Parse command-line settings for cGAN evaluation with morphology refinement.
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Pix2Pix-style cGAN with morphology refinement on "
            "the held-out test set using a consistent combined-component "
            "room-count definition."
        )
    )
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--split_path", type=str, default=SPLIT_PATH)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--max_count", type=int, default=MAX_COUNT)
    parser.add_argument("--out_csv", type=str, default=OUT_CSV)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--min_area", type=int, default=30)
    return parser.parse_args()


# Select Apple MPS acceleration when available, otherwise use the CPU.
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mean_iou(pred, gt, num_classes=NUM_CLASSES, ignore=(BG,)):
    """
    Calculate sample-level mean IoU across semantic classes while excluding
    the class IDs listed in ``ignore``.
    """
    ious = []

    # Calculate IoU independently for each eligible semantic class.
    for class_id in range(num_classes):
        if class_id in ignore:
            continue

        pred_class = pred == class_id
        gt_class = gt == class_id

        intersection = np.logical_and(
            pred_class,
            gt_class,
        ).sum()
        union = np.logical_or(
            pred_class,
            gt_class,
        ).sum()

        # Classes absent from both masks do not contribute to the sample mean.
        if union == 0:
            continue

        ious.append(intersection / union)

    return float(np.mean(ious)) if ious else 0.0


def extract_instances(
    mask,
    ignore_ids=(BG, WALL),
    min_area=INSTANCE_MIN_AREA,
):
    """
    Extract class-specific connected components.

    These components are used for adjacency and compactness only. They are
    not used for connected-region count error because the encoded target
    count was constructed from one combined non-background, non-wall mask.
    """
    instances = []

    # Process each retained semantic class independently.
    for class_id in range(NUM_CLASSES):
        if class_id in ignore_ids:
            continue

        class_mask = (mask == class_id).astype(np.uint8)

        # Skip classes whose total foreground area is below the threshold.
        if int(class_mask.sum()) < min_area:
            continue

        # Identify eight-connected components within the current semantic class.
        component_count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                class_mask,
                connectivity=8,
            )
        )

        # Component 0 represents the background.
        for component_id in range(1, component_count):
            area = int(
                stats[component_id, cv2.CC_STAT_AREA]
            )

            # Exclude components smaller than the minimum instance area.
            if area < min_area:
                continue

            x = int(
                stats[component_id, cv2.CC_STAT_LEFT]
            )
            y = int(
                stats[component_id, cv2.CC_STAT_TOP]
            )
            width = int(
                stats[component_id, cv2.CC_STAT_WIDTH]
            )
            height = int(
                stats[component_id, cv2.CC_STAT_HEIGHT]
            )

            instance_mask = (
                labels == component_id
            ).astype(np.uint8)

            # Store the class identity and component geometry required by
            # the adjacency and compactness calculations.
            instances.append(
                {
                    "class_id": class_id,
                    "area": area,
                    "bbox": (x, y, width, height),
                    "mask": instance_mask,
                }
            )

    return instances


def count_rooms_from_semantic_mask(
    mask,
    background_id=BG,
    wall_id=WALL,
):
    """
    Count connected regions from one combined binary room-region mask.

    All non-background and non-wall semantic pixels are merged before
    connected components are calculated. Semantic class boundaries do not
    create additional regions. This matches the count construction used
    for the conditioning channel during preprocessing.
    """
    # Merge all room-type classes before connected-component analysis.
    room_region_mask = np.logical_and(
        mask != background_id,
        mask != wall_id,
    ).astype(np.uint8)

    if int(room_region_mask.sum()) == 0:
        return 0

    # Use the same eight-connectivity rule applied during preprocessing.
    component_count, _ = cv2.connectedComponents(
        room_region_mask,
        connectivity=8,
    )

    # OpenCV includes the background as component 0.
    return int(component_count - 1)


def compactness_of_instance(instance_mask):
    """
    Compute compactness as 4*pi*A/P^2 for one binary instance.
    """
    area = float(instance_mask.sum())

    if area <= 0:
        return 0.0

    # Extract external contours for perimeter estimation.
    contours, _ = cv2.findContours(
        instance_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return 0.0

    # Sum contour lengths when more than one external contour is present.
    perimeter = sum(
        cv2.arcLength(contour, True)
        for contour in contours
    )

    # Avoid division by zero for degenerate components.
    if perimeter <= 1e-6:
        return 0.0

    return float(
        (4.0 * np.pi * area)
        / (perimeter * perimeter)
    )


def adjacency_edges(instances):
    """
    Build class-level adjacency edges from class-specific instances.
    """
    edges = set()
    kernel = np.ones((3, 3), np.uint8)

    # Dilate each component once so nearby regions can be tested for contact.
    dilated_masks = [
        cv2.dilate(
            instance["mask"].astype(np.uint8),
            kernel,
            iterations=1,
        )
        for instance in instances
    ]

    # Compare every unique pair of extracted components.
    for first_index in range(len(instances)):
        first_class = instances[first_index]["class_id"]

        for second_index in range(
            first_index + 1,
            len(instances),
        ):
            second_class = instances[
                second_index
            ]["class_id"]

            # Same-class component pairs do not create class-level adjacency edges.
            if first_class == second_class:
                continue

            # Overlap after one dilation step is treated as adjacency.
            touching = np.logical_and(
                dilated_masks[first_index] > 0,
                dilated_masks[second_index] > 0,
            ).any()

            if touching:
                # Store each unordered class pair only once.
                edges.add(
                    tuple(
                        sorted(
                            (first_class, second_class)
                        )
                    )
                )

    return edges


def f1_edges(predicted_edges, ground_truth_edges):
    """
    Calculate F1 score between two class-level adjacency-edge sets.
    """
    # Two empty edge sets represent complete agreement for the sample.
    if not predicted_edges and not ground_truth_edges:
        return 1.0

    # If only one set is empty, no adjacency relationship is matched.
    if not predicted_edges or not ground_truth_edges:
        return 0.0

    # Derive the F1 components from the predicted and reference edge sets.
    true_positives = len(
        predicted_edges.intersection(
            ground_truth_edges
        )
    )
    false_positives = len(
        predicted_edges - ground_truth_edges
    )
    false_negatives = len(
        ground_truth_edges - predicted_edges
    )

    precision_denominator = (
        true_positives + false_positives
    )
    recall_denominator = (
        true_positives + false_negatives
    )

    precision = (
        true_positives / precision_denominator
        if precision_denominator
        else 0.0
    )
    recall = (
        true_positives / recall_denominator
        if recall_denominator
        else 0.0
    )

    if precision + recall == 0:
        return 0.0

    return float(
        2 * precision * recall
        / (precision + recall)
    )


def boundary_violation_rate(
    pred_mask,
    support_mask,
):
    """
    Measure the proportion of predicted non-background pixels outside the
    binary floor-plan support mask.
    """
    # Include every predicted foreground class, including wall/structure.
    predicted_non_background = pred_mask != BG
    total_predicted_pixels = int(
        predicted_non_background.sum()
    )

    if total_predicted_pixels == 0:
        return 0.0

    # Count predicted foreground pixels positioned outside the support mask.
    outside_pixels = np.logical_and(
        predicted_non_background,
        support_mask == 0,
    ).sum()

    return float(
        outside_pixels / total_predicted_pixels
    )


def get_expected_room_count_from_input(
    inputs,
    max_count,
):
    """
    Recover the encoded connected-region count from the normalised
    conditional count channel.
    """
    # The encoded scalar is repeated across the full second input channel.
    count_channel = (
        inputs[0, 1]
        .detach()
        .cpu()
        .numpy()
    )
    normalised_count = float(
        count_channel.max()
    )

    # Reverse the preprocessing normalisation to recover the integer count.
    return int(
        round(normalised_count * max_count)
    )


def room_count_error(
    expected_count,
    predicted_count,
):
    """
    Calculate absolute error between the encoded target count and the
    connected-region count reconstructed from the prediction.
    """
    return abs(
        expected_count - predicted_count
    )


def load_generator(device, checkpoint_path):
    """
    Load the U-Net generator stored in a cGAN checkpoint.
    """
    generator = UNet(
        in_channels=2,
        out_channels=NUM_CLASSES,
        base=16,
    ).to(device)

    # Load the checkpoint selected using validation performance.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # Prevent an incompatible checkpoint from being evaluated accidentally.
    if "generator_state" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain 'generator_state'. "
            "Use a cGAN checkpoint produced by train_cgan.py."
        )

    # Restore the retained generator parameters and enable evaluation behaviour.
    generator.load_state_dict(
        checkpoint["generator_state"]
    )
    generator.eval()

    return generator, checkpoint


# Evaluate the retained cGAN generator after morphology refinement.
def main():
    args = parse_args()

    device = get_device()
    print("Device:", device)

    # Load the processed samples and the fixed train/validation/test split.
    dataset = FloorplanNPZDataset(
        args.data_dir,
        max_count=args.max_count,
    )
    split = load_split(args.split_path)

    # Restrict evaluation to the held-out test partition.
    test_indices = split["test"]
    test_dataset = Subset(
        dataset,
        test_indices,
    )
    loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print(f"Data folder: {args.data_dir}")
    print(f"Split file: {args.split_path}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"MAX_COUNT: {args.max_count}")
    print(f"Output CSV: {args.out_csv}")
    print(f"Morphology kernel size: {args.kernel_size}")
    print(f"Morphology min area: {args.min_area}")
    print(
        "Evaluating cGAN + morphology on held-out test samples: "
        f"{len(test_dataset)}"
    )
    print(
        "Room-count definition: connected components in one combined "
        "non-background, non-wall mask."
    )

    # Only the generator is needed for semantic prediction at evaluation time.
    generator, checkpoint = load_generator(
        device,
        args.ckpt_path,
    )

    print(
        "Loaded checkpoint: "
        f"epoch={checkpoint.get('epoch', 'unknown')}, "
        f"stored_val_iou={checkpoint.get('val_iou', 'unknown')}"
    )

    # Store one complete metric record for each held-out floor sample.
    rows = []

    for test_position, (inputs, targets) in enumerate(
        loader
    ):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Generate the raw semantic prediction without gradient calculation.
        with torch.no_grad():
            logits = generator(inputs)
            predictions = torch.argmax(
                logits,
                dim=1,
            )

        # Convert the raw prediction and reference target to NumPy masks.
        pred_mask = (
            predictions[0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        gt_mask = (
            targets[0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        # Recover the filled binary support condition from the first input channel.
        support_mask = (
            inputs[0, 0]
            .detach()
            .cpu()
            .numpy()
            > 0.5
        ).astype(np.uint8)

        # Recover the encoded connected-region count from the second input channel.
        expected_room_count = (
            get_expected_room_count_from_input(
                inputs,
                max_count=args.max_count,
            )
        )

        # Apply the fixed morphology procedure to the raw cGAN prediction.
        pred_refined = refine_semantic_mask_morphology(
            pred_mask,
            num_classes=NUM_CLASSES,
            kernel_size=args.kernel_size,
            min_area=args.min_area,
        ).astype(np.uint8)

        # Reconstruct connected-region counts using the same combined-mask rule.
        gt_room_count = (
            count_rooms_from_semantic_mask(
                gt_mask
            )
        )
        predicted_room_count = (
            count_rooms_from_semantic_mask(
                pred_refined
            )
        )

        # Check agreement between the encoded condition and reconstructed target count.
        target_count_matches_gt = int(
            expected_room_count
            == gt_room_count
        )

        count_error = room_count_error(
            expected_room_count,
            predicted_room_count,
        )

        # Calculate semantic overlap using the refined output.
        miou = mean_iou(
            pred_refined,
            gt_mask,
            ignore=(BG,),
        )

        # Extract class-specific components for adjacency and compactness only.
        gt_instances = extract_instances(
            gt_mask
        )
        predicted_instances = extract_instances(
            pred_refined
        )

        # Evaluate the refined output without clipping it to the support boundary.
        boundary_violation = (
            boundary_violation_rate(
                pred_refined,
                support_mask,
            )
        )

        # Compare class-level adjacency relationships in prediction and target.
        gt_edges = adjacency_edges(
            gt_instances
        )
        predicted_edges = adjacency_edges(
            predicted_instances
        )
        adjacency_f1 = f1_edges(
            predicted_edges,
            gt_edges,
        )

        # Calculate compactness for retained ground-truth components.
        gt_compactness_values = [
            compactness_of_instance(
                instance["mask"]
            )
            for instance in gt_instances
        ]

        # Calculate compactness for retained refined prediction components.
        predicted_compactness_values = [
            compactness_of_instance(
                instance["mask"]
            )
            for instance in predicted_instances
        ]

        # Average component compactness within the current floor sample.
        gt_compactness = (
            float(
                np.mean(
                    gt_compactness_values
                )
            )
            if gt_compactness_values
            else 0.0
        )
        predicted_compactness = (
            float(
                np.mean(
                    predicted_compactness_values
                )
            )
            if predicted_compactness_values
            else 0.0
        )

        # Preserve the complete sample-level evaluation record.
        rows.append(
            {
                "idx": test_position,
                "dataset_index": int(
                    test_indices[test_position]
                ),
                "miou_no_bg": miou,
                "adj_f1": adjacency_f1,
                "compact_gt": gt_compactness,
                "compact_pred": predicted_compactness,
                "num_inst_gt": len(
                    gt_instances
                ),
                "num_inst_pred": len(
                    predicted_instances
                ),
                "boundary_violation_rate": (
                    boundary_violation
                ),
                "expected_room_count": (
                    expected_room_count
                ),
                "gt_room_count_combined": (
                    gt_room_count
                ),
                "target_count_matches_gt": (
                    target_count_matches_gt
                ),
                "predicted_room_count": (
                    predicted_room_count
                ),
                "room_count_error": (
                    count_error
                ),
            }
        )

        # Print periodic progress during held-out evaluation.
        if (test_position + 1) % 50 == 0:
            print(
                f"[{test_position + 1}/"
                f"{len(test_dataset)}] "
                f"mIoU={miou:.3f} "
                f"adjF1={adjacency_f1:.3f} "
                f"comp_pred={predicted_compactness:.3f} "
                f"BVR={boundary_violation:.3f} "
                f"expected={expected_room_count} "
                f"predicted={predicted_room_count} "
                f"RCerr={count_error}"
            )

    # Avoid creating an empty results file if no samples were evaluated.
    if not rows:
        raise RuntimeError(
            "No evaluation rows were produced."
        )

    output_directory = (
        os.path.dirname(args.out_csv)
        or "."
    )
    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    # Save all sample-level measurements before calculating overall averages.
    with open(
        args.out_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate sample-level values across the complete held-out test partition.
    mean_miou = float(
        np.mean(
            [
                row["miou_no_bg"]
                for row in rows
            ]
        )
    )
    mean_adjacency_f1 = float(
        np.mean(
            [
                row["adj_f1"]
                for row in rows
            ]
        )
    )
    mean_compactness = float(
        np.mean(
            [
                row["compact_pred"]
                for row in rows
            ]
        )
    )
    mean_boundary_violation = float(
        np.mean(
            [
                row[
                    "boundary_violation_rate"
                ]
                for row in rows
            ]
        )
    )
    room_count_mae = float(
        np.mean(
            [
                row["room_count_error"]
                for row in rows
            ]
        )
    )
    mean_expected_room_count = float(
        np.mean(
            [
                row["expected_room_count"]
                for row in rows
            ]
        )
    )
    mean_predicted_room_count = float(
        np.mean(
            [
                row["predicted_room_count"]
                for row in rows
            ]
        )
    )

    # Check preprocessing/evaluation consistency for the encoded count.
    target_count_mismatches = sum(
        1
        for row in rows
        if not row["target_count_matches_gt"]
    )

    print("\nSaved:", args.out_csv)
    print(
        f"Summary over {len(rows)} samples:"
    )
    print(
        f" mean mIoU (no bg): "
        f"{mean_miou:.3f}"
    )
    print(
        f" mean adjacency F1: "
        f"{mean_adjacency_f1:.3f}"
    )
    print(
        " mean compactness (pred): "
        f"{mean_compactness:.3f}"
    )
    print(
        " mean boundary violation rate: "
        f"{mean_boundary_violation:.3f}"
    )
    print(
        " mean expected room count: "
        f"{mean_expected_room_count:.3f}"
    )
    print(
        " mean predicted room count: "
        f"{mean_predicted_room_count:.3f}"
    )
    print(
        f" mean room-count MAE: "
        f"{room_count_mae:.3f}"
    )
    print(
        " encoded-count versus ground-truth combined-count "
        f"mismatches: {target_count_mismatches}/"
        f"{len(rows)}"
    )

    if target_count_mismatches:
        print(
            "WARNING: Some encoded room counts do not match the "
            "combined-component count reconstructed from the ground-truth "
            "mask. Review the preprocessing rule before treating the "
            "room-count metric as final."
        )
    else:
        print(
            "Room-count consistency check passed: every encoded target "
            "count matched the combined-component count reconstructed "
            "from the ground-truth mask."
        )


if __name__ == "__main__":
    main()
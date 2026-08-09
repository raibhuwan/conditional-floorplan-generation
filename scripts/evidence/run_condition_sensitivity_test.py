#!/usr/bin/env python3
"""Condition-sensitivity, determinism and runtime evidence for the ARP project.

Run this script from the repository root, where ``src/`` is available.
It does not modify training data or checkpoints. Results are written to a new
output folder and compressed into a small ZIP file for review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.models.unet import UNet
from src.refinement.morphology import refine_semantic_mask_morphology


NUM_CLASSES = 9
BG = 0
WALL = 8
PALETTE_RGB = {
    0: (255, 255, 255),
    1: (230, 230, 250),
    2: (176, 224, 230),
    3: (152, 251, 152),
    4: (255, 228, 181),
    5: (255, 182, 193),
    6: (221, 160, 221),
    7: (240, 230, 140),
    8: (60, 60, 60),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether changing the encoded connected-region count affects "
            "U-Net output, verify determinism, and measure local runtime."
        )
    )
    parser.add_argument(
        "--outline_path",
        default="inputs/boundary.png",
        help="Path to the binary floor-plan support image.",
    )
    parser.add_argument(
        "--ckpt_path",
        default="outputs/checkpoints/unet_base16_best.pt",
        help="Path to the selected U-Net checkpoint.",
    )
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6, 7],
        help="Encoded count values to test.",
    )
    parser.add_argument(
        "--repeat_count",
        type=int,
        default=4,
        help="Count value used for the repeated determinism test.",
    )
    parser.add_argument(
        "--repeat_runs",
        type=int,
        default=3,
        help="Number of repeated outputs for the determinism test.",
    )
    parser.add_argument("--max_count", type=int, default=32)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--min_area", type=int, default=30)
    parser.add_argument(
        "--timing_runs",
        type=int,
        default=30,
        help="Number of timed inference and morphology repetitions.",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=5,
        help="Untimed warm-up repetitions before timing.",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/condition_sensitivity_test",
        help="Folder in which evidence files are written.",
    )
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def load_support(path: Path, size: int, threshold: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read support image: {path}")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    support = (image > threshold).astype(np.float32)
    if support.mean() < 0.05:
        support = (image <= threshold).astype(np.float32)
    return support


def build_input(support: np.ndarray, count: int, max_count: int) -> torch.Tensor:
    if count < 0 or count > max_count:
        raise ValueError(f"Count {count} must be between 0 and {max_count}.")
    count_value = float(count) / float(max_count)
    count_channel = np.full_like(support, count_value, dtype=np.float32)
    array = np.stack([support.astype(np.float32), count_channel], axis=0)
    return torch.from_numpy(array).unsqueeze(0)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[UNet, dict[str, Any]]:
    model = UNet(in_channels=2, out_channels=NUM_CLASSES, base=16).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if "generator_state" in checkpoint:
        model.load_state_dict(checkpoint["generator_state"])
    elif "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        raise KeyError("Checkpoint does not contain generator_state or model_state.")
    model.eval()
    metadata = {
        "epoch": checkpoint.get("epoch", "unknown"),
        "stored_validation_iou": checkpoint.get("val_iou", "unknown"),
    }
    return model, metadata


def predict(
    model: UNet,
    tensor: torch.Tensor,
    support: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        logits = model(tensor.to(device))
        prediction = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    prediction[support == 0] = BG
    return prediction


def encoded_connected_region_count(mask: np.ndarray) -> int:
    """Match the project count definition: combined non-BG, non-wall components."""
    binary = ((mask != BG) & (mask != WALL)).astype(np.uint8)
    component_total, _ = cv2.connectedComponents(binary, connectivity=8)
    return int(component_total - 1)


def mask_hash(mask: np.ndarray) -> str:
    return hashlib.sha256(mask.tobytes()).hexdigest()


def colourise(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, colour in PALETTE_RGB.items():
        rgb[mask == class_id] = colour
    return rgb


def save_mask_outputs(mask: np.ndarray, png_path: Path, npy_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = colourise(mask)
    cv2.imwrite(str(png_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(str(npy_path), mask.astype(np.uint8))


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def timing_stats(milliseconds: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(statistics.mean(milliseconds)),
        "median_ms": float(statistics.median(milliseconds)),
        "min_ms": float(min(milliseconds)),
        "max_ms": float(max(milliseconds)),
        "p95_ms": percentile(milliseconds, 95),
    }


def make_contact_sheet(
    support: np.ndarray,
    raw_by_count: dict[int, np.ndarray],
    morph_by_count: dict[int, np.ndarray],
    output_path: Path,
) -> None:
    counts = list(raw_by_count.keys())
    tile_h, tile_w = support.shape
    label_h = 34
    rows = 3
    sheet = np.full((rows * (tile_h + label_h), len(counts) * tile_w, 3), 255, dtype=np.uint8)

    support_rgb = np.repeat((support[..., None] * 255).astype(np.uint8), 3, axis=2)
    for col, count in enumerate(counts):
        x0 = col * tile_w
        tiles = [support_rgb, colourise(raw_by_count[count]), colourise(morph_by_count[count])]
        labels = [
            f"Support | input={count}",
            f"Raw | pred={encoded_connected_region_count(raw_by_count[count])}",
            f"Morph | pred={encoded_connected_region_count(morph_by_count[count])}",
        ]
        for row, (tile, label) in enumerate(zip(tiles, labels)):
            y0 = row * (tile_h + label_h)
            sheet[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
            cv2.putText(
                sheet,
                label,
                (x0 + 8, y0 + tile_h + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
    if args.repeat_runs < 2:
        raise ValueError("repeat_runs must be at least 2.")
    if args.timing_runs < 1 or args.warmup_runs < 0:
        raise ValueError("timing_runs must be positive and warmup_runs non-negative.")

    support_path = Path(args.outline_path).resolve()
    checkpoint_path = Path(args.ckpt_path).resolve()
    output_dir = Path(args.out_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    support = load_support(support_path, args.size, args.threshold)

    load_start = time.perf_counter()
    model, checkpoint_metadata = load_model(checkpoint_path, device)
    synchronize(device)
    model_load_ms = (time.perf_counter() - load_start) * 1000.0

    raw_by_count: dict[int, np.ndarray] = {}
    morph_by_count: dict[int, np.ndarray] = {}
    result_rows: list[dict[str, Any]] = []

    for count in args.counts:
        tensor = build_input(support, count, args.max_count)

        synchronize(device)
        start = time.perf_counter()
        raw = predict(model, tensor, support, device)
        synchronize(device)
        one_inference_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        morph = refine_semantic_mask_morphology(
            raw,
            num_classes=NUM_CLASSES,
            kernel_size=args.kernel_size,
            min_area=args.min_area,
        )
        morph[support == 0] = BG
        one_morphology_ms = (time.perf_counter() - start) * 1000.0

        raw_by_count[count] = raw
        morph_by_count[count] = morph
        save_mask_outputs(
            raw,
            output_dir / "generated" / f"count_{count:02d}_raw.png",
            output_dir / "generated" / f"count_{count:02d}_raw.npy",
        )
        save_mask_outputs(
            morph,
            output_dir / "generated" / f"count_{count:02d}_morphology.png",
            output_dir / "generated" / f"count_{count:02d}_morphology.npy",
        )

        result_rows.append(
            {
                "requested_encoded_count": count,
                "normalised_condition": count / args.max_count,
                "predicted_connected_regions_raw": encoded_connected_region_count(raw),
                "predicted_connected_regions_morphology": encoded_connected_region_count(morph),
                "non_background_pixels_raw": int((raw != BG).sum()),
                "non_background_pixels_morphology": int((morph != BG).sum()),
                "raw_sha256": mask_hash(raw),
                "morphology_sha256": mask_hash(morph),
                "single_inference_ms": one_inference_ms,
                "single_morphology_ms": one_morphology_ms,
            }
        )

    # Determinism: repeat the same input and compare exact class-id masks.
    repeated: list[np.ndarray] = []
    repeated_hashes: list[str] = []
    repeat_tensor = build_input(support, args.repeat_count, args.max_count)
    for run_index in range(args.repeat_runs):
        repeated_mask = predict(model, repeat_tensor, support, device)
        repeated.append(repeated_mask)
        repeated_hashes.append(mask_hash(repeated_mask))
        save_mask_outputs(
            repeated_mask,
            output_dir / "determinism" / f"count_{args.repeat_count:02d}_repeat_{run_index + 1}.png",
            output_dir / "determinism" / f"count_{args.repeat_count:02d}_repeat_{run_index + 1}.npy",
        )
    deterministic_exact = all(np.array_equal(repeated[0], item) for item in repeated[1:])
    repeat_changed_pixels = [int(np.count_nonzero(repeated[0] != item)) for item in repeated[1:]]

    # Sensitivity: compare adjacent requested counts using exact changed-pixel rates.
    adjacent_sensitivity: list[dict[str, Any]] = []
    for left, right in zip(args.counts[:-1], args.counts[1:]):
        raw_changed = int(np.count_nonzero(raw_by_count[left] != raw_by_count[right]))
        morph_changed = int(np.count_nonzero(morph_by_count[left] != morph_by_count[right]))
        total_pixels = int(raw_by_count[left].size)
        adjacent_sensitivity.append(
            {
                "from_count": left,
                "to_count": right,
                "raw_changed_pixels": raw_changed,
                "raw_changed_fraction": raw_changed / total_pixels,
                "morphology_changed_pixels": morph_changed,
                "morphology_changed_fraction": morph_changed / total_pixels,
            }
        )

    all_raw_hashes = {mask_hash(mask) for mask in raw_by_count.values()}
    all_morph_hashes = {mask_hash(mask) for mask in morph_by_count.values()}

    # Runtime benchmark after warm-up. Model load time is reported separately.
    timing_tensor = build_input(support, args.repeat_count, args.max_count).to(device)
    for _ in range(args.warmup_runs):
        _ = predict(model, timing_tensor, support, device)
    synchronize(device)

    inference_times_ms: list[float] = []
    last_raw: np.ndarray | None = None
    for _ in range(args.timing_runs):
        synchronize(device)
        start = time.perf_counter()
        last_raw = predict(model, timing_tensor, support, device)
        synchronize(device)
        inference_times_ms.append((time.perf_counter() - start) * 1000.0)

    assert last_raw is not None
    morphology_times_ms: list[float] = []
    for _ in range(args.timing_runs):
        start = time.perf_counter()
        refined = refine_semantic_mask_morphology(
            last_raw,
            num_classes=NUM_CLASSES,
            kernel_size=args.kernel_size,
            min_area=args.min_area,
        )
        refined[support == 0] = BG
        morphology_times_ms.append((time.perf_counter() - start) * 1000.0)

    # Write structured evidence.
    with (output_dir / "condition_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    with (output_dir / "adjacent_count_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(adjacent_sensitivity[0].keys()))
        writer.writeheader()
        writer.writerows(adjacent_sensitivity)

    summary = {
        "test_purpose": (
            "Evidence only: assess sensitivity to the encoded connected-region count, "
            "determinism, and runtime. This test does not establish exact architectural room-count control."
        ),
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "device": str(device),
            "mps_available": bool(torch.backends.mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "inputs": {
            "support_path": str(support_path),
            "support_foreground_fraction": float(support.mean()),
            "checkpoint_path": str(checkpoint_path),
            "counts_tested": list(args.counts),
            "max_count": args.max_count,
            "image_size": args.size,
            "checkpoint_metadata": checkpoint_metadata,
        },
        "condition_sensitivity": {
            "unique_raw_outputs": len(all_raw_hashes),
            "unique_morphology_outputs": len(all_morph_hashes),
            "all_tested_raw_outputs_identical": len(all_raw_hashes) == 1,
            "all_tested_morphology_outputs_identical": len(all_morph_hashes) == 1,
            "adjacent_comparisons": adjacent_sensitivity,
        },
        "determinism": {
            "repeated_count": args.repeat_count,
            "repeat_runs": args.repeat_runs,
            "exactly_identical": deterministic_exact,
            "hashes": repeated_hashes,
            "changed_pixels_against_first": repeat_changed_pixels,
        },
        "runtime": {
            "checkpoint_load_ms": model_load_ms,
            "warmup_runs": args.warmup_runs,
            "timing_runs": args.timing_runs,
            "inference_only": timing_stats(inference_times_ms),
            "morphology_only": timing_stats(morphology_times_ms),
            "note": "Inference excludes checkpoint loading and image saving. Morphology is timed separately.",
        },
        "results": result_rows,
    }
    with (output_dir / "condition_sensitivity_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    make_contact_sheet(
        support,
        raw_by_count,
        morph_by_count,
        output_dir / "condition_sensitivity_contact_sheet.png",
    )

    readme_text = f"""Condition-sensitivity evidence\n\nCounts tested: {args.counts}\nRepeated count: {args.repeat_count} ({args.repeat_runs} runs)\nDevice: {device}\n\nKey interpretation rule:\nThis test checks whether the scalar encoded connected-region condition changes the output.\nIt does not prove that the model can produce an exact requested number of architectural rooms.\n\nSee condition_sensitivity_summary.json and the contact sheet for the results.\n"""
    (output_dir / "README.txt").write_text(readme_text, encoding="utf-8")

    archive_path = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir)

    print("\nCondition-sensitivity test complete")
    print(json.dumps({
        "device": str(device),
        "counts_tested": args.counts,
        "unique_raw_outputs": len(all_raw_hashes),
        "deterministic_exact": deterministic_exact,
        "mean_inference_ms": summary["runtime"]["inference_only"]["mean_ms"],
        "mean_morphology_ms": summary["runtime"]["morphology_only"]["mean_ms"],
        "evidence_folder": str(output_dir),
        "upload_file": archive_path,
    }, indent=2))


if __name__ == "__main__":
    main()

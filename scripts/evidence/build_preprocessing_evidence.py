"""Build a compact preprocessing evidence bundle from processed CubiCasa5K data.

The script reads the processed NPZ folders, reproduces the final filtering
checks, summarises the encoded connected-region distribution, and creates
representative visual examples of excluded samples. It does not modify any data.

Run from the project root:
    python3 build_preprocessing_evidence.py --data-dir data

The script prefers:
    data/processed_npz_full
    data/processed_npz_clean_full
and falls back to the non-"_full" folders when needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REQUIRED_KEYS = {"sem", "outline", "room_count", "sample_id"}
DEFAULT_MIN_COUNT = 3
TARGET_SHAPE = (256, 256)
MAX_CLASS_ID = 8

# Class colours used only for compact evidence previews.
PALETTE = {
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


def scalar_to_text(value: Any) -> str:
    """Convert NumPy scalar/0-D array values into readable text."""
    try:
        # Extract the Python scalar stored inside a zero-dimensional NumPy array.
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
    except Exception:
        pass

    return str(value)


def inspect_sample(path: Path, min_count: int) -> dict[str, Any]:
    """Inspect one NPZ and reproduce the final filtering decision."""
    # Initialise a complete result record so failed samples can still
    # be written to the evidence CSV with an explicit exclusion reason.
    result: dict[str, Any] = {
        "filename": path.name,
        "path": str(path),
        "sample_id": "",
        "room_count": None,
        "unique_classes": "",
        "unique_class_count": None,
        "semantic_nonzero_pixels": None,
        "outline_pixels": None,
        "valid": False,
        "reason": "unknown_error",
    }

    try:
        with np.load(path, allow_pickle=True) as data:
            keys = set(data.files)

            # Ensure the processed sample contains every field required
            # by the training and filtering pipeline.
            if not REQUIRED_KEYS.issubset(keys):
                result["reason"] = "missing_required_keys"
                result["missing_keys"] = ",".join(
                    sorted(REQUIRED_KEYS - keys)
                )
                return result

            # Load the semantic mask, filled support mask and historical
            # room_count field containing the encoded connected-region count.
            sem = np.asarray(data["sem"])
            outline = np.asarray(data["outline"])
            room_count = int(
                np.asarray(data["room_count"]).item()
            )
            sample_id = scalar_to_text(data["sample_id"])
            uniq = np.unique(sem)

            # Store descriptive values before applying filtering rules so
            # excluded samples can still be inspected later.
            result.update(
                {
                    "sample_id": sample_id,
                    "room_count": room_count,
                    "unique_classes": ",".join(
                        str(int(x))
                        for x in uniq.tolist()
                    ),
                    "unique_class_count": int(len(uniq)),
                    "semantic_nonzero_pixels": int(
                        np.count_nonzero(sem)
                    ),
                    "outline_pixels": int(
                        np.asarray(outline).sum()
                    ),
                    "sem_shape": "x".join(
                        map(str, sem.shape)
                    ),
                    "outline_shape": "x".join(
                        map(str, outline.shape)
                    ),
                }
            )

            # Keep the same check order as scripts/filter_dataset.py.
            if sem.shape != TARGET_SHAPE:
                result["reason"] = "bad_sem_shape"
                return result

            if outline.shape != TARGET_SHAPE:
                result["reason"] = "bad_outline_shape"
                return result

            # Despite the historical field name, this threshold is applied
            # to the encoded connected-region count.
            if room_count < min_count:
                result["reason"] = "room_count_below_min"
                return result

            if outline.sum() == 0:
                result["reason"] = "empty_outline"
                return result

            if np.count_nonzero(sem) == 0:
                result["reason"] = "empty_semantic_mask"
                return result

            if len(uniq) < 3:
                result["reason"] = "too_few_classes"
                return result

            if (
                int(uniq.min()) < 0
                or int(uniq.max()) > MAX_CLASS_ID
            ):
                result["reason"] = "class_out_of_range"
                return result

            # A sample is retained only after passing every filtering rule.
            result["valid"] = True
            result["reason"] = "kept"
            return result

    except Exception as exc:
        # Record loading failures rather than terminating the complete
        # evidence-generation process.
        result["reason"] = "load_error"
        result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return result


# Convert a semantic class-ID mask into an RGB preview image.
def colourise_semantic(sem: np.ndarray) -> Image.Image:
    rgb = np.zeros(
        (sem.shape[0], sem.shape[1], 3),
        dtype=np.uint8,
    )

    # Apply the predefined display colour associated with each class ID.
    for class_id, colour in PALETTE.items():
        rgb[sem == class_id] = colour

    # Highlight unexpected class IDs in magenta so invalid values
    # remain visually obvious in generated evidence.
    unknown = ~np.isin(
        sem,
        np.array(list(PALETTE.keys())),
    )
    rgb[unknown] = (255, 0, 255)

    return Image.fromarray(rgb, mode="RGB")


# Convert arbitrary text into a filesystem-safe evidence filename.
def safe_filename(text: str) -> str:
    text = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        text,
    )

    # Limit filename length while retaining a non-empty fallback name.
    return text[:140].strip("_") or "sample"


# Render one excluded sample as a side-by-side semantic and support preview.
def render_example(
    npz_path: Path,
    record: dict[str, Any],
    out_path: Path,
) -> None:
    with np.load(npz_path, allow_pickle=True) as data:
        sem = np.asarray(data["sem"])
        outline = np.asarray(data["outline"])

    # Use nearest-neighbour resizing so semantic class boundaries
    # are displayed without interpolation between class IDs.
    sem_img = colourise_semantic(sem).resize(
        (256, 256),
        Image.Resampling.NEAREST,
    )

    outline_img = Image.fromarray(
        outline.astype(np.uint8) * 255,
        mode="L",
    ).convert("RGB")

    outline_img = outline_img.resize(
        (256, 256),
        Image.Resampling.NEAREST,
    )

    # Place the semantic target and filled binary support side by side.
    canvas = Image.new(
        "RGB",
        (560, 330),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    canvas.paste(sem_img, (16, 54))
    canvas.paste(outline_img, (288, 54))

    draw.text(
        (16, 12),
        (
            f"Sample: "
            f"{record.get('sample_id') or record['filename']}"
        ),
        fill="black",
    )
    draw.text(
        (16, 30),
        (
            f"Reason: {record['reason']} | "
            f"encoded count: {record.get('room_count')}"
        ),
        fill="black",
    )
    draw.text(
        (16, 314),
        "Semantic mask",
        fill="black",
    )
    draw.text(
        (288, 314),
        "Binary support mask",
        fill="black",
    )

    canvas.save(out_path)


# Select a deterministic set of excluded samples covering the observed
# filtering reasons and a range of encoded counts.
def select_representative(
    records: list[dict[str, Any]],
    max_examples: int,
) -> list[dict[str, Any]]:
    excluded = [
        r
        for r in records
        if not r["valid"]
    ]

    # Group excluded samples by their first failing filtering rule.
    groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in excluded:
        groups[record["reason"]].append(record)

    selected: list[dict[str, Any]] = []

    # Select examples deterministically within each exclusion category
    # rather than randomly sampling evidence.
    for reason in sorted(groups):
        group = sorted(
            groups[reason],
            key=lambda r: (
                (
                    -1
                    if r.get("room_count") is None
                    else int(r["room_count"])
                ),
                (
                    -1
                    if r.get("unique_class_count") is None
                    else int(r["unique_class_count"])
                ),
                r["filename"],
            ),
        )

        # Allow more examples from the encoded-count filter because it
        # accounts for the main group of excluded samples.
        allowance = (
            6
            if reason == "room_count_below_min"
            else 3
        )
        allowance = min(
            allowance,
            len(group),
        )

        if allowance == 1:
            indices = [0]
        else:
            # Spread selections across the ordered group to avoid taking
            # only neighbouring examples from the same distribution region.
            indices = sorted(
                set(
                    round(
                        i
                        * (len(group) - 1)
                        / (allowance - 1)
                    )
                    for i in range(allowance)
                )
            )

        selected.extend(
            group[i]
            for i in indices
        )

    # Fill remaining slots deterministically if needed.
    selected_keys = {
        r["filename"]
        for r in selected
    }

    for record in sorted(
        excluded,
        key=lambda r: (
            r["reason"],
            r["filename"],
        ),
    ):
        if len(selected) >= max_examples:
            break

        if record["filename"] not in selected_keys:
            selected.append(record)
            selected_keys.add(record["filename"])

    return selected[:max_examples]


# Write a collection of evidence records using a fixed CSV column order.
def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


# Locate the matching processed and clean dataset directories.
def find_data_pair(
    data_dir: Path,
    source_name: str | None,
    clean_name: str | None,
) -> tuple[Path, Path]:
    # Use explicitly supplied folder names when both are provided.
    if source_name and clean_name:
        source = data_dir / source_name
        clean = data_dir / clean_name

    else:
        # Prefer the final full-dataset folders and fall back to
        # the earlier non-full folder names when necessary.
        preferred = [
            (
                "processed_npz_full",
                "processed_npz_clean_full",
            ),
            (
                "processed_npz",
                "processed_npz_clean",
            ),
        ]

        source = clean = None

        for (
            source_candidate,
            clean_candidate,
        ) in preferred:
            candidate_source = (
                data_dir / source_candidate
            )
            candidate_clean = (
                data_dir / clean_candidate
            )

            if (
                candidate_source.is_dir()
                and candidate_clean.is_dir()
            ):
                source, clean = candidate_source, candidate_clean
                break

        if source is None or clean is None:
            raise FileNotFoundError(
                "Could not find a processed/clean folder pair. Expected either "
                "processed_npz_full + processed_npz_clean_full or "
                "processed_npz + processed_npz_clean."
            )

    # Validate both directories before attempting to inspect samples.
    if not source.is_dir():
        raise FileNotFoundError(
            f"Processed source folder not found: {source}"
        )

    if not clean.is_dir():
        raise FileNotFoundError(
            f"Clean folder not found: {clean}"
        )

    return source, clean


# Build the complete preprocessing evidence bundle without modifying source data.
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a compact CubiCasa5K preprocessing evidence bundle."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Project data folder.",
    )
    parser.add_argument(
        "--source-name",
        default=None,
        help="Optional processed source folder name.",
    )
    parser.add_argument(
        "--clean-name",
        default=None,
        help="Optional clean destination folder name.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=DEFAULT_MIN_COUNT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preprocessing_evidence"),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=12,
    )
    args = parser.parse_args()

    # Resolve the processed/clean folder pair before starting inspection.
    try:
        source_dir, clean_dir = find_data_pair(
            args.data_dir,
            args.source_name,
            args.clean_name,
        )
    except FileNotFoundError as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 2

    # Create the evidence output folders.
    output_dir = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    examples_dir = (
        output_dir / "representative_excluded"
    )
    examples_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load complete filename lists from the processed and clean datasets.
    source_files = sorted(
        source_dir.glob("*.npz")
    )
    clean_files = sorted(
        clean_dir.glob("*.npz")
    )

    if not source_files:
        print(
            f"ERROR: No NPZ files found in {source_dir}",
            file=sys.stderr,
        )
        return 3

    clean_names = {
        p.name
        for p in clean_files
    }
    source_names = {
        p.name
        for p in source_files
    }

    records: list[dict[str, Any]] = []

    # Reproduce the filtering decision for every processed floor sample.
    for index, path in enumerate(
        source_files,
        start=1,
    ):
        record = inspect_sample(
            path,
            args.min_count,
        )

        # Compare the reproduced decision with actual membership
        # of the retained clean-data folder.
        record["present_in_clean_folder"] = (
            path.name in clean_names
        )

        records.append(record)

        if (
            index % 500 == 0
            or index == len(source_files)
        ):
            print(
                f"Inspected {index}/"
                f"{len(source_files)} samples"
            )

    # Summarise the first failing rule assigned to each processed sample.
    reason_counts = Counter(
        r["reason"]
        for r in records
    )

    valid_names = {
        r["filename"]
        for r in records
        if r["valid"]
    }

    invalid_names = {
        r["filename"]
        for r in records
        if not r["valid"]
    }

    # Build encoded connected-region count distributions before and
    # after filtering while preserving the historical room_count field.
    counts_all = Counter(
        int(r["room_count"])
        for r in records
        if r.get("room_count") is not None
    )

    counts_kept = Counter(
        int(r["room_count"])
        for r in records
        if (
            r["valid"]
            and r.get("room_count") is not None
        )
    )

    counts_excluded = Counter(
        int(r["room_count"])
        for r in records
        if (
            not r["valid"]
            and r.get("room_count") is not None
        )
    )

    count_values = sorted(
        set(counts_all)
        | set(counts_kept)
        | set(counts_excluded)
    )

    distribution_rows = [
        {
            "encoded_connected_region_count": count,
            "processed_samples": counts_all[count],
            "kept_samples": counts_kept[count],
            "excluded_samples": counts_excluded[count],
        }
        for count in count_values
    ]

    # Calculate descriptive statistics over every available encoded count.
    count_numeric = [
        int(r["room_count"])
        for r in records
        if r.get("room_count") is not None
    ]

    # Reconcile the reproduced filtering decisions with the actual
    # contents of the retained clean-data folder.
    summary = {
        "source_folder": str(source_dir),
        "clean_folder": str(clean_dir),
        "minimum_encoded_count_threshold": args.min_count,
        "total_processed_npz": len(source_files),
        "total_clean_npz": len(clean_files),
        "filter_valid_count": len(valid_names),
        "filter_excluded_count": len(invalid_names),
        "excluded_by_reason": dict(
            sorted(reason_counts.items())
        ),
        "source_minus_clean_count": len(
            source_names - clean_names
        ),
        "clean_minus_source_count": len(
            clean_names - source_names
        ),
        "valid_missing_from_clean_count": len(
            valid_names - clean_names
        ),
        "invalid_present_in_clean_count": len(
            invalid_names & clean_names
        ),
        "encoded_count_statistics": {
            "minimum": (
                min(count_numeric)
                if count_numeric
                else None
            ),
            "maximum": (
                max(count_numeric)
                if count_numeric
                else None
            ),
            "mean": (
                float(np.mean(count_numeric))
                if count_numeric
                else None
            ),
            "median": (
                float(np.median(count_numeric))
                if count_numeric
                else None
            ),
        },
    }

    # Use a common field order for the complete and excluded-sample CSV files.
    all_fields = [
        "filename",
        "sample_id",
        "room_count",
        "unique_classes",
        "unique_class_count",
        "semantic_nonzero_pixels",
        "outline_pixels",
        "sem_shape",
        "outline_shape",
        "valid",
        "reason",
        "present_in_clean_folder",
        "missing_keys",
        "error",
    ]

    write_csv(
        output_dir / "all_processed_samples.csv",
        records,
        all_fields,
    )

    write_csv(
        output_dir / "excluded_samples.csv",
        [
            r
            for r in records
            if not r["valid"]
        ],
        all_fields,
    )

    write_csv(
        output_dir
        / "connected_region_count_distribution.csv",
        distribution_rows,
        [
            "encoded_connected_region_count",
            "processed_samples",
            "kept_samples",
            "excluded_samples",
        ],
    )

    # Save the headline filtering totals and consistency checks in JSON form.
    with (
        output_dir / "preprocessing_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    # Select and render representative excluded samples for visual inspection.
    selected = select_representative(
        records,
        args.max_examples,
    )

    selected_rows: list[
        dict[str, Any]
    ] = []

    rendered_paths: list[Path] = []

    source_lookup = {
        p.name: p
        for p in source_files
    }

    for number, record in enumerate(
        selected,
        start=1,
    ):
        image_name = (
            f"{number:02d}_"
            f"{safe_filename(record['reason'])}_"
            f"{safe_filename(record['filename'])}.png"
        )

        image_path = (
            examples_dir / image_name
        )

        try:
            render_example(
                source_lookup[
                    record["filename"]
                ],
                record,
                image_path,
            )

            rendered_paths.append(
                image_path
            )

            selected_rows.append(
                {
                    **record,
                    "preview_file": str(
                        image_path.relative_to(
                            output_dir
                        )
                    ),
                }
            )

        except Exception as exc:
            # Preserve the sample metadata even if its visual preview fails.
            selected_rows.append(
                {
                    **record,
                    "preview_file": "",
                    "preview_error": str(exc),
                }
            )

    selected_fields = (
        all_fields
        + [
            "preview_file",
            "preview_error",
        ]
    )

    write_csv(
        output_dir
        / "selected_excluded_examples.csv",
        selected_rows,
        selected_fields,
    )

    # Contact sheet for quick appendix selection.
    if rendered_paths:
        thumbnails: list[
            Image.Image
        ] = []

        for path in rendered_paths:
            with Image.open(path) as img:
                thumb = img.convert("RGB")
                thumb.thumbnail((560, 330))
                thumbnails.append(
                    thumb.copy()
                )

        # Arrange preview images in two columns.
        columns = 2
        rows = math.ceil(
            len(thumbnails) / columns
        )

        sheet = Image.new(
            "RGB",
            (
                columns * 560,
                rows * 330,
            ),
            "white",
        )

        for i, thumb in enumerate(
            thumbnails
        ):
            sheet.paste(
                thumb,
                (
                    (i % columns) * 560,
                    (i // columns) * 330,
                ),
            )

        sheet.save(
            output_dir
            / "excluded_examples_contact_sheet.png"
        )

    # Generate a text summary describing the evidence artefacts.
    # This existing output string is retained unchanged.
    readme = f"""ARP preprocessing evidence bundle

Generated from:
- Processed folder: {source_dir}
- Clean folder: {clean_dir}
- Minimum encoded connected-region threshold: {args.min_count}

Key totals:
- Processed NPZ samples: {len(source_files)}
- Clean/retained NPZ samples: {len(clean_files)}
- Excluded by reproduced filter: {len(invalid_names)}

Files:
- preprocessing_summary.json: headline totals and reconciliation checks.
- all_processed_samples.csv: one row per processed sample.
- excluded_samples.csv: one row per excluded sample and the first failing rule.
- connected_region_count_distribution.csv: count distribution before and after filtering.
- selected_excluded_examples.csv: metadata for appendix-ready examples.
- excluded_examples_contact_sheet.png: compact visual overview.
- representative_excluded/: individual visual examples.

Terminology note:
The NPZ key is named room_count in the implementation, but it is generated by
8-connected component analysis of the combined non-background, non-wall mask.
For the dissertation, describe it as an encoded connected-region count unless a
separate room-instance validation supports stronger wording.
"""

    (
        output_dir / "README.txt"
    ).write_text(
        readme,
        encoding="utf-8",
    )

    # Package the generated evidence into a compressed archive while
    # retaining the same relative folder structure.
    zip_path = output_dir.with_name(
        output_dir.name + "_bundle.zip"
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted(
            output_dir.rglob("*")
        ):
            if path.is_file():
                archive.write(
                    path,
                    arcname=str(
                        path.relative_to(
                            output_dir.parent
                        )
                    ),
                )

    print("\nSummary")
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )
    print(
        f"\nEvidence folder: "
        f"{output_dir.resolve()}"
    )
    print(
        f"Upload this compact file: "
        f"{zip_path.resolve()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
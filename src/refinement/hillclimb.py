import numpy as np
import cv2


BG = 0


def compactness(mask: np.ndarray) -> float:
    """
    Compute compactness:
        4*pi*A / P^2
    """

    area = float(mask.sum())

    if area <= 0:
        return 0.0

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0

    perimeter = 0.0

    for cnt in contours:
        perimeter += cv2.arcLength(cnt, True)

    if perimeter <= 1e-6:
        return 0.0

    return float((4.0 * np.pi * area) / (perimeter * perimeter))


def refine_class_hillclimb(
    class_mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 3,
):
    """
    Lightweight hill-climbing refinement.

    Try local morphology operations and keep changes
    only if compactness improves.
    """

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    current = class_mask.astype(np.uint8)
    best_score = compactness(current)

    operations = [
        lambda x: cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel),
        lambda x: cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel),
        lambda x: cv2.dilate(x, kernel, iterations=1),
        lambda x: cv2.erode(x, kernel, iterations=1),
    ]

    for _ in range(iterations):

        improved = False

        for op in operations:

            candidate = op(current)

            score = compactness(candidate)

            if score > best_score:
                current = candidate
                best_score = score
                improved = True

        if not improved:
            break

    return current


def refine_semantic_mask_hillclimb(
    mask: np.ndarray,
    num_classes: int = 9,
    ignore_classes=(BG,),
    kernel_size: int = 3,
    iterations: int = 3,
):
    """
    Apply hill-climb refinement independently
    to each semantic class.
    """

    refined = np.full_like(mask, fill_value=BG, dtype=np.uint8)

    for class_id in range(num_classes):

        if class_id in ignore_classes:
            continue

        class_mask = (mask == class_id).astype(np.uint8)

        if class_mask.sum() == 0:
            continue

        refined_class = refine_class_hillclimb(
            class_mask,
            kernel_size=kernel_size,
            iterations=iterations,
        )

        refined[refined_class > 0] = class_id

    return refined
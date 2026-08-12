import numpy as np
import cv2

BG = 0

def compactness(mask: np.ndarray) -> float:
    """
    Compute compactness:
        4*pi*A / P^2
    """

    # Use the number of foreground pixels as the region area.
    area = float(mask.sum())

    # An empty region has no meaningful compactness score.
    if area <= 0:
        return 0.0

    # Extract only external contours for perimeter measurement.
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0

    # Sum the closed-contour perimeter of all detected foreground regions.
    perimeter = 0.0

    for cnt in contours:
        perimeter += cv2.arcLength(cnt, True)

    # Avoid division by zero or unstable values for negligible perimeters.
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

    # Create the structuring element shared by all candidate operations.
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Initialise the search from the original binary class mask.
    current = class_mask.astype(np.uint8)
    best_score = compactness(current)

    # Define the local morphological changes considered at each search step.
    operations = [
        lambda x: cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel),
        lambda x: cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel),
        lambda x: cv2.dilate(x, kernel, iterations=1),
        lambda x: cv2.erode(x, kernel, iterations=1),
    ]

    # Repeat the greedy search up to the specified maximum number of iterations.
    for _ in range(iterations):

        improved = False

        for op in operations:

            # Apply one candidate operation to the current best mask.
            candidate = op(current)

            # Evaluate the candidate using the compactness objective.
            score = compactness(candidate)

            # Accept the candidate only when it improves the current best score.
            if score > best_score:
                current = candidate
                best_score = score
                improved = True

        # Stop early when none of the candidate operations improves compactness.
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

    # Start with a background-only output before reconstructing refined classes.
    refined = np.full_like(mask, fill_value=BG, dtype=np.uint8)

    # Process semantic classes independently so the hill-climb search
    # operates on binary masks rather than directly on multi-class labels.
    for class_id in range(num_classes):

        if class_id in ignore_classes:
            continue

        # Extract the current semantic class as a binary foreground mask.
        class_mask = (mask == class_id).astype(np.uint8)

        # Skip classes that are absent from the prediction.
        if class_mask.sum() == 0:
            continue

        # Optimise the current class using compactness as the acceptance criterion.
        refined_class = refine_class_hillclimb(
            class_mask,
            kernel_size=kernel_size,
            iterations=iterations,
        )

        # Reconstruct the multi-class output. Later class IDs overwrite
        # earlier ones if independently refined regions overlap.
        refined[refined_class > 0] = class_id

    return refined
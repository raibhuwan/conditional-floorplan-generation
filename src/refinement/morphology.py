import numpy as np
import cv2


BG = 0
WALL = 8


def remove_small_components(binary_mask: np.ndarray, min_area: int = 30) -> np.ndarray:
    """
    Remove small connected components from a binary mask.
    """
    # Convert the input to the format expected by OpenCV connected-component analysis.
    binary_mask = binary_mask.astype(np.uint8)

    # Identify eight-connected foreground components and obtain their statistics.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)

    # Component 0 is the background, so only foreground components are examined.
    for comp_id in range(1, n):
        area = stats[comp_id, cv2.CC_STAT_AREA]

        # Retain only components that satisfy the minimum-area threshold.
        if area >= min_area:
            cleaned[labels == comp_id] = 1

    return cleaned


def refine_semantic_mask_morphology(
    mask: np.ndarray,
    num_classes: int = 9,
    ignore_classes=(BG,),
    kernel_size: int = 3,
    min_area: int = 30,
) -> np.ndarray:
    """
    Lightweight morphology-based refinement for semantic floor plan masks.

    Input:
        mask: [H, W] semantic class id mask

    Output:
        refined: [H, W] refined semantic class id mask

    Main idea:
        - process each semantic class separately
        - remove tiny noisy regions
        - smooth boundaries with opening/closing
        - preserve major semantic regions
    """

    # Start with a background-only output and add each refined class separately.
    refined = np.full_like(mask, fill_value=BG, dtype=np.uint8)

    # Create the structuring element used by the opening and closing operations.
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Process each semantic class independently rather than applying
    # morphology directly to the multi-class mask.
    for class_id in range(num_classes):
        if class_id in ignore_classes:
            continue

        # Extract the current semantic class as a binary mask.
        class_mask = (mask == class_id).astype(np.uint8)

        # Skip processing when the class is not present in the prediction.
        if class_mask.sum() == 0:
            continue

        # Remove small isolated noise first
        class_mask = remove_small_components(class_mask, min_area=min_area)

        # Smooth small boundary noise
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)

        # Fill small holes/gaps
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)

        # Remove small artefacts again after morphology
        class_mask = remove_small_components(class_mask, min_area=min_area)

        # Reconstruct the multi-class mask from the refined binary class region.
        # Later class IDs overwrite earlier ones if morphological regions overlap.
        refined[class_mask > 0] = class_id

    return refined
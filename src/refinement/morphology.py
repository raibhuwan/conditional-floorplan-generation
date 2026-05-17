import numpy as np
import cv2


BG = 0
WALL = 8


def remove_small_components(binary_mask: np.ndarray, min_area: int = 30) -> np.ndarray:
    """
    Remove small connected components from a binary mask.
    """
    binary_mask = binary_mask.astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)

    for comp_id in range(1, n):
        area = stats[comp_id, cv2.CC_STAT_AREA]
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

    refined = np.full_like(mask, fill_value=BG, dtype=np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for class_id in range(num_classes):
        if class_id in ignore_classes:
            continue

        class_mask = (mask == class_id).astype(np.uint8)

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

        refined[class_mask > 0] = class_id

    return refined
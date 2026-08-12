import torch
import torch.nn.functional as F


def to_one_hot(mask, num_classes):
    """
    Convert semantic mask:
        [B,H,W]
    into one-hot:
        [B,C,H,W]
    """

    # Expand each class ID into a separate binary class channel.
    one_hot = F.one_hot(mask, num_classes=num_classes)

    # Move the class dimension before the spatial dimensions and convert
    # the result to floating point for use by neural-network components.
    one_hot = one_hot.permute(0, 3, 1, 2).float()

    # Return the final batch of one-hot semantic masks.
    return one_hot
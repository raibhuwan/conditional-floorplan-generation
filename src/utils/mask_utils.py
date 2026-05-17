import torch
import torch.nn.functional as F


def to_one_hot(mask, num_classes):
    """
    Convert semantic mask:
        [B,H,W]
    into one-hot:
        [B,C,H,W]
    """
    one_hot = F.one_hot(mask, num_classes=num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).float()
    return one_hot
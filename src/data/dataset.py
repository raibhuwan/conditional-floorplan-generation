import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class FloorplanNPZDataset(Dataset):
    """
    Load processed floor-plan samples stored as NPZ files.

    Returns:
        x: FloatTensor [2, H, W]
            - channel 0: filled binary support mask
            - channel 1: normalised encoded connected-region count
              repeated as a constant spatial channel

        y: LongTensor [H, W]
            - semantic class IDs from 0 to 8
    """

    # Locate all processed samples and store the maximum count used for normalisation.
    def __init__(self, npz_dir: str, max_count: int = 13):
        self.npz_dir = npz_dir

        # Sort filenames so sample ordering remains consistent when the dataset is loaded.
        self.files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))

        # Stop early if the processed dataset directory contains no samples.
        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in: {npz_dir}")

        self.max_count = max_count

    # Return the number of processed samples available in the dataset.
    def __len__(self):
        return len(self.files)

    # Load one processed sample and construct its model input and semantic target.
    def __getitem__(self, idx: int):
        d = np.load(self.files[idx], allow_pickle=True)

        # Load the semantic target, filled support mask and encoded connected-region count.
        sem = d["sem"].astype(np.int64)              # [H,W] ints for CE loss
        outline = d["outline"].astype(np.float32)    # [H,W] float
        room_count = float(d["room_count"])

        # Normalise the encoded connected-region count to [0, 1].
        count_norm = min(room_count / float(self.max_count), 1.0)

        # Repeat the normalised scalar across the spatial dimensions so it can
        # be supplied to the convolutional model as a second input channel.
        count_chan = np.full_like(
            outline,
            fill_value=count_norm,
            dtype=np.float32
        )

        # Stack the support mask and count channel into the final [2,H,W] input tensor.
        x = np.stack([outline, count_chan], axis=0)

        # Convert NumPy arrays into PyTorch tensors with the required data types.
        x = torch.from_numpy(x)      # float32
        y = torch.from_numpy(sem)    # int64

        # Return the two-channel condition together with its semantic target mask.
        return x, y
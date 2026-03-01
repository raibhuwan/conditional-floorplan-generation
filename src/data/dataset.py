import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class FloorplanNPZDataset(Dataset):
    """
    Loads .npz files produced by preprocessing.

    Returns:
      x: FloatTensor [2, H, W]
         - channel 0: outline (0/1)
         - channel 1: room_count normalized (constant map)
      y: LongTensor [H, W]
         - semantic class ids 0..8
    """
    def __init__(self, npz_dir: str, max_count: int = 13):
        self.npz_dir = npz_dir
        self.files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in: {npz_dir}")
        self.max_count = max_count

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx], allow_pickle=True)

        sem = d["sem"].astype(np.int64)              # [H,W] ints for CE loss
        outline = d["outline"].astype(np.float32)    # [H,W] float
        room_count = float(d["room_count"])

        # Normalize room_count to [0,1]
        count_norm = min(room_count / float(self.max_count), 1.0)
        count_chan = np.full_like(outline, fill_value=count_norm, dtype=np.float32)

        # Stack into [2,H,W]
        x = np.stack([outline, count_chan], axis=0)

        x = torch.from_numpy(x)      # float32
        y = torch.from_numpy(sem)    # int64

        return x, y
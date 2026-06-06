import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """Streams pickled patches of shape (C, L, L, L) from a directory.

    Path is explicit (no os.getcwd). Data is already channel-first, so it
    maps straight to a torch tensor with no axis swapping.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.paths = sorted(self.root.glob("*.pickle"))
        if not self.paths:
            raise FileNotFoundError(f"no *.pickle patches in {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with open(self.paths[idx], "rb") as f:
            arr = pickle.load(f)
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))

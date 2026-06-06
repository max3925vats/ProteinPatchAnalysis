import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..patches import AtomPatch, voxelize
from ..spec import PatchSpec


class PatchDataset(Dataset):
    """Streams AtomPatch pickles from a directory, voxelizing on load.

    The on-disk artifact is the representation-agnostic atom-set; this
    dataset turns it into the (C, L, L, L) grid the conv VAE consumes.
    Results are cached in memory after first access (idempotent reads).

    The cache is unbounded (one grid per accessed patch). Fine for the
    expected scale (hundreds–thousands of patches); add an eviction policy
    before scaling to tens of thousands.
    """

    def __init__(self, root: str | Path, spec: PatchSpec):
        self.root = Path(root)
        self.spec = spec
        self.paths = sorted(self.root.glob("*.pickle"))
        if not self.paths:
            raise FileNotFoundError(f"no *.pickle patches in {self.root}")
        self._cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx not in self._cache:
            with open(self.paths[idx], "rb") as f:
                patch: AtomPatch = pickle.load(f)
            # voxelize returns (C, L, L, L) float32 numpy array
            grid = voxelize(patch, self.spec)
            self._cache[idx] = torch.from_numpy(
                np.ascontiguousarray(grid, dtype=np.float32))
        return self._cache[idx]

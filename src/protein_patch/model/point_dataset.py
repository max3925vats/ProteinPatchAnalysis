import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..atom_types import atom_channel
from ..patches import AtomPatch, voxelize
from ..spec import PatchSpec


def _point_features(patch: AtomPatch) -> np.ndarray:
    """(N, 7) per-atom features = centered xyz (3) + CNOS one-hot (4).

    Filters to atoms whose element is in (C, N, O, S) — the SAME set the
    voxelizer keeps — so the point and voxel views describe identical atoms
    (feature parity).
    """
    rows = []
    for xyz, el in zip(patch.coords, patch.elements):
        oneh = atom_channel(el)
        if oneh.sum() == 0:          # skip H and exotic atoms, as voxelize does
            continue
        rows.append(np.concatenate([np.asarray(xyz, dtype=np.float32), oneh]))
    if not rows:
        return np.empty((0, 7), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


class PointPatchDataset(Dataset):
    """Streams AtomPatch pickles as (point_features, target_grid) pairs.

    point_features is the CNOS-filtered atom set (N, 7); target_grid is the
    voxelized patch (C, L, L, L) — the reconstruction target shared with the
    voxel path. Same on-disk artifact as PatchDataset; different view of it.
    """

    def __init__(self, root: str | Path, spec: PatchSpec):
        self.root = Path(root)
        self.spec = spec
        self.paths = sorted(self.root.glob("*.pickle"))
        if not self.paths:
            raise FileNotFoundError(f"no *.pickle patches in {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        with open(self.paths[idx], "rb") as f:
            patch: AtomPatch = pickle.load(f)
        point_feats = _point_features(patch)
        if point_feats.shape[0] == 0:
            # No CNOS atoms -> the masked max-pool would reduce over an empty
            # set. Fail explicitly with provenance rather than crash downstream.
            raise ValueError(
                f"patch {self.paths[idx].name} has no C/N/O/S atoms to encode")
        feats = torch.from_numpy(point_feats)
        grid = torch.from_numpy(
            np.ascontiguousarray(voxelize(patch, self.spec), dtype=np.float32))
        return feats, grid


def pad_collate(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """Collate variable-N point samples into a padded, masked batch.

    Returns ((feats (B, Nmax, 7), mask (B, Nmax) bool), targets (B, C, L, L, L)).
    mask is True for real atoms, False for padding.
    """
    feats_list = [b[0] for b in batch]
    targets = torch.stack([b[1] for b in batch])
    n_max = max(f.shape[0] for f in feats_list)
    B, feat_dim = len(batch), feats_list[0].shape[1]
    feats = torch.zeros(B, n_max, feat_dim, dtype=torch.float32)
    mask = torch.zeros(B, n_max, dtype=torch.bool)
    for i, f in enumerate(feats_list):
        n = f.shape[0]
        feats[i, :n] = f
        mask[i, :n] = True
    return (feats, mask), targets

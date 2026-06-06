import pickle
import numpy as np
import torch
import pytest

from protein_patch.spec import PatchSpec
from protein_patch.patches import AtomPatch
from protein_patch.model.dataset import PatchDataset


def _atom_patch(rng, n=5):
    coords = (rng.random((n, 3)).astype("float32") - 0.5) * 3.0   # within cube
    return AtomPatch(coords, ["C"] * n, {"resname": "ALA"}, ("x", "A", 1, "", "ALA"))


def test_dataset_voxelizes_atompatch_on_load(tmp_path, rng):
    spec = PatchSpec(grid_voxels=16, voxel_size=0.5)
    for i in range(3):
        with open(tmp_path / f"p{i}.pickle", "wb") as f:
            pickle.dump(_atom_patch(rng), f)
    ds = PatchDataset(tmp_path, spec)
    assert len(ds) == 3
    x = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (4, 16, 16, 16)
    assert x.dtype == torch.float32
    assert x.max() > 0
    assert torch.equal(ds[0], ds[0])


def test_dataset_raises_on_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        PatchDataset(tmp_path, PatchSpec(grid_voxels=16))

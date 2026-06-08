import pickle

import pytest
import torch

from protein_patch.spec import PatchSpec
from protein_patch.patches import AtomPatch
from protein_patch.model.point_dataset import PointPatchDataset, pad_collate


def _write(tmp_path, name, coords, elements):
    p = AtomPatch(coords, elements, {"rel_sasa": 0.5}, ("x", "A", 1, "", "ALA"))
    with open(tmp_path / name, "wb") as f:
        pickle.dump(p, f)


def test_item_returns_point_feats_and_grid_target(tmp_path, rng):
    spec = PatchSpec(grid_voxels=16)
    coords = (rng.random((5, 3)).astype("float32") - 0.5) * 3.0
    _write(tmp_path, "p0.pickle", coords, ["C", "N", "O", "S", "C"])
    ds = PointPatchDataset(tmp_path, spec)
    feats, target = ds[0]
    assert feats.shape == (5, 7)                      # xyz(3) + CNOS one-hot(4)
    assert target.shape == (4, 16, 16, 16)
    assert target.dtype == torch.float32


def test_cnos_filtering_drops_other_elements(tmp_path, rng):
    spec = PatchSpec(grid_voxels=16)
    coords = (rng.random((4, 3)).astype("float32") - 0.5) * 3.0
    _write(tmp_path, "p0.pickle", coords, ["C", "H", "N", "H"])   # 2 of 4 are H
    ds = PointPatchDataset(tmp_path, spec)
    feats, _ = ds[0]
    assert feats.shape == (2, 7)                      # only C and N survive


def test_pad_collate_pads_and_masks(tmp_path, rng):
    spec = PatchSpec(grid_voxels=16)
    _write(tmp_path, "a.pickle", (rng.random((3, 3)).astype("float32") - .5) * 3, ["C"] * 3)
    _write(tmp_path, "b.pickle", (rng.random((6, 3)).astype("float32") - .5) * 3, ["C"] * 6)
    ds = PointPatchDataset(tmp_path, spec)
    (feats, mask), targets = pad_collate([ds[0], ds[1]])
    assert feats.shape == (2, 6, 7)                   # padded to max N = 6
    assert mask.shape == (2, 6) and mask.dtype == torch.bool
    assert mask[0].sum() == 3 and mask[1].sum() == 6  # real-atom counts
    assert targets.shape == (2, 4, 16, 16, 16)


def test_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PointPatchDataset(tmp_path, PatchSpec(grid_voxels=16))


def test_patch_with_no_cnos_atoms_raises(tmp_path, rng):
    spec = PatchSpec(grid_voxels=16)
    coords = (rng.random((3, 3)).astype("float32") - 0.5) * 3.0
    _write(tmp_path, "p0.pickle", coords, ["H", "H", "H"])   # nothing to encode
    ds = PointPatchDataset(tmp_path, spec)
    with pytest.raises(ValueError):
        _ = ds[0]

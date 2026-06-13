import pickle

import numpy as np
import pytest
import torch

from protein_patch.spec import PatchSpec
from protein_patch.config import ContrastiveConfig
from protein_patch.patches import AtomPatch
from protein_patch.model.encoders import VoxelEncoder, PointCloudEncoder
from protein_patch.contrastive_train import (
    ContrastiveViewDataset, train_contrastive, encode_dataset, embedding_std,
)


def _fresh_encoder(kind, spec):
    return (VoxelEncoder(spec) if kind == "voxel"
            else PointCloudEncoder(in_dim=7, feature_dim=128))


@pytest.fixture
def structured(tmp_path, rng):
    """8 distinct patches (each clustered around its own center) -> a learnable
    contrastive task; two proteins so same-protein masking is exercised."""
    spec = PatchSpec(grid_voxels=16)
    d = tmp_path / "train"; d.mkdir()
    for i in range(8):
        center = np.array([i - 4, 0, 0], dtype="float32")
        coords = center + (rng.random((6, 3)).astype("float32") - 0.5)
        pid = f"prot{i % 2}"
        patch = AtomPatch(coords, ["C", "N", "O", "S", "C", "N"],
                          {"rel_sasa": 0.5}, (pid, "A", i, "", "ALA"))
        with open(d / f"p{i}.pickle", "wb") as f:
            pickle.dump(patch, f)
    return spec, str(d)


def test_view_dataset_two_differing_views(structured):
    spec, d = structured
    ds = ContrastiveViewDataset("voxel", d, spec, ContrastiveConfig())
    v1, v2, pid = ds[0]
    assert v1.shape == (4, 16, 16, 16)
    assert not torch.allclose(v1, v2)            # independent augmentations
    assert isinstance(pid, str)


def test_view_dataset_empty_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ContrastiveViewDataset("voxel", str(tmp_path), PatchSpec(grid_voxels=16),
                               ContrastiveConfig())


def test_embedding_std_zero_on_collapse_positive_on_spread():
    assert embedding_std(np.ones((10, 4))) == pytest.approx(0.0)
    assert embedding_std(np.random.default_rng(0).random((10, 4))) > 0.05


@pytest.mark.parametrize("kind", ["voxel", "point"])
def test_train_contrastive_decreases_and_does_not_collapse(structured, kind):
    spec, d = structured
    ccfg = ContrastiveConfig(epochs=25, batch_size=4, proj_dim=16, hidden=32, seed=0)
    std_untrained = embedding_std(encode_dataset(_fresh_encoder(kind, spec), d, spec, kind))
    hist, encoder = train_contrastive(kind, d, spec, ccfg)
    assert all(np.isfinite(v) for v in hist["loss"])
    assert hist["loss"][-1] < hist["loss"][0]            # learning happened
    # anti-collapse: the trained encoder still spreads the distinct patches,
    # not far below an untrained baseline (a collapsed encoder -> std ~ 0).
    std_trained = embedding_std(encode_dataset(encoder, d, spec, kind))
    assert std_trained > 0.2 * std_untrained


def test_train_contrastive_exclude_center_runs(tmp_path, rng):
    # environment-only training: central atom removed before the views are built.
    spec = PatchSpec(grid_voxels=16)
    d = tmp_path / "train"; d.mkdir()
    for i in range(6):
        coords = (rng.random((6, 3)).astype("float32") - 0.5) * 3.0
        mask = np.zeros(6, dtype=bool); mask[0] = True          # one central atom
        patch = AtomPatch(coords, ["C", "N", "O", "S", "C", "N"],
                          {"rel_sasa": 0.5}, (f"p{i % 2}", "A", i, "", "ALA"), mask)
        with open(d / f"p{i}.pickle", "wb") as f:
            pickle.dump(patch, f)
    ccfg = ContrastiveConfig(epochs=2, batch_size=4, proj_dim=16, hidden=32)
    hist, enc = train_contrastive("point", str(d), spec, ccfg, exclude_center_atoms=True)
    assert all(np.isfinite(v) for v in hist["loss"])


def test_point_view_never_empty_with_hydrogens(tmp_path, rng):
    # 1 CNOS atom (C) among many H + aggressive dropout: the CNOS fallback (C7)
    # must keep every point view encodable rather than producing an empty tensor.
    spec = PatchSpec(grid_voxels=16)
    d = tmp_path / "t"; d.mkdir()
    coords = (rng.random((6, 3)).astype("float32") - 0.5) * 3.0
    patch = AtomPatch(coords, ["C", "H", "H", "H", "H", "H"],
                      {"rel_sasa": 0.5}, ("p", "A", 1, "", "ALA"))
    with open(d / "p.pickle", "wb") as f:
        pickle.dump(patch, f)
    ds = ContrastiveViewDataset("point", str(d), spec, ContrastiveConfig(drop_frac=0.8, seed=0))
    for _ in range(10):
        v1, v2, _pid = ds[0]
        assert v1.shape[0] >= 1 and v2.shape[0] >= 1

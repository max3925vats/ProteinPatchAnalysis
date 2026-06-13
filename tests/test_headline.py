import pickle

import numpy as np
import pytest

from protein_patch.spec import PatchSpec
from protein_patch.config import ContrastiveConfig
from protein_patch.patches import AtomPatch
from protein_patch.contrastive_train import train_contrastive, encode_patches
from protein_patch.headline import (
    burial_task, identity_task, ss_task, build_matrix, capacity_sensitivity,
)


def _synthetic_dssp(root):
    # (chain, resseq) -> 8-state; our synthetic patches are chain "A", resseq 0..5
    return {("A", i): ("H" if i % 2 else "E") for i in range(6)}


def _write_split(d, rng):
    d.mkdir()
    for i in range(6):
        coords = (rng.random((6, 3)).astype("float32") - 0.5) * 3.0
        mask = np.zeros(6, dtype=bool); mask[0] = True            # central atom
        res = "ALA" if i % 2 == 0 else "VAL"                       # 2 identity classes
        rel = 0.2 + 0.8 * (i / 5)                                  # spread for burial
        patch = AtomPatch(coords, ["C", "N", "O", "S", "C", "N"],
                          {"rel_sasa": rel}, (f"p{i % 2}", "A", i, "", res), mask)
        with open(d / f"p{i}.pickle", "wb") as f:
            pickle.dump(patch, f)


@pytest.fixture
def dataset(tmp_path, rng):
    _write_split(tmp_path / "train", rng)
    _write_split(tmp_path / "val", rng)
    return PatchSpec(grid_voxels=16), str(tmp_path / "train"), str(tmp_path / "val")


def _ccfg(**kw):
    base = dict(epochs=1, batch_size=4, proj_dim=8, hidden=16)
    base.update(kw)
    return ContrastiveConfig(**base)


def test_build_matrix_populates_all_six_cells_and_routes_encoder(dataset):
    spec, tr, va = dataset
    tasks = [burial_task(), ss_task(_synthetic_dssp), identity_task()]
    out = build_matrix(tr, va, spec, _ccfg(), tasks, seeds=(0, 1))
    assert set(out) == {(k, t) for k in ("voxel", "point")
                        for t in ("burial", "ss", "identity")}      # all 6 cells
    for cell in out.values():
        assert 0.0 <= cell["knn_mean"] <= 1.0 and 0.0 <= cell["linear_mean"] <= 1.0
        assert cell["knn_std"] >= 0.0 and cell["n_seeds"] == 2
    # only identity uses the center-excluded encoder
    assert out[("voxel", "identity")]["use_excluded_encoder"] is True
    assert out[("point", "burial")]["use_excluded_encoder"] is False
    assert out[("point", "ss")]["use_excluded_encoder"] is False


def test_exclusion_actually_changes_the_embeddings(dataset):
    # H3 non-vacuity: the center-excluded path must change the encoder input, not
    # just flip a metadata flag.
    spec, tr, va = dataset
    _, enc = train_contrastive("voxel", tr, spec, _ccfg())
    full = encode_patches(enc, tr, spec, "voxel", exclude_center_atoms=False)
    excl = encode_patches(enc, tr, spec, "voxel", exclude_center_atoms=True)
    assert full.shape == excl.shape and not np.allclose(full, excl)


def test_unknown_identity_labels_are_filtered(tmp_path, rng):
    # a non-standard residue -> identity label -1 must be dropped, not crash the probe
    spec = PatchSpec(grid_voxels=16)
    for split in ("train", "val"):
        d = tmp_path / split; d.mkdir()
        for i in range(6):
            coords = (rng.random((6, 3)).astype("float32") - 0.5) * 3.0
            mask = np.zeros(6, dtype=bool); mask[0] = True
            res = "XYZ" if i == 0 else ("ALA" if i % 2 else "VAL")   # one -> -1
            patch = AtomPatch(coords, ["C", "N", "O", "S", "C", "N"],
                              {"rel_sasa": 0.5}, (f"p{i % 2}", "A", i, "", res), mask)
            with open(d / f"p{i}.pickle", "wb") as f:
                pickle.dump(patch, f)
    out = build_matrix(str(tmp_path / "train"), str(tmp_path / "val"), spec,
                       _ccfg(), [identity_task()], seeds=(0,))
    assert 0.0 <= out[("voxel", "identity")]["linear_mean"] <= 1.0


def test_capacity_sensitivity_reports_winner_per_config(dataset):
    spec, tr, va = dataset
    configs = {
        "narrow": {"voxel": _ccfg(proj_dim=8), "point": _ccfg(proj_dim=8)},
        "wide":   {"voxel": _ccfg(proj_dim=16), "point": _ccfg(proj_dim=16)},
    }
    report = capacity_sensitivity(tr, va, spec, burial_task(), configs)
    assert set(report) == {"narrow", "wide"}
    for label in report:
        assert report[label]["winner"] in ("voxel", "point")
        assert set(report[label]["acc"]) == {"voxel", "point"}

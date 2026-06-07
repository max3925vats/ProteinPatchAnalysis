import pickle

import numpy as np
import pytest

from protein_patch.spec import PatchSpec
from protein_patch.benchmark_spec import BenchmarkSpec
from protein_patch.patches import AtomPatch
from protein_patch.comparison import train_encoder_vae, embed, run_burial_cell


@pytest.fixture
def tiny_dataset(tmp_path, rng):
    spec = PatchSpec(grid_voxels=16)
    for split in ("train", "val"):
        d = tmp_path / split
        d.mkdir()
        for i in range(6):
            coords = (rng.random((5, 3)).astype("float32") - 0.5) * 3.0
            rel = 0.2 + 0.8 * (i / 5)            # spread across the retained range
            patch = AtomPatch(coords, ["C", "N", "O", "C", "C"],
                              {"rel_sasa": rel}, ("x", "A", i, "", "ALA"))
            with open(d / f"p{i}.pickle", "wb") as f:
                pickle.dump(patch, f)
    return spec, tmp_path


@pytest.mark.parametrize("kind", ["voxel", "point"])
def test_train_encoder_vae_smoke(tiny_dataset, kind):
    spec, root = tiny_dataset
    bspec = BenchmarkSpec(epochs=1, latent_dim=4, batch_size=2)
    hist, model = train_encoder_vae(kind, str(root / "train"), str(root / "val"),
                                    spec, bspec)
    assert len(hist["train_loss"]) == 1 and len(hist["val_loss"]) == 1
    assert all(np.isfinite(v) for v in hist["train_loss"])


@pytest.mark.parametrize("kind", ["voxel", "point"])
def test_run_burial_cell_returns_populated_dict(tiny_dataset, kind):
    spec, root = tiny_dataset
    bspec = BenchmarkSpec(epochs=1, latent_dim=4, batch_size=2)
    cell = run_burial_cell(kind, str(root / "train"), str(root / "val"), spec, bspec)
    assert cell["kind"] == kind
    assert 0.0 <= cell["accuracy"] <= 1.0
    assert cell["n_val"] == 6
    assert cell["n_classes"] == 3

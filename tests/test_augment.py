import numpy as np

from protein_patch.patches import AtomPatch
from protein_patch.augment import jitter_coords, drop_atoms, augment_atom_patch


def _patch(rng, n=8):
    coords = (rng.random((n, 3)).astype("float32") - 0.5) * 3.0
    return AtomPatch(coords, ["C"] * n, {"rel_sasa": 0.5}, ("x", "A", 1, "", "ALA"))


def test_jitter_changes_values_preserves_shape():
    rng = np.random.default_rng(0)
    coords = (rng.random((5, 3)).astype("float32"))
    out = jitter_coords(coords, sigma=0.5, rng=rng)
    assert out.shape == coords.shape and out.dtype == np.float32
    assert not np.allclose(out, coords)


def test_drop_atoms_reduces_but_keeps_minimum():
    rng = np.random.default_rng(0)
    coords = rng.random((10, 3)).astype("float32")
    elements = list("CNOSCNOSCN")
    c, e = drop_atoms(coords, elements, frac=0.5, rng=rng, min_keep=2)
    assert 2 <= len(e) < 10
    assert c.shape[0] == len(e)               # coords/elements stay aligned


def test_drop_atoms_floor_when_already_small():
    rng = np.random.default_rng(0)
    coords = rng.random((2, 3)).astype("float32")
    c, e = drop_atoms(coords, ["C", "N"], frac=0.9, rng=rng, min_keep=2)
    assert len(e) == 2                        # never drops below min_keep


def test_augment_preserves_metadata_and_is_deterministic():
    base = _patch(np.random.default_rng(123))
    a = augment_atom_patch(base, sigma=0.4, drop_frac=0.2,
                           rng=np.random.default_rng(7))
    b = augment_atom_patch(base, sigma=0.4, drop_frac=0.2,
                           rng=np.random.default_rng(7))
    assert isinstance(a, AtomPatch)
    assert a.attrs == base.attrs and a.provenance == base.provenance
    assert np.allclose(a.coords, b.coords) and a.elements == b.elements   # seeded

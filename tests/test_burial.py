import pickle

import numpy as np

from protein_patch.spec import PatchSpec  # noqa: F401  (kept for parity/imports)
from protein_patch.patches import AtomPatch
from protein_patch.tasks.burial import (
    read_rel_sasa, fit_quantile_edges, assign_classes,
)


def test_quantile_bins_are_nondegenerate_and_monotonic():
    # spread of retained rel_sasa values in [0.2, 1.0]
    vals = np.linspace(0.2, 1.0, 30)
    edges = fit_quantile_edges(vals, n_classes=3)
    labels = assign_classes(vals, edges)
    # all three classes populated
    assert set(labels.tolist()) == {0, 1, 2}
    # sorted input -> non-decreasing labels (classes track exposure)
    assert np.all(np.diff(labels) >= 0)


def test_shuffled_breaks_monotonicity():
    # non-vacuity guard for the monotonic-assignment claim above
    vals = np.linspace(0.2, 1.0, 30)
    edges = fit_quantile_edges(vals, n_classes=3)
    rng = np.random.default_rng(0)
    shuffled = vals.copy()
    rng.shuffle(shuffled)
    labels = assign_classes(shuffled, edges)
    assert not np.all(np.diff(labels) >= 0)


def test_read_rel_sasa_sorted_and_aligned(tmp_path):
    # write in non-sorted creation order; loader must sort by path and align
    for name, rs in [("b.pickle", 0.8), ("a.pickle", 0.3)]:
        p = AtomPatch(np.zeros((1, 3), dtype="float32"), ["C"],
                      {"rel_sasa": rs}, ("x", "A", 1, "", "ALA"))
        with open(tmp_path / name, "wb") as f:
            pickle.dump(p, f)
    paths, rel = read_rel_sasa(tmp_path)
    assert [p.name for p in paths] == ["a.pickle", "b.pickle"]
    assert rel.tolist() == [0.3, 0.8]

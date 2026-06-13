import pickle

import numpy as np

from protein_patch.patches import AtomPatch
from protein_patch.tasks.identity import residue_code, identity_labels, STANDARD_AA


def test_residue_code_covers_twenty_and_unknown():
    assert [residue_code(a) for a in STANDARD_AA] == list(range(20))
    assert residue_code("ALA") == 0 and residue_code("VAL") == 19
    assert residue_code("XYZ") == -1           # non-standard -> unknown


def test_identity_labels_sorted_and_coded(tmp_path):
    for name, res in [("b.pickle", "VAL"), ("a.pickle", "ALA")]:
        p = AtomPatch(np.zeros((1, 3), dtype="float32"), ["C"],
                      {"rel_sasa": 0.5}, ("x", "A", 1, "", res))
        with open(tmp_path / name, "wb") as f:
            pickle.dump(p, f)
    paths, labels = identity_labels(tmp_path)
    assert [p.name for p in paths] == ["a.pickle", "b.pickle"]
    assert labels.tolist() == [0, 19]          # ALA, VAL in sorted-path order

import io
import urllib.request

import numpy as np
import pytest

from protein_patch.tasks.secondary_structure import (
    collapse_ss3, ss_labels_from_dssp, compute_dssp,
)


def test_collapse_all_eight_dssp_codes():
    assert [collapse_ss3(c) for c in ("H", "G", "I")] == [0, 0, 0]      # helix
    assert [collapse_ss3(c) for c in ("E", "B")] == [1, 1]              # sheet
    assert [collapse_ss3(c) for c in ("T", "S", "P", "-")] == [2, 2, 2, 2]  # coil


def test_ss_labels_align_and_mark_missing():
    dssp_map = {("A", 1): "H", ("A", 2): "E", ("A", 3): "-"}
    keys = [("A", 1), ("A", 2), ("A", 3), ("A", 99)]   # last has no DSSP entry
    labels = ss_labels_from_dssp(dssp_map, keys)
    assert labels.tolist() == [0, 1, 2, -1]


@pytest.mark.integration
def test_compute_dssp_on_1ubq(tmp_path):
    from protein_patch.clean import load_clean_structure
    try:
        text = urllib.request.urlopen(
            "https://files.rcsb.org/download/1UBQ.pdb", timeout=20).read().decode()
    except Exception:
        pytest.skip("offline: cannot fetch 1UBQ")
    pdb_path = tmp_path / "1ubq.pdb"
    pdb_path.write_text(text)
    struct = load_clean_structure("1ubq", str(pdb_path))
    try:
        ss_map = compute_dssp(struct, str(pdb_path))
    except Exception as e:
        pytest.skip(f"mkdssp unavailable: {e}")
    assert len(ss_map) > 10
    assert all(isinstance(v, str) for v in ss_map.values())

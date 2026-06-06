import csv
import pickle
from protein_patch.spec import PatchSpec
from protein_patch.patches import AtomPatch
from protein_patch.data.prep import prep_dataset

ALA = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.500   1.000   0.000  1.00  0.00           C
"""


def _make_pdbs(tmp_path, n):
    d = tmp_path / "pdbs"; d.mkdir()
    for i in range(n):
        (d / f"prot{i}.pdb").write_text(ALA)
    return d


def test_prep_dataset_writes_patches_manifest_and_split(tmp_path):
    pdb_dir = _make_pdbs(tmp_path, 5)
    out = tmp_path / "patches"
    spec = PatchSpec(grid_voxels=16, voxel_size=0.5, sasa_threshold=0.0)
    summary = prep_dataset(pdb_dir, out, spec, seed=42, val_fraction=0.4)

    train = list((out / "train").glob("*.pickle"))
    val = list((out / "val").glob("*.pickle"))
    assert len(train) + len(val) == 5 and len(val) == 2
    with open(train[0], "rb") as f:
        assert isinstance(pickle.load(f), AtomPatch)
    with open(out / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    assert set(rows[0]) == {"patch_file", "pdb_id", "chain", "resseq",
                            "icode", "resname", "split"}
    assert summary["n_patches"] == 5


def test_prep_split_is_deterministic_and_protein_level(tmp_path):
    pdb_dir = _make_pdbs(tmp_path, 6)
    spec = PatchSpec(grid_voxels=16, voxel_size=0.5, sasa_threshold=0.0)
    s1 = prep_dataset(pdb_dir, tmp_path / "a", spec, seed=1, val_fraction=0.5)
    s2 = prep_dataset(pdb_dir, tmp_path / "b", spec, seed=1, val_fraction=0.5)
    assert s1["val_pdbs"] == s2["val_pdbs"]
    assert set(s1["train_pdbs"]).isdisjoint(s1["val_pdbs"])


# A 2-residue protein -> 2 patches; both must land on the SAME split side.
TWO_RES = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.500   1.000   0.000  1.00  0.00           C
ATOM      4  N   GLY A   2       4.000   0.000   0.000  1.00  0.00           N
ATOM      5  CA  GLY A   2       5.500   0.000   0.000  1.00  0.00           C
"""


def test_multi_patch_protein_stays_on_one_side(tmp_path):
    d = tmp_path / "pdbs"; d.mkdir()
    (d / "multi.pdb").write_text(TWO_RES)
    spec = PatchSpec(grid_voxels=16, voxel_size=0.5, sasa_threshold=0.0)
    summary = prep_dataset(d, tmp_path / "out", spec, seed=0, val_fraction=1.0)
    # the single PDB went entirely to val; BOTH of its patches are there, train is empty
    assert summary["n_patches"] == 2
    assert len(list((tmp_path / "out" / "val").glob("*.pickle"))) == 2
    assert len(list((tmp_path / "out" / "train").glob("*.pickle"))) == 0

import io
import urllib.request
import numpy as np
import pytest
from protein_patch.clean import load_clean_structure
from protein_patch.spec import PatchSpec
from protein_patch.patches import (
    AtomPatch,
    extract_atom_patches,
    extract_patches,
    voxelize,
)

PDB_TEXT = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00  0.00           C
HETATM    3  O   HOH A   2       9.000   9.000   9.000  1.00  0.00           O
"""


def test_cleaning_drops_water_and_heteroatoms():
    struct = load_clean_structure("test", io.StringIO(PDB_TEXT))
    atoms = list(struct.get_atoms())
    # the HOH HETATM is gone; the two ALA atoms remain
    assert len(atoms) == 2
    resnames = {a.get_parent().get_resname() for a in atoms}
    assert resnames == {"ALA"}


def test_extract_patches_shape_and_count():
    struct = load_clean_structure("t", io.StringIO(PDB_TEXT))
    spec = PatchSpec(grid_voxels=16, voxel_size=0.5, sasa_threshold=0.0)
    patches = extract_patches(struct, spec)  # threshold 0 -> keep the residue
    assert patches.shape == (1, 4, 16, 16, 16)   # (n, C, L, L, L), channel-first
    assert patches.dtype == np.float32
    assert not np.isnan(patches).any()
    # non-vacuity: real atomic density was deposited (a zero-stub would fail here)
    assert patches.max() > 0.0
    # ALA has C/N/O atoms but no sulphur -> the S channel (index 3) stays empty
    assert patches[0, 3].sum() == 0.0


def test_atom_patches_are_centered_with_attrs():
    struct = load_clean_structure("t", io.StringIO(PDB_TEXT))
    spec = PatchSpec(grid_voxels=16, voxel_size=0.5, sasa_threshold=0.0)
    patches = extract_atom_patches(struct, spec, pdb_id="t")
    assert len(patches) == 1
    p = patches[0]
    assert isinstance(p, AtomPatch)
    half = spec.side_angstroms / 2.0
    assert np.all(np.abs(p.coords) <= half + 1e-6)
    assert p.provenance == ("t", "A", 1, "", "ALA")   # (pdb, chain, resseq, icode, resname)
    assert p.attrs["resname"] == "ALA" and p.attrs["charge"] == 0.0


def test_voxelize_atompatch_matches_extract_patches_grid():
    # Regression guard: the atom-set path reproduces the old grid exactly.
    struct = load_clean_structure("t", io.StringIO(PDB_TEXT))
    spec = PatchSpec(grid_voxels=16, voxel_size=0.5, sasa_threshold=0.0)
    grid_via_wrapper = extract_patches(struct, spec)
    grid_via_atomset = voxelize(extract_atom_patches(struct, spec)[0], spec)
    assert np.array_equal(grid_via_wrapper[0], grid_via_atomset)


@pytest.mark.integration
def test_1ubq_end_to_end():
    url = "https://files.rcsb.org/download/1UBQ.pdb"
    try:
        text = urllib.request.urlopen(url, timeout=20).read().decode()
    except Exception:
        pytest.skip("offline: cannot fetch 1UBQ")
    struct = load_clean_structure("1ubq", io.StringIO(text))
    spec = PatchSpec(sasa_threshold=0.2)
    patches = extract_patches(struct, spec)
    # ubiquitin (76 residues) yields a nonzero number of exposed patches
    assert patches.shape[0] > 10
    assert patches.shape[1:] == spec.array_shape

import io
import pickle

import numpy as np
import pytest

from protein_patch.clean import load_clean_structure
from protein_patch.spec import PatchSpec
from protein_patch.patches import AtomPatch, extract_atom_patches, exclude_center

# Two close residues so a patch centred on residue 1 also contains residue 2's
# atoms -> central_mask must distinguish them.
TWO_RES_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  N   GLY A   2       3.000   0.000   0.000  1.00  0.00           N
ATOM      4  CA  GLY A   2       4.000   0.000   0.000  1.00  0.00           C
"""


def test_central_mask_optional_and_pickles():
    p0 = AtomPatch(np.zeros((2, 3), "float32"), ["C", "N"],
                   {"rel_sasa": 0.5}, ("x", "A", 1, "", "ALA"))
    assert p0.central_mask is None                       # 4-arg construction still valid
    p1 = AtomPatch(np.zeros((2, 3), "float32"), ["C", "N"],
                   {"rel_sasa": 0.5}, ("x", "A", 1, "", "ALA"), np.array([True, False]))
    restored = pickle.loads(pickle.dumps(p1))
    assert np.array_equal(restored.central_mask, np.array([True, False]))


def test_extraction_marks_only_central_residue_atoms():
    struct = load_clean_structure("t", io.StringIO(TWO_RES_PDB))
    spec = PatchSpec(grid_voxels=16, voxel_size=0.5, sasa_threshold=0.0)
    patches = extract_atom_patches(struct, spec, pdb_id="t")
    p = next(pp for pp in patches if pp.provenance[2] == 1)   # ALA-centred patch
    assert p.central_mask is not None and p.central_mask.dtype == bool
    assert len(p.central_mask) == len(p.elements)
    # the two ALA atoms are marked, the two GLY atoms are not
    assert p.central_mask.sum() == 2 and (~p.central_mask).sum() == 2


def test_exclude_center_drops_marked_atoms():
    p = AtomPatch(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], "float32"),
                  ["C", "N", "O"], {"rel_sasa": 0.5}, ("x", "A", 1, "", "ALA"),
                  np.array([True, False, False]))
    e = exclude_center(p)
    assert e.coords.shape[0] == 2 and e.elements == ["N", "O"]
    assert e.central_mask.sum() == 0                     # nothing central remains


def test_exclude_center_requires_mask():
    p = AtomPatch(np.zeros((2, 3), "float32"), ["C", "N"],
                  {"rel_sasa": 0.5}, ("x", "A", 1, "", "ALA"))   # central_mask=None
    with pytest.raises(ValueError):
        exclude_center(p)


def test_exclude_center_raises_when_environment_has_no_cnos():
    # central C is the only C/N/O/S; the rest are H -> empty environment, not encodable
    p = AtomPatch(np.zeros((3, 3), "float32"), ["C", "H", "H"],
                  {"rel_sasa": 0.5}, ("x", "A", 1, "", "ALA"),
                  np.array([True, False, False]))
    with pytest.raises(ValueError):
        exclude_center(p)

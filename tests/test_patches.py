import io
import numpy as np
from protein_patch.clean import load_clean_structure
from protein_patch.spec import PatchSpec
from protein_patch.patches import extract_patches

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

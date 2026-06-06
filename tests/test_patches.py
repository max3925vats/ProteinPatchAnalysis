import io
from protein_patch.clean import load_clean_structure

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

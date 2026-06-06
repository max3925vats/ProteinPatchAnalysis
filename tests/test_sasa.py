import io
from protein_patch.clean import load_clean_structure
from protein_patch.sasa import relative_sasa

PDB_TEXT = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.500   1.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.500   1.000   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.500   1.500   0.500  1.00  0.00           C
"""


def test_relative_sasa_returns_fraction_per_residue():
    struct = load_clean_structure("t", io.StringIO(PDB_TEXT))
    rel = relative_sasa(struct)
    # one residue, exposure expressed as a fraction in [0, 1]
    assert len(rel) == 1
    (key, value), = rel.items()
    assert key == (0, "A", 1, "ALA")   # (model_id, chain, resseq, resname)
    assert 0.0 <= value <= 1.0

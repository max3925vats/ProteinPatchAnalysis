import math
from protein_patch.data.attributes import residue_attributes, KYTE_DOOLITTLE


def test_known_residue_attributes():
    a = residue_attributes("ALA", 0.5)
    assert a == {"resname": "ALA", "hydropathy": 1.8, "charge": 0.0, "rel_sasa": 0.5}


def test_charged_residues():
    assert residue_attributes("ASP", 0.3)["charge"] == -1.0
    assert residue_attributes("GLU", 0.3)["charge"] == -1.0
    assert residue_attributes("LYS", 0.3)["charge"] == 1.0
    assert residue_attributes("ARG", 0.3)["charge"] == 1.0


def test_unknown_residue_hydropathy_is_nan():
    assert math.isnan(residue_attributes("XYZ", 0.1)["hydropathy"])
    assert len(KYTE_DOOLITTLE) == 20

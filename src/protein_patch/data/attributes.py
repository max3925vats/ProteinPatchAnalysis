import math
from typing import TypedDict


class ResidueAttributes(TypedDict):
    """Chemical descriptors of a residue used to colour latent-space maps."""
    resname: str
    hydropathy: float
    charge: float
    rel_sasa: float


# Kyte & Doolittle (1982) hydropathy index.
KYTE_DOOLITTLE: dict[str, float] = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
    "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
    "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6,
    "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
}

# Formal side-chain charge at pH ~7 (His treated as neutral; pKa ~6).
_CHARGE: dict[str, float] = {"ASP": -1.0, "GLU": -1.0, "LYS": 1.0, "ARG": 1.0}


def residue_attributes(resname: str, rel_sasa: float) -> ResidueAttributes:
    """Chemical attributes of a residue used to colour latent-space maps."""
    name = resname.strip().upper()
    return {
        "resname": name,
        "hydropathy": KYTE_DOOLITTLE.get(name, math.nan),
        "charge": _CHARGE.get(name, 0.0),
        "rel_sasa": float(rel_sasa),
    }

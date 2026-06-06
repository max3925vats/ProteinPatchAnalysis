from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Structure import Structure
from Bio.PDB.Polypeptide import is_aa

# Tien et al. (2013) theoretical max ASA (A^2) per residue.
MAX_ASA: dict[str, float] = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLU": 223.0, "GLN": 225.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}


def relative_sasa(structure: Structure) -> dict[tuple[str, int, str], float]:
    """Per-residue relative solvent accessibility in [0, 1].

    Computes absolute SASA with Biopython's Shrake-Rupley implementation
    (no GROMACS), then divides each residue's area by its Tien max ASA.
    Key = (chain_id, resseq, resname).
    """
    ShrakeRupley().compute(structure, level="R")
    out: dict[tuple[str, int, str], float] = {}
    for model in structure:
        for chain in model:
            for res in chain:
                name = res.get_resname().strip()
                if not is_aa(res, standard=True) or name not in MAX_ASA:
                    continue
                key = (chain.id, res.id[1], name)
                # clamp to [0, 1]; SASA is non-negative but guard defensively
                out[key] = max(0.0, min(res.sasa / MAX_ASA[name], 1.0))
    return out

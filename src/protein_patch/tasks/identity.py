import numpy as np

from .io import read_patch_meta

# Standard 20 amino acids -> 0..19 (fixed order = stable class ids).
STANDARD_AA: tuple[str, ...] = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)
_AA_IDX = {a: i for i, a in enumerate(STANDARD_AA)}


def residue_code(resname: str) -> int:
    """Class id in [0, 20) for a standard residue; -1 for anything else."""
    return _AA_IDX.get(resname.strip().upper(), -1)


def identity_labels(root) -> tuple[list, np.ndarray]:
    """(paths, central-residue identity codes) in sorted-path order."""
    paths, prov = read_patch_meta(root)
    labels = np.array([residue_code(p[4]) for p in prov], dtype=np.int64)
    return paths, labels

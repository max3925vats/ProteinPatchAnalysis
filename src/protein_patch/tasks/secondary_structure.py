import numpy as np

# 3-state collapse of DSSP's 8 states. Helix = {H,G,I}, sheet = {E,B}, the rest
# (T, S, P, '-', blanks) = coil.
HELIX = frozenset({"H", "G", "I"})
SHEET = frozenset({"E", "B"})
HELIX_C, SHEET_C, COIL_C = 0, 1, 2


def collapse_ss3(dssp_code: str) -> int:
    """Map an 8-state DSSP code to {helix=0, sheet=1, coil=2}."""
    c = dssp_code.strip().upper()
    if c in HELIX:
        return HELIX_C
    if c in SHEET:
        return SHEET_C
    return COIL_C


def ss_labels_from_dssp(dssp_map: dict, keys: list) -> np.ndarray:
    """3-state labels for each (chain, resseq) key; -1 where DSSP has no entry.

    `dssp_map`: {(chain, resseq) -> 8-state code}. `keys`: per-patch (chain,
    resseq) in the same order as the embeddings. Missing -> -1 (filter before
    probing).
    """
    out = [collapse_ss3(dssp_map[k]) if k in dssp_map else -1 for k in keys]
    return np.array(out, dtype=np.int64)


def compute_dssp(structure, pdb_path: str) -> dict:
    """Run DSSP and return {(chain, resseq) -> 8-state code}.

    Integration-only: requires the `mkdssp` binary. Not used by unit tests.
    """
    from Bio.PDB.DSSP import DSSP

    dssp = DSSP(structure[0], pdb_path)
    out: dict = {}
    for chain, res_id in dssp.keys():
        out[(chain, res_id[1])] = dssp[(chain, res_id)][2]   # SS code field
    return out

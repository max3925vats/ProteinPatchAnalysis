import pickle
from pathlib import Path

from ..patches import AtomPatch


def read_patch_meta(root: str | Path) -> tuple[list[Path], list[tuple]]:
    """Per-patch provenance in sorted-path order (aligns with the datasets).

    Returns (paths, provenances) where provenance = (pdb_id, chain, resseq,
    icode, resname). Probe-label tasks read what they need from this.
    """
    paths = sorted(Path(root).glob("*.pickle"))
    prov = []
    for p in paths:
        with open(p, "rb") as f:
            patch: AtomPatch = pickle.load(f)
        prov.append(patch.provenance)
    return paths, prov

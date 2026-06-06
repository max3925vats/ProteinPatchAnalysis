import csv
import logging
import pickle
import random
from pathlib import Path

from ..patches import extract_atom_patches
from ..spec import PatchSpec
from ..clean import load_clean_structure

logger = logging.getLogger(__name__)


def process_one(pdb_path: str | Path, spec: PatchSpec) -> list:
    """Clean one PDB and extract its AtomPatches (pure; no disk writes)."""
    pdb_id = Path(pdb_path).stem
    structure = load_clean_structure(pdb_id, str(pdb_path))
    return extract_atom_patches(structure, spec, pdb_id=pdb_id)


def prep_dataset(pdb_dir: str | Path, out_dir: str | Path, spec: PatchSpec,
                 seed: int = 42, val_fraction: float = 0.2) -> dict:
    """Build train/val AtomPatch pickles + manifest from a folder of PDBs.

    Split is seeded and protein-level (all patches of one PDB share a split).
    Per-protein failures are logged and skipped.

    Args:
        pdb_dir: Directory containing *.pdb files.
        out_dir: Output directory; train/ and val/ subdirs are created.
        spec: Geometry and SASA parameters for patch extraction.
        seed: Random seed for the train/val split shuffle.
        val_fraction: Fraction of PDBs to assign to the val split.

    Returns:
        Dict with keys: n_patches, val_pdbs (sorted list), train_pdbs (sorted list).
    """
    pdb_dir, out_dir = Path(pdb_dir), Path(out_dir)
    pdbs = sorted(pdb_dir.glob("*.pdb"))
    shuffled = pdbs[:]
    random.Random(seed).shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    val_pdbs = {p.stem for p in shuffled[:n_val]}
    for split in ("train", "val"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    train_pdbs: list[str] = []
    for pdb in pdbs:
        split = "val" if pdb.stem in val_pdbs else "train"
        if split == "train":
            train_pdbs.append(pdb.stem)
        try:
            patches = process_one(pdb, spec)
        except Exception as e:
            logger.warning("skipping %s: %s", pdb.name, e)
            continue
        for p in patches:
            pid, chain, resseq, resname = p.provenance
            fname = f"{pid}_{chain}_{resseq}.pickle"
            with open(out_dir / split / fname, "wb") as f:
                pickle.dump(p, f)
            rows.append({"patch_file": fname, "pdb_id": pid, "chain": chain,
                         "resseq": resseq, "resname": resname, "split": split})

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patch_file", "pdb_id", "chain",
                                          "resseq", "resname", "split"])
        w.writeheader()
        w.writerows(rows)

    return {"n_patches": len(rows), "val_pdbs": sorted(val_pdbs),
            "train_pdbs": sorted(train_pdbs)}

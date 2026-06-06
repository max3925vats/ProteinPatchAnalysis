from typing import IO, Union

from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure

_WATER = {"HOH", "WAT", "TIP", "TIP3", "SOL"}


def load_clean_structure(name: str, handle: Union[IO, str]) -> Structure:
    """Parse a PDB and strip waters + heteroatoms in-process.

    Replaces the old clean.sh (pdb4amber + sed). A residue is dropped if
    its hetfield flag is set (HETATM) or it is a known water. Hydrogen
    handling is irrelevant here: only C/N/O/S are voxelized downstream.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(name, handle)
    for model in structure:
        for chain in model:
            drop = [
                res.id for res in chain
                if res.id[0].strip() != "" or res.get_resname().strip() in _WATER
            ]
            for rid in drop:
                chain.detach_child(rid)
    return structure

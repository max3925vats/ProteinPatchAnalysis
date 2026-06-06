from dataclasses import dataclass

import numpy as np
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from .spec import PatchSpec
from .sasa import relative_sasa
from .voxelize import voxelize_atoms, center_of_geometry
from .data.attributes import residue_attributes


def _residue_atoms(residue: Residue) -> tuple[np.ndarray, list[str]]:
    """Extract (coords, elements) arrays from a Biopython Residue."""
    coords, elements = [], []
    for atom in residue:
        coords.append(atom.get_coord())
        elements.append(atom.element)
    return np.array(coords, dtype=float), elements


@dataclass
class AtomPatch:
    """Atoms within the cube around one exposed residue, centered on its COG."""
    coords: np.ndarray          # (n_atoms, 3), centered (origin = residue COG)
    elements: list[str]         # element symbol per atom
    attrs: dict                 # residue_attributes(...) output
    provenance: tuple           # (pdb_id, chain_id, resseq, resname)


def extract_atom_patches(structure: Structure, spec: PatchSpec,
                         pdb_id: str = "") -> list[AtomPatch]:
    """One AtomPatch per solvent-exposed residue (atoms within the cube)."""
    exposure = relative_sasa(structure)
    half = spec.side_angstroms / 2.0
    patches: list[AtomPatch] = []
    for model in structure:
        for chain in model:
            for res in chain:
                name = res.get_resname().strip()
                key = (chain.id, res.id[1], name)
                if exposure.get(key, 0.0) < spec.sasa_threshold:
                    continue
                coords, _ = _residue_atoms(res)
                if len(coords) == 0:
                    continue
                center = center_of_geometry(coords)
                sel_c, sel_e = [], []
                for atom in model.get_atoms():
                    c = atom.get_coord()
                    if np.all(np.abs(c - center) <= half):
                        sel_c.append(c)
                        sel_e.append(atom.element)
                centered = np.asarray(sel_c, dtype=np.float64) - center
                attrs = residue_attributes(name, exposure.get(key, 0.0))
                patches.append(AtomPatch(centered, sel_e, attrs,
                                         (pdb_id, chain.id, res.id[1], name)))
    return patches


def voxelize(patch: AtomPatch, spec: PatchSpec) -> np.ndarray:
    """Voxelize a centered AtomPatch into a (C, L, L, L) float32 grid."""
    half = spec.side_angstroms / 2.0
    grid_min = np.full(3, -half, dtype=np.float64)
    return voxelize_atoms(patch.coords, patch.elements, grid_min, spec)


def extract_patches(structure: Structure, spec: PatchSpec) -> np.ndarray:
    """Backward-compatible voxel-grid output: (n, C, L, L, L), channel-first."""
    patches = extract_atom_patches(structure, spec)
    if not patches:
        return np.empty((0, *spec.array_shape), dtype=np.float32)
    return np.stack([voxelize(p, spec) for p in patches]).astype(np.float32)

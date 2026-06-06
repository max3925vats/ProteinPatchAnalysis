from dataclasses import dataclass

import numpy as np
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from .spec import PatchSpec
from .sasa import relative_sasa
from .voxelize import voxelize_atoms, center_of_geometry
from .data.attributes import ResidueAttributes, residue_attributes


def _residue_cog(residue: Residue) -> np.ndarray:
    """Center of geometry of a residue's own atoms (float64)."""
    coords = [atom.get_coord() for atom in residue]
    return center_of_geometry(np.asarray(coords, dtype=np.float64))


@dataclass(frozen=True)
class AtomPatch:
    """Atoms within the cube around one exposed residue, centered on its COG."""
    coords: np.ndarray                      # (n_atoms, 3), centered (origin = COG)
    elements: list[str]                     # element symbol per atom
    attrs: ResidueAttributes                # central-residue chemical attributes
    provenance: tuple[str, str, int, str]   # (pdb_id, chain_id, resseq, resname)


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
                if len(list(res.get_atoms())) == 0:
                    continue
                center = _residue_cog(res)
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

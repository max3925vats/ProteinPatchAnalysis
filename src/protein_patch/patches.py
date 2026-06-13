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
    coords: np.ndarray                           # (n_atoms, 3), centered (origin = COG)
    elements: list[str]                          # element symbol per atom
    attrs: ResidueAttributes                     # central-residue chemical attributes
    provenance: tuple[str, str, int, str, str]   # (pdb_id, chain, resseq, icode, resname)
    central_mask: np.ndarray | None = None       # bool per atom: belongs to the central residue


def exclude_center(patch: AtomPatch) -> AtomPatch:
    """Return the patch with the central residue's own atoms removed.

    Used to build the environment-only view for the masked-identity probe, so a
    latent predicts the central residue from its surroundings (no trivial readout
    of the central sidechain). Requires `central_mask` (re-prep old patches).
    """
    mask = getattr(patch, "central_mask", None)
    if mask is None:
        raise ValueError(
            "central_mask is required to exclude the central residue; re-prep the patch")
    keep = ~np.asarray(mask, dtype=bool)
    coords = np.asarray(patch.coords)[keep]
    elements = [e for e, k in zip(patch.elements, keep) if k]
    if not any(e.strip().upper() in ("C", "N", "O", "S") for e in elements):
        # an environment with no C/N/O/S is not encodable (the voxel/point views
        # would be empty). Fail explicitly rather than crash downstream.
        raise ValueError(
            f"patch {patch.provenance} has no C/N/O/S atoms left after excluding "
            "the central residue")
    return AtomPatch(coords, elements, patch.attrs, patch.provenance,
                     np.zeros(coords.shape[0], dtype=bool))


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
                key = (model.id, chain.id, res.id[1], name)
                if exposure.get(key, 0.0) < spec.sasa_threshold:
                    continue
                if len(list(res.get_atoms())) == 0:
                    continue
                center = _residue_cog(res)
                sel_c, sel_e, sel_m = [], [], []
                for atom in model.get_atoms():
                    c = atom.get_coord()
                    if np.all(np.abs(c - center) <= half):
                        sel_c.append(c)
                        sel_e.append(atom.element)
                        sel_m.append(atom.get_parent() is res)   # central-residue atom?
                centered = np.asarray(sel_c, dtype=np.float64) - center
                attrs = residue_attributes(name, exposure.get(key, 0.0))
                icode = res.id[2].strip()   # insertion code ("" if none)
                patches.append(AtomPatch(centered, sel_e, attrs,
                                         (pdb_id, chain.id, res.id[1], icode, name),
                                         np.asarray(sel_m, dtype=bool)))
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

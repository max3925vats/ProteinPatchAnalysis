import numpy as np
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from .spec import PatchSpec
from .sasa import relative_sasa
from .voxelize import voxelize_atoms, center_of_geometry


def _residue_atoms(residue: Residue) -> tuple[np.ndarray, list[str]]:
    """Extract (coords, elements) arrays from a Biopython Residue."""
    coords, elements = [], []
    for atom in residue:
        coords.append(atom.get_coord())
        elements.append(atom.element)
    return np.array(coords, dtype=float), elements


def extract_patches(structure: Structure, spec: PatchSpec) -> np.ndarray:
    """Cut one cubic patch per solvent-exposed residue.

    For each residue with relative SASA >= spec.sasa_threshold, the grid
    is centred on that residue's center-of-geometry, so the result is
    translation-invariant and works for ANY protein (no hardcoded bounds).
    Returns (n_patches, C, L, L, L), channel-first.
    """
    exposure = relative_sasa(structure)
    half = spec.side_angstroms / 2.0
    patches = []

    for model in structure:
        for chain in model:
            for res in chain:
                key = (chain.id, res.id[1], res.get_resname().strip())
                # Keep residue if its exposure meets the threshold.
                # exposure.get(key, 0.0): residues not in the SASA dict
                # (non-standard AAs) default to 0.0, kept only at threshold==0.
                if exposure.get(key, 0.0) < spec.sasa_threshold:
                    continue

                coords, elements = _residue_atoms(res)
                if len(coords) == 0:
                    continue

                center = center_of_geometry(coords)
                grid_min = center - half

                # Collect atoms from THIS model that fall inside the cube.
                # Scoping to `model` (not `structure`) avoids pulling atoms
                # from every model of a multi-model NMR ensemble into each patch.
                all_coords, all_el = [], []
                for atom in model.get_atoms():
                    c = atom.get_coord()
                    if np.all(np.abs(c - center) <= half):
                        all_coords.append(c)
                        all_el.append(atom.element)

                grid = voxelize_atoms(
                    np.array(all_coords, dtype=float), all_el, grid_min, spec
                )
                patches.append(grid)

    if not patches:
        return np.empty((0, *spec.array_shape), dtype=np.float32)
    return np.stack(patches).astype(np.float32)

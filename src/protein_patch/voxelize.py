import numpy as np

from .atom_types import atom_channel, ELEMENTS  # noqa: F401 (re-exported for convenience)
from .spec import PatchSpec


def gaussian_density(distance: np.ndarray | float, std: float) -> np.ndarray | float:
    """exp(-d^2 / (2 std^2)). Accepts scalars or arrays (vectorized)."""
    return np.exp(-(np.asarray(distance) ** 2) / (2.0 * std**2))


def center_of_geometry(coords: np.ndarray) -> np.ndarray:
    """Mean xyz of an (n_atoms, 3) coordinate array."""
    return np.asarray(coords, dtype=float).mean(axis=0)


def voxelize_atoms(
    coords: np.ndarray,
    elements: list[str],
    grid_min: np.ndarray,
    spec: PatchSpec,
) -> np.ndarray:
    """Accumulate Gaussian atomic density onto a (C, L, L, L) grid.

    Vectorized over voxels: for each atom we add its Gaussian contribution
    to the whole channel grid at once, instead of the old triple-nested
    per-voxel Python loop. L = spec.grid_voxels.

    Args:
        coords: (n_atoms, 3) array of xyz positions in angstroms.
        elements: element symbol per atom (e.g. "C", "N", "O", "S").
        grid_min: (3,) lower-corner xyz of the voxel grid in angstroms.
        spec: geometry contract (grid size, voxel size, Gaussian std).

    Returns:
        Float32 array of shape (n_channels, L, L, L).
    """
    L = spec.grid_voxels
    grid = np.zeros(spec.array_shape, dtype=np.float32)

    # Voxel-center coordinates along each axis (shared by x/y/z).
    # Shape: (3, L) — one row per spatial axis.
    axis = grid_min[:, None] + (np.arange(L) + 0.5) * spec.voxel_size
    gx, gy, gz = np.meshgrid(axis[0], axis[1], axis[2], indexing="ij")

    for xyz, el in zip(coords, elements):
        ch = atom_channel(el)
        if ch.sum() == 0:  # skip H and exotic atoms
            continue
        d2 = (gx - xyz[0]) ** 2 + (gy - xyz[1]) ** 2 + (gz - xyz[2]) ** 2
        density = np.exp(-d2 / (2.0 * spec.gaussian_std**2)).astype(np.float32)
        grid[int(np.argmax(ch))] += density

    return grid

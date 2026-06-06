import numpy as np


def gaussian_density(distance: np.ndarray | float, std: float) -> np.ndarray | float:
    """exp(-d^2 / (2 std^2)). Accepts scalars or arrays (vectorized)."""
    return np.exp(-(np.asarray(distance) ** 2) / (2.0 * std**2))


def center_of_geometry(coords: np.ndarray) -> np.ndarray:
    """Mean xyz of an (n_atoms, 3) coordinate array."""
    return np.asarray(coords, dtype=float).mean(axis=0)

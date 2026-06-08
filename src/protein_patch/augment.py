"""Substrate-level augmentations for contrastive positive pairs.

Augment the AtomPatch (the representation-agnostic atom set), THEN derive the
voxel/point views — so both representations see the same augmented atoms (the
harness fairness rule). v0.5 uses jitter + mild dropout only; rotation is the
later orientation layer.
"""
import numpy as np

from .patches import AtomPatch


def jitter_coords(coords: np.ndarray, sigma: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Add isotropic Gaussian noise (sigma in angstroms) to atom positions."""
    coords = np.asarray(coords, dtype=np.float32)
    noise = rng.normal(0.0, sigma, size=coords.shape).astype(np.float32)
    return (coords + noise).astype(np.float32)


def drop_atoms(coords: np.ndarray, elements: list[str], frac: float,
               rng: np.random.Generator, min_keep: int = 2):
    """Randomly keep a (1 - frac) subset of atoms, never fewer than min_keep.

    Returns (coords, elements) kept in their original relative order so the two
    arrays stay aligned. A view always has >= min_keep atoms to encode.
    """
    n = len(elements)
    if n <= min_keep:
        return np.asarray(coords, dtype=np.float32), list(elements)
    keep_n = max(min_keep, int(round(n * (1.0 - frac))))
    keep_n = min(keep_n, n)
    idx = np.sort(rng.choice(n, size=keep_n, replace=False))
    coords = np.asarray(coords, dtype=np.float32)[idx]
    return coords, [elements[i] for i in idx]


def augment_atom_patch(patch: AtomPatch, sigma: float, drop_frac: float,
                       rng: np.random.Generator) -> AtomPatch:
    """One augmented view: dropout then jitter; attrs/provenance preserved."""
    coords, elements = drop_atoms(patch.coords, patch.elements, drop_frac, rng)
    coords = jitter_coords(coords, sigma, rng)
    return AtomPatch(coords, elements, patch.attrs, patch.provenance)

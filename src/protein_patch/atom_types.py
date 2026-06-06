import numpy as np

ELEMENTS: tuple[str, ...] = ("C", "N", "O", "S")
_INDEX: dict[str, int] = {el: i for i, el in enumerate(ELEMENTS)}


def atom_channel(element: str) -> np.ndarray:
    """One-hot vector over (C, N, O, S). Unknown elements -> all zeros.

    Takes the element symbol (e.g. from `atom.element` in Biopython),
    not the atom name, so "CA" (calcium) is no longer mis-typed as carbon.
    """
    vec = np.zeros(len(ELEMENTS), dtype=np.float32)
    idx = _INDEX.get(element.strip().upper())
    if idx is not None:
        vec[idx] = 1.0
    return vec

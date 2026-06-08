import pickle
from pathlib import Path

import numpy as np

from ..patches import AtomPatch


def read_rel_sasa(root: str | Path) -> tuple[list[Path], np.ndarray]:
    """Per-patch central-residue relative SASA, in sorted-path order.

    Order matches PatchDataset / PointPatchDataset (both sort *.pickle paths),
    so the returned array aligns positionally with embeddings from those
    datasets — i.e. labels[i] belongs to embedding[i].
    """
    paths = sorted(Path(root).glob("*.pickle"))
    rel = []
    for p in paths:
        with open(p, "rb") as f:
            patch: AtomPatch = pickle.load(f)
        rel.append(float(patch.attrs["rel_sasa"]))
    return paths, np.asarray(rel, dtype=np.float64)


def fit_quantile_edges(rel_sasas, n_classes: int = 3) -> np.ndarray:
    """Interior quantile cut points fit on the (train) rel_sasa distribution.

    Returns the (n_classes - 1) edges. Quantiles (not fixed thresholds) because
    every retained patch already has rel_sasa >= sasa_threshold, so an absolute
    buried/exposed cut would degenerate; quantiles split the *retained*
    population into roughly balanced classes.

    Assumes a continuous (tie-light) rel_sasa distribution. With many identical
    values (or n_classes > n unique values) quantile edges can collapse and empty
    a class — fine for real continuous SASA, revisit if labels look degenerate.
    """
    rel = np.asarray(rel_sasas, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_classes + 1)[1:-1]   # interior quantiles
    return np.quantile(rel, qs)


def assign_classes(rel_sasas, edges: np.ndarray) -> np.ndarray:
    """Map rel_sasa values to integer classes [0, n_classes) via the edges."""
    rel = np.asarray(rel_sasas, dtype=np.float64)
    return np.digitize(rel, np.asarray(edges, dtype=np.float64))

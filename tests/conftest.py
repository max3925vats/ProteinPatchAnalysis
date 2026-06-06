import numpy as np
import pytest

from protein_patch.spec import PatchSpec


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def fake_patch(rng):
    """A single channel-first patch: (n_channels, L, L, L) in [0,1].

    Dimensions are derived from PatchSpec so the fixture can never drift
    from the canonical grid size.
    """
    spec = PatchSpec()
    L = spec.grid_voxels
    return rng.random((spec.n_channels, L, L, L)).astype(np.float32)

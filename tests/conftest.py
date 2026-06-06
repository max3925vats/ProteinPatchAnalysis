import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def fake_patch(rng):
    """A single channel-first patch: (C=4, L=50, L, L) in [0,1]."""
    return rng.random((4, 50, 50, 50)).astype(np.float32)

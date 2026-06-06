import dataclasses

import pytest

from protein_patch.spec import PatchSpec


def test_default_spec_shape():
    s = PatchSpec()
    assert s.n_channels == 4
    assert s.grid_voxels == 64
    # channel-first array shape used everywhere on disk and in torch
    assert s.array_shape == (4, 64, 64, 64)
    # 64 * 0.375 = 24 A physical cube
    assert s.side_angstroms == 24.0
    # remaining defaults are part of the contract too
    assert s.voxel_size == 0.375
    assert s.gaussian_std == 1.0
    assert s.sasa_threshold == 0.2


def test_spec_is_frozen():
    s = PatchSpec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.grid_voxels = 32

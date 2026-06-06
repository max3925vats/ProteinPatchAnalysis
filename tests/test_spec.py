from protein_patch.spec import PatchSpec


def test_default_spec_shape():
    s = PatchSpec()
    assert s.n_channels == 4
    assert s.grid_voxels == 64
    # channel-first array shape used everywhere on disk and in torch
    assert s.array_shape == (4, 64, 64, 64)
    # 64 * 0.375 = 24 A physical cube
    assert s.side_angstroms == 24.0


def test_spec_is_frozen():
    s = PatchSpec()
    import dataclasses, pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.grid_voxels = 32

import numpy as np
from protein_patch.voxelize import gaussian_density, center_of_geometry


def test_gaussian_density_peaks_at_zero():
    assert gaussian_density(0.0, std=1.0) == 1.0
    assert gaussian_density(1.0, std=1.0) < 1.0
    # symmetric
    assert gaussian_density(2.0, 1.0) == gaussian_density(-2.0, 1.0)


def test_center_of_geometry():
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    np.testing.assert_allclose(center_of_geometry(coords), [1.0, 0.0, 0.0])

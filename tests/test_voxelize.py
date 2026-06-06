import numpy as np
from protein_patch.voxelize import gaussian_density, center_of_geometry, voxelize_atoms
from protein_patch.spec import PatchSpec


def test_gaussian_density_peaks_at_zero():
    assert gaussian_density(0.0, std=1.0) == 1.0
    assert gaussian_density(1.0, std=1.0) < 1.0
    # symmetric
    assert gaussian_density(2.0, 1.0) == gaussian_density(-2.0, 1.0)


def test_center_of_geometry():
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    np.testing.assert_allclose(center_of_geometry(coords), [1.0, 0.0, 0.0])


def test_voxelize_single_carbon_at_center():
    spec = PatchSpec(n_channels=4, grid_voxels=10, voxel_size=1.0, gaussian_std=1.0)
    # one carbon atom placed at voxel center index 5 exactly (grid_min=0, centers=0.5,1.5,...,9.5)
    # atom at 5.5 -> nearest center is index 5 (center 5.5), distance=0, unambiguous peak
    # NOTE: atom at 5.0 would be a tie between index 4 (4.5) and 5 (5.5); argmax picks (4,4,4)
    coords = np.array([[5.5, 5.5, 5.5]])
    elements = ["C"]
    grid_min = np.array([0.0, 0.0, 0.0])
    grid = voxelize_atoms(coords, elements, grid_min, spec)
    assert grid.shape == (4, 10, 10, 10)
    # carbon channel has all the density; others are empty
    assert grid[0].sum() > 0
    assert grid[1].sum() == 0 and grid[2].sum() == 0 and grid[3].sum() == 0
    # peak density sits at the voxel nearest the atom (unambiguous: atom is exactly on center)
    peak = np.unravel_index(np.argmax(grid[0]), grid[0].shape)
    assert peak == (5, 5, 5)

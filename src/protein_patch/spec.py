from dataclasses import dataclass


@dataclass(frozen=True)
class PatchSpec:
    """Geometry contract shared by the prep pipeline and the model.

    A patch is a cube of `grid_voxels` per side, with one channel per
    atom type, stored CHANNEL-FIRST: (n_channels, L, L, L). This is the
    on-disk layout AND PyTorch's native (C, D, H, W) layout, so no axis
    swapping is ever needed.

    Defaults: 64 voxels x 0.375 A = 24 A cube (12 A radius around the
    central residue). 0.375 A/voxel resolves the 1 A Gaussian kernel.
    """
    n_channels: int = 4            # C, N, O, S
    grid_voxels: int = 64          # voxels per cube side (clean powers of two)
    voxel_size: float = 0.375      # angstroms per voxel -> 24 A cube
    gaussian_std: float = 1.0      # density kernel width (angstroms)
    sasa_threshold: float = 0.2    # min relative exposure to keep a residue

    @property
    def array_shape(self) -> tuple[int, int, int, int]:
        L = self.grid_voxels
        return (self.n_channels, L, L, L)

    @property
    def side_angstroms(self) -> float:
        return self.grid_voxels * self.voxel_size

import torch

from protein_patch.spec import PatchSpec
from protein_patch.model.encoders import VoxelEncoder, PointCloudEncoder
from protein_patch.model.vae import VAE3D


def test_vae3d_voxel_roundtrip():
    spec = PatchSpec(grid_voxels=16)
    model = VAE3D(VoxelEncoder(spec), spec, latent_dim=4)
    x = torch.rand(2, *spec.array_shape)
    recon, mu, logvar = model(x)
    assert recon.shape == x.shape                 # decoder restores exactly L
    assert mu.shape == (2, 4) and logvar.shape == (2, 4)
    assert torch.all(recon >= 0)                  # softplus head


def test_vae3d_point_roundtrip_outputs_grid():
    spec = PatchSpec(grid_voxels=16)
    model = VAE3D(PointCloudEncoder(in_dim=7, feature_dim=16), spec, latent_dim=4)
    feats = torch.rand(2, 9, 7)
    mask = torch.ones(2, 9, dtype=torch.bool)
    recon, mu, logvar = model((feats, mask))
    # point in -> grid out: the controlled-ablation target is the voxel grid.
    assert recon.shape == (2, *spec.array_shape)
    assert mu.shape == (2, 4) and logvar.shape == (2, 4)


def test_vae3d_encode_returns_mu_logvar_for_both():
    spec = PatchSpec(grid_voxels=16)
    vox = VAE3D(VoxelEncoder(spec), spec, latent_dim=3)
    mu, logvar = vox.encode(torch.rand(1, *spec.array_shape))
    assert mu.shape == (1, 3) and logvar.shape == (1, 3)

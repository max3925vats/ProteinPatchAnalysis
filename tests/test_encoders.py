import torch

from protein_patch.spec import PatchSpec
from protein_patch.model.encoders import VoxelEncoder, PointCloudEncoder


def test_voxel_encoder_outputs_flat_feature():
    spec = PatchSpec(grid_voxels=16)
    enc = VoxelEncoder(spec)
    out = enc(torch.rand(2, spec.n_channels, 16, 16, 16))
    assert out.shape == (2, enc.feature_dim)
    assert enc.feature_dim == 128 * 2 ** 3        # 16 -> 8 -> 4 -> 2


def test_point_encoder_shape():
    enc = PointCloudEncoder(in_dim=7, feature_dim=32)
    feats = torch.rand(1, 5, 7)
    mask = torch.ones(1, 5, dtype=torch.bool)
    assert enc((feats, mask)).shape == (1, 32)


def test_point_encoder_permutation_invariant():
    torch.manual_seed(0)
    enc = PointCloudEncoder(in_dim=7, feature_dim=32)
    feats = torch.rand(1, 5, 7)
    mask = torch.ones(1, 5, dtype=torch.bool)
    out = enc((feats, mask))
    perm = torch.randperm(5)
    out2 = enc((feats[:, perm], mask[:, perm]))
    assert torch.allclose(out, out2, atol=1e-6)


def test_point_encoder_ignores_masked_atoms():
    # Appending padded atoms (mask=False) must NOT change the pooled feature.
    # This is the guard against an unmasked max-pool.
    torch.manual_seed(0)
    enc = PointCloudEncoder(in_dim=7, feature_dim=32)
    feats = torch.rand(1, 5, 7)
    mask = torch.ones(1, 5, dtype=torch.bool)
    base = enc((feats, mask))

    pad = torch.rand(1, 3, 7) * 100.0                      # large junk values
    feats_p = torch.cat([feats, pad], dim=1)
    mask_p = torch.cat([mask, torch.zeros(1, 3, dtype=torch.bool)], dim=1)
    padded = enc((feats_p, mask_p))
    assert torch.allclose(base, padded, atol=1e-6)

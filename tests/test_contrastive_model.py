import torch

from protein_patch.spec import PatchSpec
from protein_patch.model.encoders import VoxelEncoder, PointCloudEncoder
from protein_patch.model.contrastive import ContrastiveModel


def test_contrastive_model_voxel():
    spec = PatchSpec(grid_voxels=16)
    enc = VoxelEncoder(spec)
    model = ContrastiveModel(enc, hidden=64, proj_dim=32)
    out = model(torch.rand(2, spec.n_channels, 16, 16, 16))
    assert out.shape == (2, 32)
    assert model.encoder is enc


def test_contrastive_model_point():
    enc = PointCloudEncoder(in_dim=7, feature_dim=64)
    model = ContrastiveModel(enc, hidden=64, proj_dim=32)
    feats = torch.rand(2, 5, 7)
    mask = torch.ones(2, 5, dtype=torch.bool)
    assert model((feats, mask)).shape == (2, 32)

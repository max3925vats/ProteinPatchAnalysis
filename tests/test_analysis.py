import numpy as np
import torch

from protein_patch.spec import PatchSpec
from protein_patch.model.vae import ConvVAE3D
from protein_patch.analysis import embed_patches, pca_2d


class _ListDataset:
    def __init__(self, tensors): self.t = tensors
    def __len__(self): return len(self.t)
    def __getitem__(self, i): return self.t[i]


def test_embed_patches_returns_latent_matrix():
    spec = PatchSpec(grid_voxels=16)
    model = ConvVAE3D(spec, latent_dim=4)
    ds = _ListDataset([torch.rand(*spec.array_shape) for _ in range(6)])
    emb = embed_patches(model, ds)
    assert emb.shape == (6, 4)


def test_pca_2d_reduces_to_two_columns():
    rng = np.random.default_rng(0)
    x = rng.random((20, 5))
    out = pca_2d(x)
    assert out.shape == (20, 2)
    assert out[:, 0].var() >= out[:, 1].var()

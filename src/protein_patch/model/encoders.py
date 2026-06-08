import torch
from torch import nn

from ..spec import PatchSpec


class VoxelEncoder(nn.Module):
    """3D-CNN encoder: (B, C, L, L, L) grid -> flat (B, feature_dim).

    The conv stack (C->32->64->128, stride-2 x3) is the one extracted from the
    original ConvVAE3D, so the voxel path is numerically unchanged. feature_dim
    is computed from a dummy forward (no hardcoded shapes).
    """

    def __init__(self, spec: PatchSpec):
        super().__init__()
        self.spec = spec
        C, L = spec.n_channels, spec.grid_voxels
        self.net = nn.Sequential(
            nn.Conv3d(C, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            out = self.net(torch.zeros(1, C, L, L, L))
        self.feature_dim = int(out.flatten(1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class PointCloudEncoder(nn.Module):
    """PointNet-style encoder over a variable-size atom set.

    Per-atom feature = (centered xyz [3], element one-hot [4]) = in_dim. A shared
    MLP lifts each atom independently, then a MASKED max-pool over atoms gives a
    permutation-invariant feature. Masking sets padded atoms to -inf before the
    max so they can never win the pool.

    Input is the tuple (feats (B, N, in_dim), mask (B, N) bool).
    """

    def __init__(self, in_dim: int = 7, hidden: tuple[int, ...] = (64, 128),
                 feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        dims = (in_dim, *hidden, feature_dim)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:           # ReLU between layers, not after the last
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        feats, mask = inputs                          # (B, N, in_dim), (B, N)
        h = self.mlp(feats)                           # (B, N, feature_dim)
        neg_inf = torch.finfo(h.dtype).min
        h = h.masked_fill(~mask.unsqueeze(-1), neg_inf)
        return h.max(dim=1).values                    # (B, feature_dim)

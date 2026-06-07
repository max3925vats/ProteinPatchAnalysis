import torch
import torch.nn.functional as F
from torch import nn

from ..spec import PatchSpec
from .encoders import VoxelEncoder


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """z = mu + std * eps, std = exp(0.5 * logvar).

    The 0.5 is the fix for the original bug, which used exp(logvar) (the
    variance) as the scale instead of the standard deviation.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


def _downsampled_dim(length: int, times: int = 3) -> int:
    """Spatial size after `times` stride-2, kernel-3, padding-1 conv layers.

    Mirrors the VoxelEncoder downsampling so the decoder's start volume is
    derived from `spec` alone, NOT from the encoder — that's what lets a
    non-grid (point) encoder share this decoder. floor((L-1)/2)+1 per layer
    (64->8, 16->2).
    """
    d = length
    for _ in range(times):
        d = (d - 1) // 2 + 1
    return d


class VAE3D(nn.Module):
    """Encoder-agnostic 3D VAE: any encoder -> latent -> (C, L, L, L) grid.

    The encoder maps its native input to a flat (B, feature_dim) feature; mu/logvar
    heads are sized from `encoder.feature_dim`. The decoder's start volume is
    derived from `spec` (not the encoder), so the SAME decoder serves a voxel-CNN
    encoder and a point-cloud encoder — the reconstruction target is always the
    voxel grid (the controlled-ablation choice). Output interpolates back to
    exactly L, so the round-trip shape holds for any grid size.
    """

    def __init__(self, encoder: nn.Module, spec: PatchSpec, latent_dim: int = 8):
        super().__init__()
        self.spec = spec
        self.encoder = encoder
        C, L = spec.n_channels, spec.grid_voxels
        d = _downsampled_dim(L)
        self._start_shape = (128, d, d, d)
        flat = 128 * d ** 3
        self.fc_mu = nn.Linear(encoder.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder.feature_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose3d(32, C, 3, stride=2, padding=1),
        )

    def encode(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, *self._start_shape)
        h = self.dec(h)
        h = F.interpolate(h, size=(self.spec.grid_voxels,) * 3,
                          mode="trilinear", align_corners=False)
        # softplus: non-negative and unbounded, matching the accumulated
        # Gaussian density field whose voxels can exceed 1.0 where atoms
        # overlap. (sigmoid would cap the output at 1 and could not
        # reproduce those peaks.)
        return F.softplus(h)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class ConvVAE3D(VAE3D):
    """Voxel 3D-conv VAE — now just VAE3D fronted by a VoxelEncoder.

    Kept as a named subclass for backward compatibility: same public API
    (``__init__(spec, latent_dim)``, ``encode``/``decode``/``forward``) the rest
    of the codebase and tests already use.
    """

    def __init__(self, spec: PatchSpec, latent_dim: int = 8):
        super().__init__(VoxelEncoder(spec), spec, latent_dim)


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MSE reconstruction + KL, returned decomposed (total, recon, kl).

    Reconstruction is MSE (the Gaussian-likelihood loss appropriate for a
    continuous, unbounded density field), summed over voxels per sample
    then averaged over the batch. KL uses the standard closed form
    -0.5 * sum(1 + logvar - mu^2 - exp(logvar)), summed over the latent
    dim then averaged over the batch. Summing both per sample keeps them
    on the same "total per sample" footing; `kl_weight` is the ELBO/beta
    knob (1.0 = plain ELBO) and is expected to be tuned for these sparse,
    high-dimensional patches.
    """
    recon_l = F.mse_loss(recon, x, reduction="none").flatten(1).sum(1).mean()
    kl_l = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()
    return recon_l + kl_weight * kl_l, recon_l, kl_l

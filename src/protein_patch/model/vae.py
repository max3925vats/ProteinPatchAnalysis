import torch
import torch.nn.functional as F
from torch import nn

from ..spec import PatchSpec


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """z = mu + std * eps, std = exp(0.5 * logvar).

    The 0.5 is the fix for the original bug, which used exp(logvar) (the
    variance) as the scale instead of the standard deviation.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


class ConvVAE3D(nn.Module):
    """3D convolutional VAE for (C, L, L, L) patches.

    Encoder halves the spatial dims three times (e.g. 64->32->16->8 with
    padding), then a dense bottleneck to mu/logvar. Decoder mirrors it and
    interpolates back to exactly L on output, so the round-trip shape is
    guaranteed regardless of size.
    """

    def __init__(self, spec: PatchSpec, latent_dim: int = 8):
        super().__init__()
        self.spec = spec
        C, L = spec.n_channels, spec.grid_voxels
        self.enc = nn.Sequential(
            nn.Conv3d(C, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, C, L, L, L)
            enc_out = self.enc(dummy)
        self._enc_shape = enc_out.shape[1:]          # (128, d, d, d)
        flat = int(torch.prod(torch.tensor(self._enc_shape)))
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose3d(32, C, 3, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, *self._enc_shape)
        h = self.dec(h)
        h = F.interpolate(h, size=(self.spec.grid_voxels,) * 3,
                          mode="trilinear", align_corners=False)
        # softplus: non-negative and unbounded, matching the accumulated
        # Gaussian density field whose voxels can exceed 1.0 where atoms
        # overlap. (sigmoid would cap the output at 1 and could not
        # reproduce those peaks.)
        return F.softplus(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


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

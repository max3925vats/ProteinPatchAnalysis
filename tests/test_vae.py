import torch
from protein_patch.spec import PatchSpec
from protein_patch.model.vae import ConvVAE3D, reparameterize


def test_reparameterize_zero_logvar_is_mean_plus_unit_noise():
    torch.manual_seed(0)
    # discriminating case: logvar = ln(4) -> std MUST be 2 (exp(0.5*ln4)),
    # NOT 4 (the old exp(logvar) bug).
    logvar2 = torch.full((20000, 1), float(torch.log(torch.tensor(4.0))))
    z2 = reparameterize(torch.zeros(20000, 1), logvar2)
    assert abs(z2.std().item() - 2.0) < 0.1


def test_vae_round_trips_shape():
    spec = PatchSpec(grid_voxels=16)   # small for speed; spec-driven model
    model = ConvVAE3D(spec, latent_dim=4)
    x = torch.rand(2, *spec.array_shape)         # (2, 4, 16, 16, 16)
    recon, mu, logvar = model(x)
    assert recon.shape == x.shape
    assert mu.shape == (2, 4) and logvar.shape == (2, 4)
    assert torch.all((recon >= 0) & (recon <= 1))  # sigmoid output

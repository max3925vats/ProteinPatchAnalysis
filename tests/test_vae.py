import torch
from protein_patch.spec import PatchSpec
from protein_patch.model.vae import ConvVAE3D, reparameterize, vae_loss


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
    # softplus head: non-negative and UNBOUNDED (can exceed 1 to match
    # overlapping-atom density peaks). Only the lower bound is guaranteed.
    assert torch.all(recon >= 0)


def test_vae_loss_is_finite_and_decomposes():
    spec = PatchSpec(grid_voxels=16)
    model = ConvVAE3D(spec, latent_dim=4)
    x = torch.rand(2, *spec.array_shape)
    recon, mu, logvar = model(x)
    total, recon_l, kl_l = vae_loss(recon, x, mu, logvar, kl_weight=5e-4)
    assert torch.isfinite(total)
    assert recon_l >= 0 and kl_l >= 0
    assert torch.allclose(total, recon_l + 5e-4 * kl_l, atol=1e-4)


def test_vae_loss_handles_density_above_one():
    # Real patches accumulate overlapping Gaussians -> voxels > 1.0. MSE must
    # handle that; the old BCE loss would raise on targets outside [0, 1].
    # This is the regression guard for the BCE -> MSE switch.
    spec = PatchSpec(grid_voxels=16)
    model = ConvVAE3D(spec, latent_dim=4)
    x = torch.rand(2, *spec.array_shape) * 2.0   # values in [0, 2)
    assert x.max() > 1.0
    recon, mu, logvar = model(x)
    total, recon_l, kl_l = vae_loss(recon, x, mu, logvar)
    assert torch.isfinite(total) and recon_l >= 0


def test_one_epoch_training_runs(tmp_path, rng):
    import pickle
    from protein_patch.patches import AtomPatch
    from protein_patch.config import TrainConfig
    from protein_patch.model.train import train
    spec = PatchSpec(grid_voxels=16)
    for split in ("train", "val"):
        d = tmp_path / split; d.mkdir()
        for i in range(4):
            coords = (rng.random((5, 3)).astype("float32") - 0.5) * 3.0
            patch = AtomPatch(coords, ["C"] * 5, {}, ("x", "A", i, "ALA"))
            with open(d / f"p{i}.pickle", "wb") as f:
                pickle.dump(patch, f)
    cfg = TrainConfig(epochs=1, batch_size=2)
    hist = train(str(tmp_path / "train"), str(tmp_path / "val"), spec, cfg,
                 out=str(tmp_path / "h.json"))
    assert len(hist["train_loss"]) == 1 and len(hist["val_loss"]) == 1
    assert all(isinstance(v, float) for v in hist["train_loss"])

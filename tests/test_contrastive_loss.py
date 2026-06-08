import torch

from protein_patch.config import ContrastiveConfig
from protein_patch.model.contrastive import ProjectionHead, nt_xent_loss


def test_contrastive_config_defaults_frozen():
    c = ContrastiveConfig()
    assert c.temperature == 0.5 and c.proj_dim == 64 and c.epochs > 0
    import dataclasses, pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.temperature = 0.1            # type: ignore[misc]


def test_projection_head_shape():
    head = ProjectionHead(in_dim=128, hidden=64, out_dim=32)
    assert head(torch.rand(4, 128)).shape == (4, 32)


def test_loss_rewards_alignment():
    # aligned positive pairs should score LOWER than misaligned ones
    torch.manual_seed(0)
    z1 = torch.randn(8, 16)
    aligned = nt_xent_loss(z1, z1.clone(), temperature=0.5)
    misaligned = nt_xent_loss(z1, torch.roll(z1, shifts=1, dims=0), temperature=0.5)
    assert torch.isfinite(aligned) and aligned < misaligned


def test_loss_is_symmetric_under_view_swap():
    torch.manual_seed(1)
    z1, z2 = torch.randn(6, 16), torch.randn(6, 16)
    a = nt_xent_loss(z1, z2, temperature=0.5)
    b = nt_xent_loss(z2, z1, temperature=0.5)
    assert torch.allclose(a, b, atol=1e-5)


def test_same_protein_masking_lowers_loss():
    # two samples from the SAME protein, embeddings similar -> hard negatives.
    # masking them out of the denominator must reduce the loss (non-vacuity for C3).
    z1 = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                       [0.9, 0.1, 0.0, 0.0]])
    z2 = z1.clone()
    unmasked = nt_xent_loss(z1, z2, temperature=0.5, pids=None)
    masked = nt_xent_loss(z1, z2, temperature=0.5, pids=[7, 7])
    assert masked < unmasked


def test_distinct_proteins_are_not_masked():
    # the load-bearing half of C3: when all pids differ, NO negatives are masked,
    # so the loss must equal the no-pids loss (guards against masking everything).
    torch.manual_seed(2)
    z1, z2 = torch.randn(4, 16), torch.randn(4, 16)
    none = nt_xent_loss(z1, z2, 0.5, pids=None)
    distinct = nt_xent_loss(z1, z2, 0.5, pids=["a", "b", "c", "d"])
    assert torch.allclose(none, distinct, atol=1e-6)


def test_single_pair_is_finite():
    # B=1: the only other row is the positive -> no negatives -> finite (no NaN)
    z1 = torch.randn(1, 8)
    loss = nt_xent_loss(z1, z1.clone(), temperature=0.5, pids=[3])
    assert torch.isfinite(loss)

"""Decoder-free contrastive objective (SimCLR-style NT-Xent).

The encoder is the artifact; a projection head sits on top only during training
and is discarded. NT-Xent additionally masks SAME-PROTEIN pairs out of the
negatives (the false-negative fix): patches from one protein are near-identical,
so as negatives they would fight the objective.
"""
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

# Large finite negative for masked logits (safer in backward than -inf; the
# masked entries then contribute exp(.)≈0 to the softmax denominator).
_NEG = -1e9


class ProjectionHead(nn.Module):
    """MLP projection head: (B, in_dim) -> (B, out_dim)."""

    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveModel(nn.Module):
    """encoder -> projection head. `.encoder` is what you keep after training."""

    def __init__(self, encoder: nn.Module, hidden: int = 128, proj_dim: int = 64):
        super().__init__()
        self.encoder = encoder
        self.head = ProjectionHead(encoder.feature_dim, hidden, proj_dim)

    def forward(self, model_input) -> torch.Tensor:
        return self.head(self.encoder(model_input))


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5,
                 pids: Sequence[object] | None = None) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy over two views.

    z1, z2: (B, d) projected embeddings of the two augmented views (the i-th rows
    are a positive pair). Embeddings are L2-normalized internally. Each of the 2B
    rows treats its partner view as the positive and every other (non-self) row as
    a negative. If `pids` (length B protein ids) is given, same-protein rows other
    than the positive are masked out of the denominator.
    """
    B = z1.shape[0]
    z = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)], dim=0)  # (2B, d)
    n = 2 * B
    sim = (z @ z.t()) / temperature                                         # (2B, 2B)

    device = sim.device
    # positive index for row i: its partner view (i+B mod 2B)
    targets = torch.cat([torch.arange(B, device=device) + B,
                         torch.arange(B, device=device)])

    # never let a row attend to itself
    self_mask = torch.eye(n, dtype=torch.bool, device=device)
    sim = sim.masked_fill(self_mask, _NEG)

    if pids is not None:
        # map arbitrary (e.g. string pdb-id) ids to integer codes, tiled over
        # both views, so same-protein pairs can be compared as a tensor.
        pid_list = list(pids) + list(pids)                                  # (2B,)
        codes_by_id: dict[object, int] = {}
        codes = torch.tensor([codes_by_id.setdefault(p, len(codes_by_id))
                              for p in pid_list], device=device)
        same = codes.unsqueeze(0) == codes.unsqueeze(1)                      # (2B, 2B)
        pos_oh = F.one_hot(targets, num_classes=n).bool()                   # protect positives
        exclude = same & ~pos_oh & ~self_mask
        sim = sim.masked_fill(exclude, _NEG)

    return F.cross_entropy(sim, targets)

"""Decoder-free contrastive training + collapse diagnostics (voxel + point).

De-risk milestone: prove the NT-Xent objective trains stably (loss decreases, no
latent collapse) on the same patch slice the harness uses. Same-protein patches
are excluded from the negatives (the false-negative fix).
"""
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .augment import augment_atom_patch
from .config import ContrastiveConfig
from .patches import AtomPatch, voxelize
from .spec import PatchSpec
from .model.contrastive import ContrastiveModel, nt_xent_loss
from .model.dataset import PatchDataset
from .model.encoders import PointCloudEncoder, VoxelEncoder
from .model.point_dataset import PointPatchDataset, _point_features
from .model.train import set_seed

logger = logging.getLogger(__name__)
_KINDS = ("voxel", "point")


class ContrastiveViewDataset(Dataset):
    """Yields (view1, view2, pid): two independently augmented views per patch.

    Each view is the model_input for `kind` (a grid for voxel, an (N,7) point
    feature tensor for point) derived from a freshly augmented AtomPatch. `pid`
    is the source protein id (provenance[0]), used to mask same-protein negatives.

    Holds a single seeded RNG advanced per access — deterministic with the default
    DataLoader (num_workers=0). With num_workers>0 each worker would copy the same
    seed; add a worker_init_fn before enabling workers.
    """

    def __init__(self, kind: str, root: str | Path, spec: PatchSpec,
                 ccfg: ContrastiveConfig):
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
        self.kind = kind
        self.spec = spec
        self.ccfg = ccfg
        self.paths = sorted(Path(root).glob("*.pickle"))
        if not self.paths:
            raise FileNotFoundError(f"no *.pickle patches in {root}")
        self.rng = np.random.default_rng(ccfg.seed)

    def __len__(self) -> int:
        return len(self.paths)

    def _view(self, patch: AtomPatch) -> torch.Tensor:
        aug = augment_atom_patch(patch, self.ccfg.jitter_sigma,
                                 self.ccfg.drop_frac, self.rng)
        if self.kind == "voxel":
            return torch.from_numpy(
                np.ascontiguousarray(voxelize(aug, self.spec), dtype=np.float32))
        feats = _point_features(aug)
        if feats.shape[0] == 0:
            # dropout removed every C/N/O/S atom -> not encodable (C7). Fall back
            # to a jitter-only view (no dropout); the source patch always has CNOS
            # atoms, so this is guaranteed non-empty.
            aug = augment_atom_patch(patch, self.ccfg.jitter_sigma, 0.0, self.rng)
            feats = _point_features(aug)
            if feats.shape[0] == 0:
                raise ValueError(
                    f"patch {patch.provenance} has no C/N/O/S atoms to encode")
        return torch.from_numpy(feats)

    def __getitem__(self, idx: int):
        with open(self.paths[idx], "rb") as f:
            patch: AtomPatch = pickle.load(f)
        return self._view(patch), self._view(patch), patch.provenance[0]


def _pad_views(views: list[torch.Tensor]):
    n_max = max(v.shape[0] for v in views)
    B, feat_dim = len(views), views[0].shape[1]
    feats = torch.zeros(B, n_max, feat_dim, dtype=torch.float32)
    mask = torch.zeros(B, n_max, dtype=torch.bool)
    for i, v in enumerate(views):
        feats[i, :v.shape[0]] = v
        mask[i, :v.shape[0]] = True
    return feats, mask


def _collate(kind: str):
    def voxel(batch):
        return (torch.stack([b[0] for b in batch]),
                torch.stack([b[1] for b in batch]),
                [b[2] for b in batch])

    def point(batch):
        return (_pad_views([b[0] for b in batch]),
                _pad_views([b[1] for b in batch]),
                [b[2] for b in batch])

    return voxel if kind == "voxel" else point


def _to_device(x, device: str):
    return tuple(t.to(device) for t in x) if isinstance(x, tuple) else x.to(device)


def train_contrastive(kind: str, train_dir: str, spec: PatchSpec,
                      ccfg: ContrastiveConfig) -> tuple[dict[str, list[float]], torch.nn.Module]:
    """Train NT-Xent for one encoder kind; return (history, the kept encoder)."""
    set_seed(ccfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = (VoxelEncoder(spec) if kind == "voxel"
               else PointCloudEncoder(in_dim=7, feature_dim=128))
    model = ContrastiveModel(encoder, hidden=ccfg.hidden, proj_dim=ccfg.proj_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=ccfg.learning_rate)

    ds = ContrastiveViewDataset(kind, train_dir, spec, ccfg)
    dl = DataLoader(ds, batch_size=ccfg.batch_size, shuffle=True, collate_fn=_collate(kind))

    hist: dict[str, list[float]] = {"loss": []}
    for epoch in range(ccfg.epochs):
        model.train(); total = 0.0
        for in1, in2, pids in dl:
            in1, in2 = _to_device(in1, device), _to_device(in2, device)
            opt.zero_grad()
            z1, z2 = model(in1), model(in2)
            loss = nt_xent_loss(z1, z2, ccfg.temperature, pids=pids)
            loss.backward(); opt.step()
            total += loss.item()
        hist["loss"].append(total / max(len(dl), 1))
        logger.info("[contrastive:%s] epoch %d/%d loss=%.4f", kind, epoch + 1,
                    ccfg.epochs, hist["loss"][-1])
    return hist, model.encoder


def encode_dataset(encoder: torch.nn.Module, root: str, spec: PatchSpec,
                   kind: str, device: str = "cpu") -> np.ndarray:
    """Encoder features over the single-view harness dataset -> (n, feature_dim)."""
    encoder.eval().to(device)
    ds = PatchDataset(root, spec) if kind == "voxel" else PointPatchDataset(root, spec)
    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            if kind == "voxel":
                mi = item.unsqueeze(0).to(device)
            else:
                feats, _grid = item
                mask = torch.ones(1, feats.shape[0], dtype=torch.bool)
                mi = (feats.unsqueeze(0).to(device), mask.to(device))
            rows.append(encoder(mi).cpu().numpy()[0])
    return np.asarray(rows)


def embedding_std(emb: np.ndarray) -> float:
    """Mean per-dimension std of an (n, d) embedding matrix.

    The collapse diagnostic: a collapsed encoder maps everything to (near) one
    point, so this approaches 0. A healthy latent keeps it well above 0.
    """
    emb = np.asarray(emb, dtype=np.float64)
    return float(np.mean(np.std(emb, axis=0)))

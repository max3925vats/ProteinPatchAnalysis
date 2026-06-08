"""v0 harness driver: train a swappable-encoder VAE, embed, probe -> a cell.

Holds everything identical across encoders (datasets read the same AtomPatch
pickles, same BenchmarkSpec hyperparams, same seed, same grid reconstruction
target) so the *encoder* is the only variable. The shipped model/train.train()
is left untouched; this is the comparison-specific path.
"""
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from .benchmark_spec import BenchmarkSpec
from .spec import PatchSpec
from .eval.probe import knn_accuracy
from .model.dataset import PatchDataset
from .model.encoders import PointCloudEncoder, VoxelEncoder
from .model.point_dataset import PointPatchDataset, pad_collate
from .model.train import set_seed
from .model.vae import VAE3D, vae_loss
from .tasks.burial import assign_classes, fit_quantile_edges, read_rel_sasa

logger = logging.getLogger(__name__)

_KINDS = ("voxel", "point")


def _build(kind: str, train_dir: str, val_dir: str, spec: PatchSpec) -> tuple:
    """Return (train_ds, val_ds, encoder, collate_fn) for the given kind."""
    if kind == "voxel":
        return (PatchDataset(train_dir, spec), PatchDataset(val_dir, spec),
                VoxelEncoder(spec), None)
    if kind == "point":
        return (PointPatchDataset(train_dir, spec), PointPatchDataset(val_dir, spec),
                PointCloudEncoder(in_dim=7, feature_dim=128), pad_collate)
    raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")


def _to_device(model_input, device: str):
    if isinstance(model_input, tuple):
        return tuple(t.to(device) for t in model_input)
    return model_input.to(device)


def _split_batch(batch, kind: str) -> tuple:
    """(model_input, target_grid) from a DataLoader batch, per kind."""
    if kind == "voxel":
        return batch, batch                 # grid is both input and recon target
    (feats, mask), targets = batch
    return (feats, mask), targets


def train_encoder_vae(kind: str, train_dir: str, val_dir: str,
                      spec: PatchSpec, bspec: BenchmarkSpec
                      ) -> tuple[dict[str, list[float]], VAE3D]:
    """Train a VAE with the chosen encoder against the voxel-grid recon target.

    Returns (history, model). Seeded by bspec.seed; identical objective/optimizer
    for both kinds — the encoder is the only thing that changes.
    """
    set_seed(bspec.seed)
    cfg = bspec.train_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, encoder, collate = _build(kind, train_dir, val_dir, spec)
    model = VAE3D(encoder, spec, bspec.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    tl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    vl = DataLoader(val_ds, batch_size=cfg.batch_size, collate_fn=collate)

    hist: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    for epoch in range(cfg.epochs):
        model.train(); tr = 0.0
        for batch in tl:
            mi, target = _split_batch(batch, kind)
            mi, target = _to_device(mi, device), target.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(mi)
            loss, _, _ = vae_loss(recon, target, mu, logvar, cfg.kl_weight)
            loss.backward(); opt.step()
            tr += loss.item()
        model.eval(); va = 0.0
        with torch.no_grad():
            for batch in vl:
                mi, target = _split_batch(batch, kind)
                mi, target = _to_device(mi, device), target.to(device)
                recon, mu, logvar = model(mi)
                loss, _, _ = vae_loss(recon, target, mu, logvar, cfg.kl_weight)
                va += loss.item()
        hist["train_loss"].append(tr / max(len(tl), 1))
        hist["val_loss"].append(va / max(len(vl), 1))
        logger.info("[%s] epoch %d/%d train=%.4f val=%.4f", kind, epoch + 1,
                    cfg.epochs, hist["train_loss"][-1], hist["val_loss"][-1])
    return hist, model


def embed(model, dataset, kind: str, device: str = "cpu") -> np.ndarray:
    """Encode every patch to its latent mean (mu). Order = dataset path order."""
    model.eval().to(device)
    rows = []
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            if kind == "voxel":
                mi = item.unsqueeze(0).to(device)
            else:
                feats, _grid = item
                mask = torch.ones(1, feats.shape[0], dtype=torch.bool)
                mi = (feats.unsqueeze(0).to(device), mask.to(device))
            mu, _ = model.encode(mi)
            rows.append(mu.cpu().numpy()[0])
    return np.asarray(rows)


def run_burial_cell(kind: str, train_dir: str, val_dir: str,
                    spec: PatchSpec, bspec: BenchmarkSpec) -> dict[str, object]:
    """End-to-end matrix cell: train -> embed -> burial kNN probe.

    Labels are read in the same sorted-path order the datasets use, so they align
    positionally with the embeddings. Quantile edges are fit on TRAIN only.
    """
    hist, model = train_encoder_vae(kind, train_dir, val_dir, spec, bspec)
    train_ds, val_ds, _, _ = _build(kind, train_dir, val_dir, spec)

    tr_emb = embed(model, train_ds, kind)
    va_emb = embed(model, val_ds, kind)

    _, tr_rel = read_rel_sasa(train_dir)
    _, va_rel = read_rel_sasa(val_dir)
    edges = fit_quantile_edges(tr_rel, bspec.n_burial_classes)
    tr_lab = assign_classes(tr_rel, edges)
    va_lab = assign_classes(va_rel, edges)

    acc = knn_accuracy(tr_emb, tr_lab, va_emb, va_lab)
    return {"kind": kind, "accuracy": acc, "n_val": int(len(va_lab)),
            "n_classes": bspec.n_burial_classes}

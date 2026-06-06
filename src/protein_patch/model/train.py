import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import TrainConfig
from ..spec import PatchSpec
from .callbacks import EarlyStopping
from .dataset import PatchDataset
from .vae import ConvVAE3D, vae_loss

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed all RNGs for reproducibility, including CUDA and cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # deterministic cuDNN (may cost some GPU throughput, worth it for repro)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(train_dir: str, val_dir: str, spec: PatchSpec, cfg: TrainConfig,
          out: str | Path | None = None) -> dict[str, list[float]]:
    """Train the VAE and return the loss history.

    History is written to `out` as JSON only when `out` is provided; the
    default writes nothing (so calling from a notebook/script never drops a
    stray file into the working directory).
    """
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvVAE3D(spec, cfg.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    sched = torch.optim.lr_scheduler.StepLR(opt, cfg.lr_step_size, cfg.lr_gamma)

    tl = DataLoader(PatchDataset(train_dir), batch_size=cfg.batch_size, shuffle=True)
    vl = DataLoader(PatchDataset(val_dir), batch_size=cfg.batch_size)

    stopper = EarlyStopping(cfg.patience)
    hist: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    for epoch in range(cfg.epochs):
        model.train(); tr = 0.0
        for x in tl:
            x = x.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(x)
            loss, _, _ = vae_loss(recon, x, mu, logvar, cfg.kl_weight)
            loss.backward(); opt.step()
            tr += loss.item()
        sched.step()
        model.eval(); va = 0.0
        with torch.no_grad():
            for x in vl:
                x = x.to(device)
                recon, mu, logvar = model(x)
                loss, _, _ = vae_loss(recon, x, mu, logvar, cfg.kl_weight)
                va += loss.item()
        train_loss = tr / max(len(tl), 1)
        val_loss = va / max(len(vl), 1)
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        logger.info("epoch %d/%d  train=%.4f  val=%.4f",
                    epoch + 1, cfg.epochs, train_loss, val_loss)

        improved = stopper.step(val_loss)
        if improved and cfg.checkpoint_path is not None:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "best_val_loss": val_loss,
                "config": cfg,
            }, cfg.checkpoint_path)
        if stopper.should_stop:
            logger.info("early stopping at epoch %d (no val improvement in %d)",
                        epoch + 1, cfg.patience)
            break

    if out is not None:
        Path(out).write_text(json.dumps(hist))
    return hist

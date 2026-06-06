import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import TrainConfig
from ..spec import PatchSpec
from .dataset import PatchDataset
from .vae import ConvVAE3D, vae_loss


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
          out: str = "history.json") -> dict:
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvVAE3D(spec, cfg.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    sched = torch.optim.lr_scheduler.StepLR(opt, cfg.lr_step_size, cfg.lr_gamma)

    tl = DataLoader(PatchDataset(train_dir), batch_size=cfg.batch_size, shuffle=True)
    vl = DataLoader(PatchDataset(val_dir), batch_size=cfg.batch_size)

    hist: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    for _ in range(cfg.epochs):
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
        hist["train_loss"].append(tr / max(len(tl), 1))
        hist["val_loss"].append(va / max(len(vl), 1))

    Path(out).write_text(json.dumps(hist))
    return hist

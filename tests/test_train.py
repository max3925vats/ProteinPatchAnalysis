import pickle

import numpy as np
import torch

from protein_patch.spec import PatchSpec
from protein_patch.config import TrainConfig
from protein_patch.model.callbacks import EarlyStopping
from protein_patch.model.train import train


# --- EarlyStopping (pure logic, deterministic) ------------------------------

def test_early_stopping_triggers_after_patience_nonimproving():
    es = EarlyStopping(patience=2)
    assert es.step(1.0) is True        # improved (first value)
    assert es.should_stop is False
    assert es.step(2.0) is False       # worse, count -> 1
    assert es.should_stop is False
    assert es.step(3.0) is False       # worse, count -> 2 == patience
    assert es.should_stop is True


def test_early_stopping_resets_on_improvement():
    es = EarlyStopping(patience=2)
    es.step(1.0)
    es.step(2.0)                       # count -> 1
    assert es.step(0.5) is True        # improvement resets count
    assert es.should_stop is False
    es.step(0.6)                       # count -> 1
    es.step(0.7)                       # count -> 2
    assert es.should_stop is True


def test_early_stopping_disabled_when_patience_zero():
    es = EarlyStopping(patience=0)
    es.step(1.0)
    for _ in range(10):
        es.step(99.0)                  # never improving
    assert es.should_stop is False     # patience 0 never stops


# --- checkpoint saving (train wiring) ---------------------------------------

def _make_patches(tmp_path, rng):
    spec = PatchSpec(grid_voxels=16)
    for split in ("train", "val"):
        d = tmp_path / split
        d.mkdir()
        for i in range(4):
            arr = rng.random((4, 16, 16, 16)).astype("float32")
            with open(d / f"p{i}.pickle", "wb") as f:
                pickle.dump(arr, f)
    return spec


def test_train_saves_best_checkpoint(tmp_path, rng):
    spec = _make_patches(tmp_path, rng)
    ckpt = tmp_path / "best.pt"
    cfg = TrainConfig(epochs=2, batch_size=2, checkpoint_path=str(ckpt))
    train(str(tmp_path / "train"), str(tmp_path / "val"), spec, cfg)
    assert ckpt.exists()
    blob = torch.load(ckpt, weights_only=False)
    assert {"epoch", "model_state_dict", "optimizer_state_dict",
            "best_val_loss", "config"} <= set(blob.keys())
    assert isinstance(blob["best_val_loss"], float)


def test_train_writes_no_checkpoint_when_path_none(tmp_path, rng):
    spec = _make_patches(tmp_path, rng)
    cfg = TrainConfig(epochs=1, batch_size=2)   # checkpoint_path defaults to None
    train(str(tmp_path / "train"), str(tmp_path / "val"), spec, cfg)
    # no .pt artifact should appear anywhere under the work dir
    assert list(tmp_path.rglob("*.pt")) == []

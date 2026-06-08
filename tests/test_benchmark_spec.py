import dataclasses

import pytest

from protein_patch.benchmark_spec import BenchmarkSpec
from protein_patch.config import TrainConfig


def test_defaults_and_single_width():
    b = BenchmarkSpec()
    # v0: a SINGLE latent width (the sweep is v1), plus the shared knobs.
    assert b.latent_dim == 16
    assert b.seed == 0
    assert b.n_burial_classes == 3
    assert b.epochs > 0 and b.batch_size > 0


def test_is_frozen():
    b = BenchmarkSpec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.latent_dim = 32          # type: ignore[misc]


def test_train_config_mirrors_shared_knobs():
    b = BenchmarkSpec(latent_dim=8, seed=7, epochs=3,
                      batch_size=4, learning_rate=2e-3, kl_weight=0.5)
    cfg = b.train_config()
    assert isinstance(cfg, TrainConfig)
    assert cfg.latent_dim == 8
    assert cfg.seed == 7
    assert cfg.epochs == 3
    assert cfg.batch_size == 4
    assert cfg.learning_rate == 2e-3
    assert cfg.kl_weight == 0.5

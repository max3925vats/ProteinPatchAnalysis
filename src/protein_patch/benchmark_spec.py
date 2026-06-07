from dataclasses import dataclass

from .config import TrainConfig


@dataclass(frozen=True)
class BenchmarkSpec:
    """Fairness contract for the representation comparison.

    Pins everything held *identical* across encoders so the encoder is the only
    variable ("best" is meaningless otherwise). v0 fixes a SINGLE latent width;
    the width sweep, the matching-convention sensitivity study, and graph
    construction are later fields (see the master design doc).
    """
    latent_dim: int = 16            # single width for v0 (sweep is v1)
    seed: int = 0
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    kl_weight: float = 1.0
    n_burial_classes: int = 3

    def train_config(self) -> TrainConfig:
        """The shared reconstruction TrainConfig both encoders train under."""
        return TrainConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            latent_dim=self.latent_dim,
            kl_weight=self.kl_weight,
            seed=self.seed,
        )

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    lr_step_size: int = 30      # epochs between LR decay (replaces no-op `decay`)
    lr_gamma: float = 0.5       # multiplicative LR decay factor
    latent_dim: int = 4
    kl_weight: float = 5e-4
    seed: int = 42

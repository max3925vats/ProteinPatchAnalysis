from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    lr_step_size: int = 30      # epochs between LR decay (replaces no-op `decay`)
    lr_gamma: float = 0.5       # multiplicative LR decay factor
    latent_dim: int = 4
    kl_weight: float = 1.0      # ELBO/beta knob; tune for these sparse patches
    seed: int = 42
    patience: int = 0           # early-stop after N non-improving epochs (0 = off)
    checkpoint_path: str | None = None  # save best-by-val model here (None = off)

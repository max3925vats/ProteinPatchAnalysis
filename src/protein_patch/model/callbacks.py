from dataclasses import dataclass, field


@dataclass  # intentionally mutable: step() updates `best` and `count`
class EarlyStopping:
    """Tracks validation-loss improvement to decide when to stop training.

    `step(val_loss)` returns True if this value is a new best (an
    improvement), which the caller can use to trigger a checkpoint save.
    `should_stop` is True once `patience` consecutive non-improving values
    have been seen. A `patience` of 0 (or any value <= 0) disables stopping
    entirely.
    """

    patience: int
    best: float = field(default=float("inf"))
    count: int = 0

    def step(self, val_loss: float) -> bool:
        """Record a validation loss; return True if it improved on the best."""
        if val_loss < self.best:
            self.best = val_loss
            self.count = 0
            return True
        self.count += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.count >= self.patience

"""Early stopping callback utility."""

from __future__ import annotations

import math
from typing import Optional

__all__ = ["EarlyStopping"]


class EarlyStopping:
    """Monitor validation metric and stop training when it stops improving."""

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        if patience <= 0:
            raise ValueError("patience must be positive")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def step(self, score: float) -> None:
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def reset(self) -> None:  # allow reuse
        self.best_score = None
        self.counter = 0
        self.should_stop = False

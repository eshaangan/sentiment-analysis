"""Learning rate scheduler factory."""

from __future__ import annotations

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (CosineAnnealingLR, ExponentialLR, StepLR,
                                      _LRScheduler)

__all__ = ["create_scheduler"]


def create_scheduler(optimizer: Optimizer, name: str, **kwargs: Any) -> _LRScheduler:
    name = name.lower()
    if name == "step":
        return StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 5),
            gamma=kwargs.get("gamma", 0.1),
        )
    if name == "exponential":
        return ExponentialLR(optimizer, gamma=kwargs.get("gamma", 0.95))
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=kwargs.get("t_max", 10))
    raise ValueError(f"Unsupported scheduler: {name}")

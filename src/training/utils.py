"""Utility helpers for training scripts."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = ["set_seed", "get_device"]


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    logger.info("Global random seed set to %d", seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

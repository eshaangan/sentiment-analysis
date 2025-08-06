"""Regularization utilities for training.

Currently includes:
• LabelSmoothingCrossEntropy – standard label-smoothing criterion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["LabelSmoothingCrossEntropy"]


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing.

    Arguments
    ---------
    smoothing: float
        Factor in ``[0, 1)``. 0 ⇒ standard cross-entropy.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:  # noqa: D401
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        loss = -torch.sum(true_dist * log_probs, dim=-1).mean()
        return loss

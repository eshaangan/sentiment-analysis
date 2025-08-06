"""Tests for regularization utilities."""

import pytest
import torch

from src.models.regularization import LabelSmoothingCrossEntropy


def test_label_smoothing_basic():
    logits = torch.tensor([[2.0, 0.5, 0.3]])
    target = torch.tensor([0])
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss = criterion(logits, target)
    assert loss.item() > 0.0


def test_label_smoothing_invalid():
    with pytest.raises(ValueError):
        LabelSmoothingCrossEntropy(smoothing=1.2)

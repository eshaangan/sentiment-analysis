"""Tests for evaluation.metrics."""

import numpy as np

from src.evaluation.metrics import (compute_classification_metrics,
                                    get_confusion_matrix)


def test_metrics_binary():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert np.isclose(metrics["precision"], 0.6666666)
    assert np.isclose(metrics["recall"], 1.0)

    cm = get_confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)
    assert cm.sum() == 4

"""Tests for ROC AUC utility."""

import numpy as np

from src.evaluation.metrics import roc_curve_and_auc


def test_auc_value():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]
    fpr, tpr, auc = roc_curve_and_auc(y_true, y_scores)
    assert np.isclose(auc, 1.0)
    assert np.all(fpr >= 0) and np.all(fpr <= 1)
    assert np.all(tpr >= 0) and np.all(tpr <= 1)

"""Evaluation metrics wrappers using scikit-learn."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

__all__ = [
    "compute_classification_metrics",
    "get_confusion_matrix",
    "get_classification_report",
    "roc_curve_and_auc",
]


def _to_numpy(y: Sequence[int] | np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=int)


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    average: str = "binary",
) -> Dict[str, float]:
    """Return accuracy, precision, recall, f1 in a dict."""
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)
    acc = accuracy_score(y_t, y_p)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_t, y_p, average=average, zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def get_confusion_matrix(
    y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray
) -> np.ndarray:
    return confusion_matrix(_to_numpy(y_true), _to_numpy(y_pred))


def get_classification_report(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    target_names: List[str] | None = None,
) -> str:
    return classification_report(
        _to_numpy(y_true), _to_numpy(y_pred), target_names=target_names, zero_division=0
    )


def roc_curve_and_auc(
    y_true: Sequence[int] | np.ndarray,
    y_scores: Sequence[float] | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return FPR, TPR arrays and AUC value for binary classification."""
    y_t = _to_numpy(y_true)
    y_s = np.asarray(y_scores, dtype=float)
    fpr, tpr, _ = roc_curve(y_t, y_s)
    auc = roc_auc_score(y_t, y_s)
    return fpr, tpr, float(auc)

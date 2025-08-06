"""Simple hyper-parameter tuning utilities (grid search).

Usage example::

    param_grid = {
        "lr": [1e-3, 5e-4],
        "batch_size": [16, 32],
    }

    best_cfg, history = grid_search(train_func, param_grid)
"""

from __future__ import annotations

import itertools
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

__all__ = ["grid_search"]


def _dict_product(param_grid: Dict[str, List[Any]]):
    keys = list(param_grid)
    for values in itertools.product(*param_grid.values()):
        yield dict(zip(keys, values))


def grid_search(
    train_fn: Callable[[Dict[str, Any]], float],
    param_grid: Dict[str, List[Any]],
) -> Tuple[Dict[str, Any], List[Tuple[Dict[str, Any], float]]]:
    """Exhaustive grid search.

    ``train_fn`` should accept a parameter dict and return validation loss.
    Returns best parameter set and full history.
    """
    history: List[Tuple[Dict[str, Any], float]] = []
    best_params: Dict[str, Any] | None = None
    best_loss: float = float("inf")

    for params in _dict_product(param_grid):
        loss = train_fn(deepcopy(params))
        history.append((params, loss))
        logger.info("GridSearch params=%s val_loss=%.4f", params, loss)
        if loss < best_loss:
            best_loss = loss
            best_params = params
    assert best_params is not None
    return best_params, history

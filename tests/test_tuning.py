"""Test grid search utility with dummy train function."""

from src.training.tuning import grid_search


def _train_fn(params):
    # Fake loss: (lr-0.01)^2 + (batch_size/100)^2
    lr = params["lr"]
    bs = params["batch_size"]
    return (lr - 0.01) ** 2 + (bs / 100) ** 2


def test_grid_search_returns_best():
    grid = {"lr": [0.005, 0.01], "batch_size": [32, 64]}
    best, history = grid_search(_train_fn, grid)
    assert best["lr"] == 0.01
    assert best["batch_size"] == 32
    assert len(history) == 4

"""Smoke test plot_history returns a matplotlib Figure."""

from matplotlib.figure import Figure

from src.training.visualization import plot_history


def test_plot_history_returns_figure():
    hist = [
        {
            "epoch": 1,
            "train_loss": 1.0,
            "train_acc": 0.5,
            "val_loss": 0.9,
            "val_acc": 0.55,
        },
        {
            "epoch": 2,
            "train_loss": 0.8,
            "train_acc": 0.6,
            "val_loss": 0.85,
            "val_acc": 0.58,
        },
    ]
    fig = plot_history(hist)
    assert isinstance(fig, Figure)

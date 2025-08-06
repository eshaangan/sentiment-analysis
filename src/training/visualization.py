"""Visualization helpers for training history."""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt

__all__ = ["plot_history"]


def plot_history(history: List[Dict[str, float]]):
    """Plot loss/accuracy curves from Trainer.history."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(epochs, train_loss, label="train")
    ax[0].plot(epochs, val_loss, label="val")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].legend()

    ax[1].plot(epochs, train_acc, label="train")
    ax[1].plot(epochs, val_acc, label="val")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].legend()
    fig.tight_layout()
    return fig

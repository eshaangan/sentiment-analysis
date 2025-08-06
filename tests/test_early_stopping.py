"""Test EarlyStopping mechanism with Trainer."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn_model import create_cnn_model
from src.training.early_stopping import EarlyStopping
from src.training.trainer import Trainer


def _loader():
    x = torch.randint(0, 30, (40, 12))
    y = torch.randint(0, 2, (40,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_early_stopping_trigger():
    model = create_cnn_model(
        vocab_size=30, embed_dim=16, num_filters=4, filter_sizes=[3]
    )
    loader = _loader()
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)

    stopper = EarlyStopping(patience=1, min_delta=0.0, mode="min")
    trainer = Trainer(model, loader, loader, criterion, optim)
    trainer.fit(epochs=5, early_stopper=stopper)

    # should stop early so history length < 5
    assert len(trainer.history) < 5
    assert stopper.should_stop is True

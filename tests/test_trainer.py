"""Smoke test for Trainer class using tiny random data."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn_model import create_cnn_model
from src.training.trainer import Trainer
from src.training.utils import set_seed


def _create_dummy_loaders(vocab_size: int = 50):
    set_seed(1)
    x = torch.randint(0, vocab_size, (20, 10))
    y = torch.randint(0, 2, (20,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4)
    return loader


def test_trainer_train_eval():
    model = create_cnn_model(
        vocab_size=50, embed_dim=16, num_filters=4, filter_sizes=[3, 4]
    )
    train_loader = _create_dummy_loaders()
    val_loader = _create_dummy_loaders()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, grad_clip=1.0
    )

    train_loss, train_acc = trainer.train_epoch()
    val_loss, val_acc = trainer.evaluate()

    assert train_loss > 0
    assert 0 <= train_acc <= 1
    assert val_loss > 0
    assert 0 <= val_acc <= 1


def test_trainer_fit_history():
    model = create_cnn_model(
        vocab_size=50, embed_dim=16, num_filters=4, filter_sizes=[3, 4]
    )
    loader = _create_dummy_loaders()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, loader, loader, criterion, optimizer)
    trainer.fit(epochs=2)
    assert len(trainer.history) == 2
    for record in trainer.history:
        assert set(record.keys()) == {
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
        }

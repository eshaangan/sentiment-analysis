"""Test save/load checkpoint utilities."""

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn_model import create_cnn_model
from src.training.trainer import Trainer


def _dummy_loader(device="cpu"):
    x = torch.randint(0, 20, (4, 10), device=device)
    y = torch.randint(0, 2, (4,), device=device)
    return DataLoader(TensorDataset(x, y), batch_size=2)


def test_save_load_checkpoint_roundtrip():
    model = create_cnn_model(
        vocab_size=20, embed_dim=8, num_filters=4, filter_sizes=[3]
    )
    loader = _dummy_loader(device=model.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model, loader, None, criterion, optimizer)
    loss, acc = trainer.train_epoch()

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "ckpt.pt"
        trainer.save_checkpoint(ckpt_path, epoch=1, metrics={"train_loss": loss})

        # make new model/optim and load
        new_model = create_cnn_model(
            vocab_size=20, embed_dim=8, num_filters=4, filter_sizes=[3]
        )
        new_optim = torch.optim.Adam(new_model.parameters())
        metrics = Trainer.load_checkpoint(ckpt_path, new_model, new_optim)
        assert "train_loss" in metrics

        # ensure state dict identical
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

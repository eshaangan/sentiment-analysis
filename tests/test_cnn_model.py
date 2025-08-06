"""Tests for CNN-based sentiment analysis model."""

import pytest
import torch
import torch.nn as nn

from src.models.cnn_model import CNNConfig, CNNModel, create_cnn_model


class TestCNNConfig:
    def test_basic_initialisation(self):
        config = CNNConfig(vocab_size=5000, embed_dim=128)
        assert config.hidden_dim == config.num_filters * len(config.filter_sizes)
        assert config.model_type == "cnn"

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            CNNConfig(vocab_size=1000, num_filters=0)
        with pytest.raises(ValueError):
            CNNConfig(vocab_size=1000, filter_sizes=[])
        with pytest.raises(ValueError):
            CNNConfig(vocab_size=1000, filter_sizes=[-3, 4])


class TestCNNModel:
    def _create_model(self):
        return create_cnn_model(
            vocab_size=1000, embed_dim=64, num_filters=16, filter_sizes=[3, 4, 5]
        )

    def test_initialisation(self):
        model = self._create_model()
        # Ensure conv layers correct
        assert isinstance(model.convs, nn.ModuleList)
        assert len(model.convs) == 3
        for conv in model.convs:
            assert isinstance(conv, nn.Conv2d)
            assert conv.out_channels == 16

    def test_forward_pass(self):
        model = self._create_model()
        model.eval()
        batch_size, seq_len = 4, 12
        input_ids = torch.randint(
            0, 1000, (batch_size, seq_len), device=model.device
        )
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.shape == (batch_size, model.config.output_dim)

    def test_prediction(self):
        model = self._create_model()
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(
            0, 1000, (batch_size, seq_len), device=model.device
        )
        probs = model.predict(input_ids)
        assert probs.shape == (batch_size, model.config.output_dim)
        assert torch.allclose(
            probs.sum(dim=1), torch.ones(batch_size, device=model.device), atol=1e-6
        )

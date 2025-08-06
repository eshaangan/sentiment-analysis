"""Tests for Transformer-based sentiment analysis model."""

import pytest
import torch

from src.models.transformer_model import (TransformerConfig, TransformerModel,
                                          create_transformer_model)


class TestTransformerConfig:
    def test_valid_initialisation(self):
        config = TransformerConfig(vocab_size=5000, embed_dim=128, num_heads=8)
        assert config.embed_dim % config.num_heads == 0
        assert config.model_type == "transformer"

    def test_invalid_heads(self):
        with pytest.raises(ValueError):
            TransformerConfig(vocab_size=1000, embed_dim=130, num_heads=8)


class TestTransformerModel:
    def _create_model(self):
        return create_transformer_model(
            vocab_size=1000, embed_dim=64, num_heads=8, hidden_dim=128, num_layers=2
        )

    def test_forward(self):
        model = self._create_model()
        model.eval()
        batch_size, seq_len = 3, 20
        input_ids = torch.randint(
            0, 1000, (batch_size, seq_len), device=model.device
        )
        attention_mask = torch.ones(batch_size, seq_len, device=model.device)
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        assert logits.shape == (batch_size, model.config.output_dim)

    def test_predict(self):
        model = self._create_model()
        batch_size, seq_len = 2, 15
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=model.device)
        probs = model.predict(input_ids)
        assert probs.shape == (batch_size, model.config.output_dim)
        assert torch.allclose(
            probs.sum(dim=1), torch.ones(batch_size, device=model.device), atol=1e-6
        )

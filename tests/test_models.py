"""
Tests for model architecture modules.
"""

import tempfile
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.models.base_model import BaseModel, ModelConfig
from src.models.lstm_model import (AttentionLayer, AttentionLSTM,
                                   BidirectionalLSTM, LSTMConfig, LSTMModel,
                                   create_attention_lstm,
                                   create_bidirectional_lstm,
                                   create_lstm_model)


class SimpleTestModel(BaseModel):
    """Simple concrete implementation of BaseModel for testing."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Simple architecture for testing
        self.rnn = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

        # Update classifier input dimension
        rnn_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim),
        )
        # Move all layers to the correct device
        self.to(self.device)

    def _get_classifier_input_dim(self) -> int:
        return self.config.hidden_dim * (2 if self.config.bidirectional else 1)

    def forward(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        # Embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        embedded = self.dropout(embedded)

        # LSTM
        lstm_out, (hidden, _) = self.rnn(
            embedded
        )  # lstm_out: [batch_size, seq_len, hidden_dim * directions]

        # Use last hidden state for classification
        if self.config.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat(
                (hidden[-2], hidden[-1]), dim=1
            )  # [batch_size, hidden_dim * 2]
        else:
            hidden = hidden[-1]  # [batch_size, hidden_dim]

        # Classification
        logits = self.classifier(hidden)  # [batch_size, output_dim]
        return logits


class TestModelConfig:
    """Test cases for ModelConfig class."""

    def test_model_config_initialization(self):
        """Test basic ModelConfig initialization."""
        config = ModelConfig(
            vocab_size=10000, embed_dim=128, hidden_dim=256, output_dim=2
        )

        assert config.vocab_size == 10000
        assert config.embed_dim == 128
        assert config.hidden_dim == 256
        assert config.output_dim == 2
        assert config.num_layers == 2  # default
        assert config.dropout == 0.3  # default
        assert config.bidirectional is True  # default
        assert config.model_type == "lstm"  # default

    def test_model_config_validation(self):
        """Test ModelConfig validation."""
        # Test invalid vocab_size
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            ModelConfig(vocab_size=0)

        # Test invalid dropout
        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            ModelConfig(vocab_size=1000, dropout=1.5)

    def test_model_config_kwargs(self):
        """Test ModelConfig with additional kwargs."""
        config = ModelConfig(vocab_size=1000, learning_rate=0.001, custom_param="test")

        assert hasattr(config, "learning_rate")
        assert config.learning_rate == 0.001
        assert hasattr(config, "custom_param")
        assert config.custom_param == "test"

    def test_model_config_to_dict(self):
        """Test ModelConfig conversion to dictionary."""
        config = ModelConfig(vocab_size=1000, embed_dim=64)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "vocab_size" in config_dict
        assert "embed_dim" in config_dict
        assert config_dict["vocab_size"] == 1000
        assert config_dict["embed_dim"] == 64

    def test_model_config_save_load(self):
        """Test ModelConfig save and load functionality."""
        config = ModelConfig(
            vocab_size=5000, embed_dim=100, hidden_dim=200, custom_param="test_value"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_config.yaml"

            # Test save
            config.save(save_path)
            assert save_path.exists()

            # Test load (create a minimal YAML structure for testing)
            yaml_content = """
lstm:
  vocab_size: 5000
  embed_dim: 100
  hidden_dim: 200
  custom_param: test_value
            """
            save_path.write_text(yaml_content.strip())

            loaded_config = ModelConfig.from_yaml(save_path, "lstm")
            assert loaded_config.vocab_size == 5000
            assert loaded_config.embed_dim == 100
            assert loaded_config.hidden_dim == 200
            assert loaded_config.custom_param == "test_value"


class TestBaseModel:
    """Test cases for BaseModel class."""

    def create_test_config(self) -> ModelConfig:
        """Create a test configuration."""
        return ModelConfig(
            vocab_size=1000,
            embed_dim=64,
            hidden_dim=128,
            output_dim=2,
            num_layers=1,  # Simple for testing
            dropout=0.1,
            bidirectional=False,  # Simple for testing
        )

    def test_base_model_initialization(self):
        """Test BaseModel initialization."""
        config = self.create_test_config()
        model = SimpleTestModel(config)

        assert model.config == config
        assert isinstance(model.embedding, nn.Embedding)
        assert isinstance(model.dropout, nn.Dropout)
        assert isinstance(model.classifier, nn.Module)
        assert model.embedding.num_embeddings == config.vocab_size
        assert model.embedding.embedding_dim == config.embed_dim

    def test_model_device_detection(self):
        """Test automatic device detection."""
        config = self.create_test_config()
        model = SimpleTestModel(config)

        # Should be either cpu, cuda, or mps
        assert model.device.type in ["cpu", "cuda", "mps"]

        # Model should be on the detected device
        assert next(model.parameters()).device.type == model.device.type

    def test_model_forward_pass(self):
        """Test model forward pass."""
        config = self.create_test_config()
        model = SimpleTestModel(config)
        model.eval()

        # Create test input
        batch_size, seq_len = 4, 10
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=model.device
        )
        attention_mask = torch.ones(batch_size, seq_len, device=model.device)

        # Forward pass
        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        # Check output shape
        assert logits.shape == (batch_size, config.output_dim)
        assert logits.dtype == torch.float32

    def test_model_predict(self):
        """Test model prediction functionality."""
        config = self.create_test_config()
        model = SimpleTestModel(config)

        # Create test input
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=model.device
        )

        # Test predict method
        probabilities = model.predict(input_ids)

        # Check output
        assert probabilities.shape == (batch_size, config.output_dim)
        assert torch.allclose(
            probabilities.sum(dim=-1),
            torch.ones(batch_size, device=model.device),
            atol=1e-6,
        )  # Probabilities sum to 1
        assert (probabilities >= 0).all()  # All probabilities non-negative
        assert (probabilities <= 1).all()  # All probabilities <= 1

    def test_model_predict_single(self):
        """Test single prediction functionality."""
        config = self.create_test_config()
        model = SimpleTestModel(config)

        # Create single input
        seq_len = 5
        input_ids = torch.randint(0, config.vocab_size, (seq_len,))

        # Test single prediction
        predicted_class, confidence = model.predict_single(input_ids)

        # Check output types and ranges
        assert isinstance(predicted_class, int)
        assert isinstance(confidence, float)
        assert 0 <= predicted_class < config.output_dim
        assert 0 <= confidence <= 1

    def test_parameter_counting(self):
        """Test parameter counting functionality."""
        config = self.create_test_config()
        model = SimpleTestModel(config)

        # Test parameter count
        param_count = model.count_parameters()
        assert param_count > 0
        assert isinstance(param_count, int)

        # Manual count for verification
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count == manual_count

    def test_parameter_summary(self):
        """Test parameter summary functionality."""
        config = self.create_test_config()
        model = SimpleTestModel(config)

        summary = model.get_parameter_summary()

        # Check required keys
        required_keys = [
            "total_parameters",
            "trainable_parameters",
            "non_trainable_parameters",
            "layer_breakdown",
            "model_size_mb",
        ]
        for key in required_keys:
            assert key in summary

        # Check types and values
        assert isinstance(summary["total_parameters"], int)
        assert isinstance(summary["trainable_parameters"], int)
        assert isinstance(summary["model_size_mb"], float)
        assert summary["total_parameters"] >= summary["trainable_parameters"]
        assert summary["model_size_mb"] > 0

    def test_model_save_load(self):
        """Test model save and load functionality."""
        config = self.create_test_config()
        model = SimpleTestModel(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_model.pt"

            # Save model
            model.save_model(save_path)
            assert save_path.exists()

            # Create a simplified load test (full load requires model registry)
            checkpoint = torch.load(save_path, map_location="cpu")
            assert "model_state_dict" in checkpoint
            assert "model_class" in checkpoint
            assert "config" in checkpoint
            assert checkpoint["model_class"] == "SimpleTestModel"

    def test_embedding_operations(self):
        """Test embedding layer operations."""
        config = self.create_test_config()
        model = SimpleTestModel(config)

        # Test freeze/unfreeze
        model.freeze_embedding()
        assert not model.embedding.weight.requires_grad

        model.unfreeze_embedding()
        assert model.embedding.weight.requires_grad

        # Test get embedding weights
        weights = model.get_embedding_weights()
        assert weights.shape == (config.vocab_size, config.embed_dim)
        assert weights.dtype == torch.float32

        # Test set embedding weights
        new_weights = torch.randn(
            config.vocab_size, config.embed_dim, device=model.device
        )
        model.set_embedding_weights(new_weights)
        assert torch.allclose(model.embedding.weight, new_weights)

    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        config = self.create_test_config()
        model = SimpleTestModel(config)

        # Create dummy loss and backward pass
        input_ids = torch.randint(0, config.vocab_size, (2, 5), device=model.device)
        target = torch.randint(0, config.output_dim, (2,), device=model.device)

        logits = model(input_ids)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()

        # Test gradient clipping
        norm = model.apply_gradient_clipping(max_norm=1.0)
        assert isinstance(norm, float)
        assert norm >= 0

    def test_model_summary(self):
        """Test model summary generation."""
        config = self.create_test_config()
        model = SimpleTestModel(config)

        summary = model.get_model_summary()

        # Check that summary is a string and contains expected information
        assert isinstance(summary, str)
        assert "SimpleTestModel" in summary
        assert "Total Parameters" in summary
        assert "Model Size" in summary
        assert "Layer Breakdown" in summary

    def test_model_modes(self):
        """Test model training/evaluation modes."""
        config = self.create_test_config()
        model = SimpleTestModel(config)

        # Test training mode
        model.train()
        assert model.training

        # Test evaluation mode
        model.eval()
        assert not model.training

        # Test prediction in eval mode
        input_ids = torch.randint(0, config.vocab_size, (1, 5), device=model.device)
        with torch.no_grad():
            probs = model.predict(input_ids)
        assert probs.shape == (1, config.output_dim)

    def test_model_repr(self):
        """Test model string representation."""
        config = self.create_test_config()
        model = SimpleTestModel(config)

        repr_str = repr(model)
        assert isinstance(repr_str, str)
        assert "SimpleTestModel" in repr_str
        assert "config=" in repr_str


class TestLSTMConfig:
    """Test cases for LSTMConfig class."""

    def test_lstm_config_initialization(self):
        """Test basic LSTMConfig initialization."""
        config = LSTMConfig(
            vocab_size=5000,
            embed_dim=128,
            hidden_dim=256,
            output_dim=2,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            attention=True,
            pooling="attention",
        )

        assert config.vocab_size == 5000
        assert config.embed_dim == 128
        assert config.hidden_dim == 256
        assert config.output_dim == 2
        assert config.num_layers == 2
        assert config.dropout == 0.3
        assert config.bidirectional is True
        assert config.attention is True
        assert config.pooling == "attention"
        assert config.attention_heads == 8  # default
        assert config.attention_dropout == 0.1  # default

    def test_lstm_config_validation(self):
        """Test LSTMConfig validation."""
        # Test invalid lstm_dropout
        with pytest.raises(ValueError, match="lstm_dropout must be between 0 and 1"):
            LSTMConfig(vocab_size=1000, lstm_dropout=1.5)

        # Test invalid attention_heads
        with pytest.raises(ValueError, match="attention_heads must be positive"):
            LSTMConfig(vocab_size=1000, attention_heads=0)

        # Test invalid pooling
        with pytest.raises(ValueError, match="pooling must be one of"):
            LSTMConfig(vocab_size=1000, pooling="invalid")

    def test_lstm_config_attention_auto_enable(self):
        """Test that attention is automatically enabled when pooling is 'attention'."""
        config = LSTMConfig(vocab_size=1000, pooling="attention")
        assert config.attention is True


class TestAttentionLayer:
    """Test cases for AttentionLayer class."""

    def test_attention_layer_initialization(self):
        """Test AttentionLayer initialization."""
        hidden_dim = 128
        num_heads = 8
        attention = AttentionLayer(hidden_dim, num_heads, dropout=0.1)

        assert attention.hidden_dim == hidden_dim
        assert attention.num_heads == num_heads
        assert attention.head_dim == hidden_dim // num_heads

    def test_attention_layer_invalid_heads(self):
        """Test AttentionLayer with invalid number of heads."""
        with pytest.raises(
            ValueError, match="hidden_dim.*must be divisible by num_heads"
        ):
            AttentionLayer(hidden_dim=127, num_heads=8)

    def test_attention_layer_forward(self):
        """Test AttentionLayer forward pass."""
        batch_size, seq_len, hidden_dim = 4, 10, 128
        num_heads = 8

        attention = AttentionLayer(hidden_dim, num_heads, dropout=0.1).to("mps")
        x = torch.randn(batch_size, seq_len, hidden_dim, device="mps")
        mask = torch.ones(batch_size, seq_len, device="mps")

        output, weights = attention(x, mask)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert weights.shape == (batch_size, num_heads, seq_len, seq_len)

    def test_attention_layer_with_mask(self):
        """Test AttentionLayer with attention mask."""
        batch_size, seq_len, hidden_dim = 2, 8, 64
        num_heads = 4

        attention = AttentionLayer(hidden_dim, num_heads, dropout=0.1).to("mps")
        x = torch.randn(batch_size, seq_len, hidden_dim, device="mps")
        mask = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0]], device="mps"
        )

        output, weights = attention(x, mask)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert weights.shape == (batch_size, num_heads, seq_len, seq_len)

        # Check that masked positions have zero attention weights
        mask = mask.view(batch_size, 1, 1, seq_len).expand_as(weights)
        assert torch.allclose(weights.masked_fill(mask == 0, 0), weights, atol=1e-6)


class TestLSTMModel:
    """Test cases for LSTMModel class."""

    def create_lstm_config(self) -> LSTMConfig:
        """Create a test LSTM configuration."""
        return LSTMConfig(
            vocab_size=1000,
            embed_dim=64,
            hidden_dim=128,
            output_dim=2,
            num_layers=1,
            dropout=0.1,
            bidirectional=False,
        )

    def test_lstm_model_initialization(self):
        """Test LSTMModel initialization."""
        config = self.create_lstm_config()
        model = LSTMModel(config)

        assert isinstance(model.lstm, nn.LSTM)
        assert model.attention is None  # No attention by default
        assert isinstance(model.classifier, nn.Module)
        assert model.lstm.input_size == config.embed_dim
        assert model.lstm.hidden_size == config.hidden_dim
        assert model.lstm.num_layers == config.num_layers
        assert model.lstm.bidirectional == config.bidirectional

    def test_lstm_model_with_attention(self):
        """Test LSTMModel with attention mechanism."""
        config = LSTMConfig(
            vocab_size=1000,
            embed_dim=64,
            hidden_dim=128,
            output_dim=2,
            attention=True,
            pooling="attention",
        )
        model = LSTMModel(config)

        assert model.attention is not None
        assert isinstance(model.attention, AttentionLayer)
        assert model.config.pooling == "attention"

    def test_lstm_model_forward_pass(self):
        """Test LSTMModel forward pass."""
        config = self.create_lstm_config()
        model = LSTMModel(config)
        model.eval()

        batch_size, seq_len = 4, 10
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=model.device
        )
        attention_mask = torch.ones(batch_size, seq_len, device=model.device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        assert logits.shape == (batch_size, config.output_dim)
        assert logits.dtype == torch.float32

    def test_lstm_model_different_pooling(self):
        """Test LSTMModel with different pooling strategies."""
        vocab_size, embed_dim, hidden_dim, output_dim = 1000, 64, 128, 2
        batch_size, seq_len = 2, 8

        pooling_strategies = ["last", "mean", "max"]

        for pooling in pooling_strategies:
            config = LSTMConfig(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                pooling=pooling,
            )
            model = LSTMModel(config)
            model.eval()

            input_ids = torch.randint(
                0, vocab_size, (batch_size, seq_len), device=model.device
            )
            attention_mask = torch.ones(batch_size, seq_len, device=model.device)

            with torch.no_grad():
                logits = model(input_ids, attention_mask)

            assert logits.shape == (batch_size, output_dim)

    def test_lstm_model_attention_pooling(self):
        """Test LSTMModel with attention pooling."""
        config = LSTMConfig(
            vocab_size=1000,
            embed_dim=64,
            hidden_dim=128,
            output_dim=2,
            attention=True,
            pooling="attention",
        )
        model = LSTMModel(config)
        model.eval()

        batch_size, seq_len = 2, 6
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=model.device
        )
        attention_mask = torch.ones(batch_size, seq_len, device=model.device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        assert logits.shape == (batch_size, config.output_dim)

    def test_lstm_model_bidirectional(self):
        """Test LSTMModel with bidirectional LSTM."""
        config = LSTMConfig(
            vocab_size=1000,
            embed_dim=64,
            hidden_dim=128,
            output_dim=2,
            bidirectional=True,
        )
        model = LSTMModel(config)

        assert model.lstm.bidirectional is True
        assert model.lstm.hidden_size == 128
        # Bidirectional LSTM output dimension should be doubled
        expected_output_dim = 128 * 2
        assert model.classifier[0].in_features == expected_output_dim

    def test_lstm_model_get_attention_weights(self):
        """Test getting attention weights from LSTM model."""
        config = LSTMConfig(
            vocab_size=1000,
            embed_dim=64,
            hidden_dim=128,
            output_dim=2,
            attention=True,
            pooling="attention",
        )
        model = LSTMModel(config)

        batch_size, seq_len = 2, 6
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=model.device
        )
        attention_mask = torch.ones(batch_size, seq_len, device=model.device)

        attention_weights = model.get_attention_weights(input_ids, attention_mask)

        assert attention_weights is not None
        assert attention_weights.shape == (batch_size, model.attention.num_heads, seq_len, seq_len)
        # Check that attention weights sum to 1 across the sequence dimension
        expected_ones = torch.ones(batch_size, model.attention.num_heads, seq_len, device=attention_weights.device)
        assert torch.allclose(
            attention_weights.sum(dim=-1), expected_ones, atol=1e-6
        )

    def test_lstm_model_no_attention_weights(self):
        """Test getting attention weights when attention is disabled."""
        config = self.create_lstm_config()  # No attention
        model = LSTMModel(config)

        batch_size, seq_len = 2, 6
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        attention_weights = model.get_attention_weights(input_ids)

        assert attention_weights is None


class TestBidirectionalLSTM:
    """Test cases for BidirectionalLSTM class."""

    def test_bidirectional_lstm_initialization(self):
        """Test BidirectionalLSTM initialization."""
        config = ModelConfig(
            vocab_size=1000, embed_dim=64, hidden_dim=128, output_dim=2
        )
        model = BidirectionalLSTM(config)

        assert model.config.bidirectional is True
        assert model.lstm.bidirectional is True

    def test_bidirectional_lstm_forward(self):
        """Test BidirectionalLSTM forward pass."""
        config = ModelConfig(
            vocab_size=1000, embed_dim=64, hidden_dim=128, output_dim=2
        )
        model = BidirectionalLSTM(config)
        model.eval()

        batch_size, seq_len = 3, 8
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=model.device
        )

        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (batch_size, config.output_dim)


class TestAttentionLSTM:
    """Test cases for AttentionLSTM class."""

    def test_attention_lstm_initialization(self):
        """Test AttentionLSTM initialization."""
        config = ModelConfig(
            vocab_size=1000, embed_dim=64, hidden_dim=128, output_dim=2
        )
        model = AttentionLSTM(config)

        assert model.config.attention is True
        assert model.config.pooling == "attention"
        assert model.attention is not None

    def test_attention_lstm_forward(self):
        """Test AttentionLSTM forward pass."""
        config = ModelConfig(
            vocab_size=1000, embed_dim=64, hidden_dim=128, output_dim=2
        )
        model = AttentionLSTM(config)
        model.eval()

        batch_size, seq_len = 3, 8
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=model.device
        )

        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (batch_size, config.output_dim)


class TestLSTMFactoryFunctions:
    """Test cases for LSTM factory functions."""

    def test_create_lstm_model(self):
        """Test create_lstm_model factory function."""
        model = create_lstm_model(
            vocab_size=1000,
            embed_dim=64,
            hidden_dim=128,
            output_dim=2,
            bidirectional=True,
            attention=True,
            pooling="attention",
        )

        assert isinstance(model, LSTMModel)
        assert model.config.bidirectional is True
        assert model.config.attention is True
        assert model.config.pooling == "attention"

    def test_create_bidirectional_lstm(self):
        """Test create_bidirectional_lstm factory function."""
        model = create_bidirectional_lstm(
            vocab_size=1000, embed_dim=64, hidden_dim=128, output_dim=2
        )

        assert isinstance(model, BidirectionalLSTM)
        assert model.config.bidirectional is True

    def test_create_attention_lstm(self):
        """Test create_attention_lstm factory function."""
        model = create_attention_lstm(
            vocab_size=1000,
            embed_dim=64,
            hidden_dim=128,
            output_dim=2,
            attention_heads=4,
        )

        assert isinstance(model, AttentionLSTM)
        assert model.config.attention is True
        assert model.config.pooling == "attention"
        assert model.config.attention_heads == attention_heads


class TestModelIntegration:
    """Integration tests for model components."""

    def test_model_config_integration(self):
        """Test integration between ModelConfig and BaseModel."""
        # Test with YAML configuration structure
        yaml_content = """
models:
  lstm:
    vocab_size: 2000
    embed_dim: 128
    hidden_dim: 256
    output_dim: 3
    num_layers: 2
    dropout: 0.2
    bidirectional: true
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(yaml_content.strip())

            # Load config and create model
            config = ModelConfig.from_yaml(config_path, "lstm")
            model = SimpleTestModel(config)

            # Verify integration
            assert model.config.vocab_size == 2000
            assert model.config.embed_dim == 128
            assert model.config.output_dim == 3
            assert model.embedding.num_embeddings == 2000
            assert model.embedding.embedding_dim == 128

    def test_model_with_different_configs(self):
        """Test model with various configuration combinations."""
        configs = [
            # Binary classification
            ModelConfig(vocab_size=1000, embed_dim=64, hidden_dim=128, output_dim=2),
            # Multi-class classification
            ModelConfig(vocab_size=5000, embed_dim=256, hidden_dim=512, output_dim=5),
            # Large model
            ModelConfig(
                vocab_size=20000,
                embed_dim=300,
                hidden_dim=1024,
                output_dim=2,
                num_layers=3,
            ),
        ]

        for config in configs:
            model = SimpleTestModel(config)

            # Test forward pass with each config
            batch_size, seq_len = 2, 8
            input_ids = torch.randint(
                0, config.vocab_size, (batch_size, seq_len), device=model.device
            )

            with torch.no_grad():
                logits = model(input_ids)

            assert logits.shape == (batch_size, config.output_dim)

            # Test parameter count is reasonable
            param_count = model.count_parameters()
            assert param_count > 0

            # Large models should have more parameters
            if config.embed_dim > 256:
                assert param_count > 100000  # Arbitrary threshold for "large" model

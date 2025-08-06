"""Tests for YAML-based model factory."""

import tempfile
from pathlib import Path

import torch

from src.models.factory import create_model_from_yaml

YAML_CONTENT = """
lstm:
  vocab_size: 500
  embed_dim: 32
  hidden_dim: 64
  output_dim: 2
  num_layers: 1
  dropout: 0.2
  bidirectional: false
"""


def test_create_lstm_from_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "model.yaml"
        cfg.write_text(YAML_CONTENT)
        model = create_model_from_yaml(cfg, "lstm")
        assert model.config.vocab_size == 500
        batch = torch.randint(0, 500, (2, 10), device=model.device)
        logits = model(batch)
        assert logits.shape == (2, 2)

"""Factory helpers to create models directly from YAML configuration files.

The YAML structure is expected to contain a top-level key per model type
(e.g. ``lstm``, ``cnn``, ``transformer``) whose value is a mapping of
hyperparameters compatible with the corresponding *Config* classes.

Example ``model_config.yaml``::

    lstm:
      vocab_size: 12000
      embed_dim: 128
      hidden_dim: 256
      output_dim: 2
      num_layers: 2
      dropout: 0.3
      bidirectional: true

    cnn:
      vocab_size: 12000
      embed_dim: 128
      num_filters: 100
      filter_sizes: [3, 4, 5]
      dropout: 0.5

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from .cnn_model import CNNConfig, CNNModel
from .lstm_model import LSTMConfig, LSTMModel
from .transformer_model import TransformerConfig, TransformerModel

logger = logging.getLogger(__name__)

_MODEL_REGISTRY = {
    "lstm": (LSTMConfig, LSTMModel),
    "cnn": (CNNConfig, CNNModel),
    "transformer": (TransformerConfig, TransformerModel),
}

__all__ = ["create_model_from_yaml"]


def create_model_from_yaml(config_path: str | Path, model_type: str) -> Any:
    """Instantiate a model directly from YAML configuration.

    Args:
        config_path: Path to YAML file.
        model_type: One of ``lstm``, ``cnn``, ``transformer``.
    """
    model_type = model_type.lower()
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(f"Unsupported model_type: {model_type}")

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    with config_path.open("r") as f:
        data: Mapping[str, Dict[str, Any]] = yaml.safe_load(f) or {}

    params = data.get(model_type)
    if params is None:
        raise ValueError(f"No section '{model_type}' found in {config_path}")

    ConfigCls, ModelCls = _MODEL_REGISTRY[model_type]
    config = ConfigCls(**params)  # type: ignore[arg-type]
    model = ModelCls(config)  # type: ignore[call-arg]
    return model

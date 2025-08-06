"""
Model architecture module for sentiment analysis.

This module contains implementations of various neural network architectures:
- Base model class with common functionality
- LSTM-based sentiment analysis models
- CNN-based sentiment analysis models
- Transformer-based sentiment analysis models
- Embedding layers and utilities
"""

from .base_model import BaseModel, ModelConfig
from .cnn_model import CNNConfig, CNNModel, create_cnn_model
from .embeddings import build_embedding_matrix, load_text_embeddings
from .factory import create_model_from_yaml
from .lstm_model import (AttentionLayer, AttentionLSTM, BidirectionalLSTM,
                         LSTMConfig, LSTMModel, create_attention_lstm,
                         create_bidirectional_lstm, create_lstm_model)
from .regularization import LabelSmoothingCrossEntropy
from .transformer_model import (TransformerConfig, TransformerModel,
                                create_transformer_model)

__all__ = [
    "BaseModel",
    "ModelConfig",
    "LSTMModel",
    "LSTMConfig",
    "BidirectionalLSTM",
    "AttentionLSTM",
    "AttentionLayer",
    "create_lstm_model",
    "create_bidirectional_lstm",
    "create_attention_lstm",
    "CNNModel",
    "CNNConfig",
    "create_cnn_model",
    "TransformerModel",
    "TransformerConfig",
    "create_transformer_model",
    "load_text_embeddings",
    "build_embedding_matrix",
    "LabelSmoothingCrossEntropy",
    "create_model_from_yaml",
]

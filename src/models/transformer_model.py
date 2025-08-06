"""
Transformer-based sentiment analysis model implementation.

Uses PyTorch's ``nn.TransformerEncoder`` with sinusoidal positional
encoding.  Keeps the design consistent with the project BaseModel so
training utilities remain identical for all architectures.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .base_model import BaseModel, ModelConfig

logger = logging.getLogger(__name__)


class TransformerConfig(ModelConfig):
    """Configuration for Transformer models."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 2,
        max_seq_len: int = 512,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        super().__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            model_type="transformer",
            **kwargs,
        )
        self._validate_transformer()

    def _validate_transformer(self) -> None:
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (no learnable parameters)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:  # (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModel(BaseModel):
    """Text classification model using Transformer encoder."""

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.config: TransformerConfig = config  # type narrowing

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            config.embed_dim, config.dropout, config.max_seq_len
        )

        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )
        self.transformer_encoder.mask_check = False

        # Re-initialise classifier (input dim = embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim),
        )
        self.to(self.device)

        logger.info(
            "Initialized Transformer model with %d parameters", self.count_parameters()
        )

    def _get_classifier_input_dim(self) -> int:
        return self.config.embed_dim

    def forward(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:  # noqa: D401
        # Embed
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        x = self.pos_encoder(x)

        # Build src_key_padding_mask for transformer (True = padding)
        padding_mask = None
        if attention_mask is not None:
            padding_mask = attention_mask == 0  # invert

        encoded = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Pooling: use [CLS]-style by taking representation of first token
        pooled = encoded[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# ----------------------------------------------------------------------
# Factory helper
# ----------------------------------------------------------------------


def create_transformer_model(
    vocab_size: int,
    embed_dim: int = 128,
    num_heads: int = 8,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    output_dim: int = 2,
    max_seq_len: int = 512,
    **kwargs,
) -> TransformerModel:
    config = TransformerConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        output_dim=output_dim,
        max_seq_len=max_seq_len,
        **kwargs,
    )
    return TransformerModel(config)

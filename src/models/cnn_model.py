"""
CNN-based sentiment analysis model implementation.

Implements a TextCNN (Kim 2014) architecture with configurable filter
sizes and number of filters.  The model leverages the common utilities
provided by `BaseModel` (embedding layer, dropout, classifier, device
management, etc.).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_model import BaseModel, ModelConfig

logger = logging.getLogger(__name__)


class CNNConfig(ModelConfig):
    """Configuration class for CNN models."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_filters: int = 100,
        filter_sizes: Optional[Sequence[int]] = None,
        output_dim: int = 2,
        dropout: float = 0.5,
        **kwargs,
    ) -> None:
        """Create a CNN configuration.

        Args:
            vocab_size: Size of vocabulary.
            embed_dim: Embedding dimension.
            num_filters: Number of convolutional filters *per* filter size.
            filter_sizes: Iterable of kernel sizes. Defaults to ``(3, 4, 5)``.
            output_dim: Number of classes.
            dropout: Dropout applied after convolution/pooling.
            **kwargs: Additional arguments forwarded to :class:`ModelConfig`.
        """
        if filter_sizes is None:
            filter_sizes = (3, 4, 5)
        filter_sizes = tuple(int(fs) for fs in filter_sizes)
        self.num_filters = num_filters
        self.filter_sizes: tuple[int, ...] = filter_sizes  # type: ignore[assignment]

        hidden_dim = num_filters * len(filter_sizes)
        super().__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            model_type="cnn",
            **kwargs,
        )

        self._validate_cnn_config()

    # ---------------------------------------------------------------------
    # Validation helpers
    # ---------------------------------------------------------------------
    def _validate_cnn_config(self) -> None:
        if self.num_filters <= 0:
            raise ValueError("num_filters must be positive")
        if not self.filter_sizes:
            raise ValueError("filter_sizes must contain at least one element")
        if any(fs <= 0 for fs in self.filter_sizes):
            raise ValueError("All filter sizes must be positive integers")


class CNNModel(BaseModel):
    """TextCNN model for sentence-level classification."""

    def __init__(self, config: CNNConfig):
        # Initialise BaseModel (creates embedding / classifier shell)
        self.config: CNNConfig = config  # narrow type for IDEs
        super().__init__(config)

        # Convolution layers: (batch, channel=1, seq_len, embed_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=config.num_filters,
                    kernel_size=(fs, config.embed_dim),
                )
                for fs in config.filter_sizes
            ]
        )

        # Re-initialise classifier with correct input dim (done *after* convs)
        self.classifier = nn.Sequential(
            nn.Linear(self._get_classifier_input_dim(), config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim),
        )
        self.to(self.device)

        logger.info(
            "Initialized CNN model with %d parameters",
            self.count_parameters(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_classifier_input_dim(
        self,
    ) -> int:  # noqa: D401 – short description sufficient
        """Return the concatenated feature dimension from all filters."""
        return self.config.num_filters * len(self.config.filter_sizes)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:  # noqa: D401
        """Compute logits for *input_ids*.

        Args:
            input_ids: LongTensor [batch, seq_len]
            attention_mask: *Unused* for CNN but kept for compatibility.
        """
        # Squeeze any extra singleton dimensions to ensure [batch, seq_len]
        while input_ids.dim() > 2 and input_ids.size(1) == 1:
            input_ids = input_ids.squeeze(1)

        # Embedding -> (batch, seq_len, embed_dim)
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)

        # Add channel dim: (batch, 1, seq_len, embed_dim)
        embedded = embedded.unsqueeze(1)

        # Convolution + ReLU + Max-over-time pooling
        conved: list[Tensor] = []
        for conv in self.convs:
            x = conv(embedded)  # (batch, num_filters, seq_len - fs + 1, 1)
            x = F.relu(x.squeeze(3))  # (batch, num_filters, seq_len - fs + 1)
            x = F.max_pool1d(x, x.size(2)).squeeze(2)  # (batch, num_filters)
            conved.append(x)

        # Concatenate along filter dimension
        cat = torch.cat(conved, dim=1)  # (batch, num_filters * len(filter_sizes))
        cat = self.dropout(cat)

        logits = self.classifier(cat)
        return logits


# ----------------------------------------------------------------------
# Factory helper
# ----------------------------------------------------------------------


def create_cnn_model(
    vocab_size: int,
    embed_dim: int = 128,
    num_filters: int = 100,
    filter_sizes: Sequence[int] | None = None,
    output_dim: int = 2,
    dropout: float = 0.5,
    **kwargs,
) -> CNNModel:
    """Factory for quickly building a CNN model with sensible defaults."""
    config = CNNConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_filters=num_filters,
        filter_sizes=filter_sizes,
        output_dim=output_dim,
        dropout=dropout,
        **kwargs,
    )
    return CNNModel(config)

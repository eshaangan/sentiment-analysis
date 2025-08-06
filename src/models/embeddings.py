"""Embedding loading and initialization utilities.

Supports loading pre-trained word embeddings (e.g. GloVe, word2vec text
format) and building an ``torch.Tensor`` embedding matrix aligned with a
project ``Vocabulary`` (or any ``word -> index`` mapping).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = [
    "load_text_embeddings",
    "build_embedding_matrix",
]


def load_text_embeddings(
    file_path: str | Path,
    *,
    expected_dim: int | None = None,
    encoding: str = "utf-8",
    max_tokens: int | None = None,
) -> Dict[str, np.ndarray]:
    """Load embeddings from a plain-text file (GloVe‐style).

    The file should contain whitespace-separated *token value\_1 … value\_d*
    per line.

    Args:
        file_path: Path to embedding text file.
        expected_dim: If given, validate that every vector has this size.
        encoding: Text encoding.
        max_tokens: If given, stop after reading *max_tokens* (for quick tests).

    Returns:
        ``dict`` mapping *word -> np.ndarray* of shape ``(dim,)``.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    embeddings: Dict[str, np.ndarray] = {}
    with file_path.open("r", encoding=encoding) as f:
        for i, line in enumerate(f):
            if max_tokens is not None and i >= max_tokens:
                break
            parts = line.rstrip().split()
            if len(parts) < 2:
                continue  # skip bad line
            token, *vector = parts
            vector_f = np.asarray(vector, dtype="float32")
            if expected_dim is not None and vector_f.size != expected_dim:
                logger.warning(
                    "Skipping %s due to dim mismatch (%d != %d)",
                    token,
                    vector_f.size,
                    expected_dim,
                )
                continue
            embeddings[token] = vector_f
    if expected_dim is None and embeddings:
        expected_dim = next(iter(embeddings.values())).size
    logger.info(
        "Loaded %d embeddings from %s (dim=%s)",
        len(embeddings),
        file_path,
        expected_dim,
    )
    return embeddings


def build_embedding_matrix(
    vocab: Mapping[str, int],
    embeddings: Mapping[str, Sequence[float]],
    *,
    embed_dim: int,
    unk_std: float = 0.05,
) -> Tensor:
    """Create embedding matrix aligned with *vocab*.

    Args:
        vocab: Mapping word -> index.
        embeddings: Pre-trained embeddings mapping.
        embed_dim: Expected dimensionality.
        unk_std: Standard deviation for random vectors for OOV words.

    Returns:
        ``torch.FloatTensor`` of shape ``(len(vocab), embed_dim)``.
    """
    matrix = np.random.normal(0.0, unk_std, size=(len(vocab), embed_dim)).astype(
        "float32"
    )
    oov = 0
    for word, idx in vocab.items():
        vec = embeddings.get(word)
        if vec is not None and len(vec) == embed_dim:
            matrix[idx] = np.asarray(vec, dtype="float32")
        else:
            oov += 1
    logger.info("Built embedding matrix - OOV tokens: %d / %d", oov, len(vocab))
    return torch.tensor(matrix)

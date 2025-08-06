"""Unit tests for embedding utilities."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from src.models.embeddings import build_embedding_matrix, load_text_embeddings


def _create_dummy_embedding_file(dim: int = 50) -> Path:
    words = ["the", "cat", "sat", "on", "mat"]
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        for w in words:
            vec = np.arange(dim, dtype="float32") / (dim * (words.index(w) + 1))
            vec_str = " ".join(f"{v:.4f}" for v in vec)
            tmp.write(f"{w} {vec_str}\n")
        return Path(tmp.name)


def test_load_text_embeddings():
    dim = 20
    emb_file = _create_dummy_embedding_file(dim)
    emb = load_text_embeddings(emb_file, expected_dim=dim)
    assert len(emb) == 5
    assert np.allclose(emb["cat"][0], 0.0)


def test_build_embedding_matrix():
    dim = 10
    emb_file = _create_dummy_embedding_file(dim)
    emb = load_text_embeddings(emb_file, expected_dim=dim)

    vocab = {"the": 0, "dog": 1, "sat": 2}
    matrix = build_embedding_matrix(vocab, emb, embed_dim=dim, unk_std=0.01)

    assert matrix.shape == (len(vocab), dim)
    # known word uses pretrained vector
    assert torch.allclose(matrix[0], torch.tensor(emb["the"]))
    # unknown word is non-zero (random)
    assert not torch.allclose(matrix[1], torch.zeros(dim))

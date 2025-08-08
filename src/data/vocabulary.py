"""
Vocabulary building and word-to-index mapping for sentiment analysis.
Handles frequency-based filtering and special token management.
"""

import json
import pickle
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any

# Avoid importing pandas at module import time to keep inference lightweight
pd = None  # type: ignore
import torch
from src.data.preprocessing import TextPreprocessor


class Vocabulary:
    """
    Vocabulary class for building and managing word-to-index mappings.

    Features:
    - Frequency-based word filtering
    - Special token handling (UNK, PAD, SOS, EOS)
    - Configurable vocabulary size limits
    - Text-to-sequence and sequence-to-text conversion
    - Vocabulary persistence (save/load)
    """

    # Special tokens
    PAD_TOKEN = "<PAD>"  # Padding token for sequence batching
    UNK_TOKEN = "<UNK>"  # Unknown word token
    SOS_TOKEN = "<SOS>"  # Start of sequence token
    EOS_TOKEN = "<EOS>"  # End of sequence token

    def __init__(
        self,
        max_vocab_size: Optional[int] = None,
        min_frequency: int = 2,
        include_special_tokens: bool = True,
        lowercase: bool = True,
    ):
        """
        Initialize vocabulary builder.

        Args:
            max_vocab_size: Maximum vocabulary size (None for unlimited)
            min_frequency: Minimum word frequency to include in vocabulary
            include_special_tokens: Whether to include special tokens
            lowercase: Whether to convert words to lowercase
        """
        self.max_vocab_size = max_vocab_size
        self.min_frequency = min_frequency
        self.include_special_tokens = include_special_tokens
        self.lowercase = lowercase

        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()
        self.is_built = False
        self.vocab_size = 0

        if self.include_special_tokens:
            self._add_special_tokens()

    def _add_special_tokens(self) -> None:
        """Add special tokens to vocabulary."""
        special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN,
        ]
        for token in special_tokens:
            if token not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[token] = idx
                self.idx_to_word[idx] = token
        self.vocab_size = len(self.word_to_idx)

    def build_from_texts(
        self,
        texts: List[str],
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> None:
        """
        Build vocabulary from a list of texts.

        Args:
            texts: List of text strings
            preprocessor: Optional text preprocessor to apply
        """
        for text in texts:
            tokens: List[str]
            if preprocessor:
                tokens = preprocessor.preprocess_and_tokenize(text)
            else:
                tokens = text.lower().split() if self.lowercase else text.split()
            if self.lowercase:
                tokens = [t.lower() for t in tokens]
            self.word_counts.update(tokens)
        self._build_vocabulary_from_counts()

    def _build_vocabulary_from_counts(self) -> None:
        """Build vocabulary from word counts."""
        filtered = [w for w, c in self.word_counts.items() if c >= self.min_frequency]
        filtered.sort(key=lambda w: self.word_counts[w], reverse=True)
        special_count = self._special_token_count()
        if self.max_vocab_size and len(filtered) > self.max_vocab_size - special_count:
            filtered = filtered[: self.max_vocab_size - special_count]
        for word in filtered:
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        self.vocab_size = len(self.word_to_idx)
        self.is_built = True

    def _special_token_count(self) -> int:
        """Return count of special tokens."""
        return 4 if self.include_special_tokens else 0

    def text_to_sequence(
        self,
        text: str,
        preprocessor: Optional[TextPreprocessor] = None,
        add_special_tokens: bool = False,
    ) -> List[int]:
        """
        Convert text to sequence of token indices.

        Args:
            text: Input text
            preprocessor: Optional text preprocessor
            add_special_tokens: Whether to add SOS/EOS tokens
        """
        if text is None:
            return []
        if not self.is_built:
            raise ValueError("Vocabulary not built yet. Call build_from_texts first.")
        if preprocessor:
            tokens = preprocessor.preprocess_and_tokenize(text)
        else:
            tokens = text.lower().split() if self.lowercase else text.split()
        if self.lowercase:
            tokens = [t.lower() for t in tokens]
        seq: List[int] = []
        if add_special_tokens:
            seq.append(self.word_to_idx[self.SOS_TOKEN])
        for t in tokens:
            if t not in self.word_to_idx:
                # Use UNK token for unknown words
                seq.append(self.word_to_idx[self.UNK_TOKEN])
            else:
                seq.append(self.word_to_idx[t])
        if add_special_tokens:
            seq.append(self.word_to_idx[self.EOS_TOKEN])
        return seq

    def sequence_to_text(
        self, sequence: List[int], remove_special_tokens: bool = True
    ) -> str:
        """
        Convert sequence of token indices back to text.
        """
        if not self.is_built:
            raise ValueError("Vocabulary not built yet. Call build_from_texts first.")
        tokens: List[str] = []
        specials = {self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN}
        # Flatten sequence if it's a list of lists
        if isinstance(sequence, dict):
            sequence = sequence.get("input_ids", [])
            if isinstance(sequence, torch.Tensor):
                sequence = sequence.flatten().tolist()
        if sequence and isinstance(sequence[0], list):
            sequence = [item for sublist in sequence for item in sublist]

        for idx in sequence:
            word = self.idx_to_word.get(idx, self.UNK_TOKEN)
            if remove_special_tokens and word in specials:
                continue
            tokens.append(word)
        return " ".join(tokens)

    def contains_word(self, word: str) -> bool:
        return (word.lower() if self.lowercase else word) in self.word_to_idx

    def get_word_frequency(self, word: str) -> int:
        return self.word_counts.get(word.lower() if self.lowercase else word, 0)

    def get_most_common_words(self, n: int = 10) -> List[tuple]:
        return self.word_counts.most_common(n)

    def get_vocabulary_stats(self) -> Dict[str, Union[int, float]]:
        if not self.is_built:
            return {"status": "not_built"}
        return {
            "status": "built",
            "vocab_size": self.vocab_size,
            "special_tokens": self._special_token_count(),
            "content_words": self.vocab_size - self._special_token_count(),
            "min_frequency": self.min_frequency,
            "max_vocab_size": self.max_vocab_size,
            "total_unique_words": len(self.word_counts),
            "total_word_count": sum(self.word_counts.values()),
            "avg_frequency": (
                sum(self.word_counts.values()) / len(self.word_counts)
                if self.word_counts
                else 0
            ),
            "most_common_word": (
                self.word_counts.most_common(1)[0] if self.word_counts else None
            ),
        }

    def save(self, path: Union[str, Path]) -> None:
        data = {
            "word_to_idx": self.word_to_idx,
            "idx_to_word": self.idx_to_word,
            "word_counts": dict(self.word_counts),
            "max_vocab_size": self.max_vocab_size,
            "min_frequency": self.min_frequency,
            "include_special_tokens": self.include_special_tokens,
            "lowercase": self.lowercase,
            "is_built": self.is_built,
            "vocab_size": self.vocab_size,
        }
        p = Path(path)
        if p.suffix == ".json":
            p.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        else:
            with open(p if p.suffix == ".pkl" else p.with_suffix(".pkl"), "wb") as f:
                pickle.dump(data, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Vocabulary":
        p = Path(path)
        if p.suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            data = pickle.loads(p.read_bytes())
        vocab = cls(
            max_vocab_size=data["max_vocab_size"],
            min_frequency=data["min_frequency"],
            include_special_tokens=data["include_special_tokens"],
            lowercase=data["lowercase"],
        )
        vocab.word_to_idx = data["word_to_idx"]
        vocab.idx_to_word = {int(k): v for k, v in data["idx_to_word"].items()}
        vocab.word_counts = Counter(data["word_counts"])
        vocab.is_built = data["is_built"]
        vocab.vocab_size = data["vocab_size"]
        return vocab


def create_vocabulary_from_data(
    train_csv_path: Union[str, Path],
    test_csv_path: Union[str, Path],
    text_column: str = "review",
    max_vocab_size: int = 10000,
    min_frequency: int = 2,
    preprocessor: Optional[TextPreprocessor] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Vocabulary:
    """Create vocabulary from training and test data combined."""
    global pd
    if pd is None:
        import pandas as pd  # type: ignore
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Combine both datasets for vocabulary building
    all_texts = (
        train_df[text_column].astype(str).tolist()
        + test_df[text_column].astype(str).tolist()
    )

    vocab = Vocabulary(max_vocab_size, min_frequency)
    vocab.build_from_texts(all_texts, preprocessor)
    if save_path:
        vocab.save(save_path)
    return vocab


# Add DataFrame builder method


def create_merged_vocabulary(
    train_csv_path: Union[str, Path],
    test_csv_path: Union[str, Path],
    text_column: str = "review",
    max_vocab_size: int = 10000,
    min_frequency: int = 2,
    preprocessor: Optional[TextPreprocessor] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Vocabulary:
    """Backward-compat wrapper expected by old tests.

    Calls create_vocabulary_from_data with same parameters.
    """
    return create_vocabulary_from_data(
        train_csv_path,
        test_csv_path,
        text_column=text_column,
        max_vocab_size=max_vocab_size,
        min_frequency=min_frequency,
        preprocessor=preprocessor,
        save_path=save_path,
    )


def _build_from_dataframe(
    self,
    df: Any,
    text_column: str,
    preprocessor: Optional[TextPreprocessor] = None,
) -> None:
    texts = df[text_column].astype(str).tolist()
    self.build_from_texts(texts, preprocessor)


Vocabulary.build_from_dataframe = _build_from_dataframe

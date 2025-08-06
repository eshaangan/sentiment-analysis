"""
Tokenization and text-to-sequence conversion utilities for sentiment analysis.
Handles sequence padding, truncation, and tensor conversion for PyTorch models.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from src.data.preprocessing import TextPreprocessor
from src.data.vocabulary import Vocabulary


class Tokenizer:
    """
    Advanced tokenizer for converting text to padded sequences for PyTorch models.

    Features:
    - Text-to-sequence conversion with vocabulary
    - Sequence padding and truncation
    - Batch processing with variable lengths
    - Tensor conversion for PyTorch
    - Special token handling
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        preprocessor: Optional[TextPreprocessor] = None,
        max_length: Optional[int] = None,
        padding: str = "max_length",
        truncation: bool = True,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
    ):
        """
        Initialize tokenizer with vocabulary and processing options.

        Args:
            vocabulary: Vocabulary instance for text-to-index conversion
            preprocessor: Optional text preprocessor
            max_length: Maximum sequence length (None for no limit)
            padding: Padding strategy ("max_length", "longest", "do_not_pad")
            truncation: Whether to truncate sequences exceeding max_length
            add_special_tokens: Whether to add SOS/EOS tokens
            return_tensors: Format for returned tensors ("pt" for PyTorch)
        """
        self.vocabulary = vocabulary
        self.preprocessor = preprocessor
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.add_special_tokens = add_special_tokens
        self.return_tensors = return_tensors

        # Validate vocabulary
        if not vocabulary.is_built:
            raise ValueError("Vocabulary must be built before using tokenizer")

        # Get special token indices
        self.pad_token_id = vocabulary.word_to_idx.get(vocabulary.PAD_TOKEN, 0)
        self.unk_token_id = vocabulary.word_to_idx.get(vocabulary.UNK_TOKEN, 1)
        self.sos_token_id = vocabulary.word_to_idx.get(vocabulary.SOS_TOKEN, 2)
        self.eos_token_id = vocabulary.word_to_idx.get(vocabulary.EOS_TOKEN, 3)

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: Optional[bool] = None,
        max_length: Optional[int] = None,
        padding: Optional[str] = None,
        truncation: Optional[bool] = None,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], Tensor, Dict[str, Tensor]]:
        """
        Encode text(s) to sequence(s) of token indices.

        Args:
            text: Single text or list of texts to encode
            add_special_tokens: Override default special token setting
            max_length: Override default max length
            padding: Override default padding strategy
            truncation: Override default truncation setting
            return_tensors: Override default tensor format

        Returns:
            Encoded sequence(s) as indices, tensors, or dict with metadata
        """
        # Use provided parameters or fall back to instance defaults
        add_special_tokens = (
            add_special_tokens
            if add_special_tokens is not None
            else self.add_special_tokens
        )
        max_length = max_length if max_length is not None else self.max_length
        padding = padding if padding is not None else self.padding
        truncation = truncation if truncation is not None else self.truncation
        return_tensors = (
            return_tensors if return_tensors is not None else self.return_tensors
        )

        # Handle single text vs batch
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]

        # Convert texts to sequences
        sequences = []
        for txt in texts:
            sequence = self.vocabulary.text_to_sequence(
                txt,
                preprocessor=self.preprocessor,
                add_special_tokens=add_special_tokens,
            )
            sequences.append(sequence)

        # Apply truncation if needed
        if truncation and max_length is not None:
            sequences = [
                self._truncate_sequence(seq, max_length, add_special_tokens)
                for seq in sequences
            ]

        # Apply padding if needed
        if padding != "do_not_pad":
            sequences = self._pad_sequences(sequences, max_length, padding)

        # Convert to tensors if requested
        if return_tensors == "pt":
            if padding == "do_not_pad" and is_batch:
                sequences = [torch.tensor(s, dtype=torch.long) for s in sequences]
                return sequences
            # If we encoded a single sequence, sequences will be a list of ints.
            # Wrap it only if it is NOT already a list of lists.
            if not is_batch and sequences and isinstance(sequences[0], int):
                sequences = [sequences]
            tensor_sequences = torch.tensor(sequences, dtype=torch.long)
            attention_mask = (tensor_sequences != self.pad_token_id).long()
            return {"input_ids": tensor_sequences, "attention_mask": attention_mask}
        return sequences if is_batch else sequences[0]

    def _truncate_sequence(
        self, sequence: List[int], max_length: int, has_special_tokens: bool
    ) -> List[int]:
        """
        Truncate sequence to maximum length while preserving special tokens.

        Args:
            sequence: Input sequence of token indices
            max_length: Maximum allowed length
            has_special_tokens: Whether sequence has SOS/EOS tokens

        Returns:
            Truncated sequence
        """
        if len(sequence) <= max_length:
            return sequence

        if has_special_tokens and len(sequence) > 2:
            # Preserve SOS and EOS tokens
            sos_token = sequence[0] if sequence[0] == self.sos_token_id else None
            eos_token = sequence[-1] if sequence[-1] == self.eos_token_id else None

            if sos_token is not None and eos_token is not None:
                # Truncate middle part
                middle_length = max_length - 2
                truncated = [sos_token] + sequence[1 : middle_length + 1] + [eos_token]
            elif sos_token is not None:
                # Truncate from end, preserve SOS
                truncated = (
                    sequence[: max_length - 1] + [eos_token]
                    if eos_token
                    else sequence[:max_length]
                )
            elif eos_token is not None:
                # Truncate from start, preserve EOS
                truncated = (
                    [sos_token] + sequence[-(max_length - 1) :]
                    if sos_token
                    else sequence[-max_length:]
                )
            else:
                # No special tokens to preserve
                truncated = sequence[:max_length]
        else:
            # Simple truncation
            truncated = sequence[:max_length]

        return truncated

    def _pad_sequences(
        self,
        sequences: List[List[int]],
        max_length: Optional[int],
        padding_strategy: str,
    ) -> List[List[int]]:
        """
        Pad sequences according to strategy.

        Args:
            sequences: List of sequences to pad
            max_length: Maximum length for padding
            padding_strategy: "max_length", "longest", or "do_not_pad"

        Returns:
            Padded sequences
        """
        if padding_strategy == "do_not_pad":
            return sequences

        # Determine target length
        if padding_strategy == "max_length":
            if max_length is None:
                raise ValueError(
                    "max_length must be specified for max_length padding"
                )
            target_length = max_length
        elif padding_strategy == "longest":
            target_length = max(len(seq) for seq in sequences) if sequences else 0
        else:
            raise ValueError(f"Unsupported padding strategy: {padding_strategy}")

        # Pad sequences
        padded_sequences = []
        for sequence in sequences:
            if len(sequence) < target_length:
                # Pad with PAD tokens
                padded = sequence + [self.pad_token_id] * (
                    target_length - len(sequence)
                )
            else:
                padded = sequence
            padded_sequences.append(padded)

        return padded_sequences

    def decode(
        self, token_ids: Union[List[int], Tensor], skip_special_tokens: bool = True
    ) -> str:
        """
        Decode sequence of token indices back to text.

        Args:
            token_ids: Sequence of token indices
            skip_special_tokens: Whether to remove special tokens

        Returns:
            Decoded text string
        """
        # Convert tensor to list if needed
        if isinstance(token_ids, Tensor):
            token_ids = token_ids.tolist()

        # Use vocabulary to convert back to text
        return self.vocabulary.sequence_to_text(
            token_ids, remove_special_tokens=skip_special_tokens
        )

    def batch_decode(
        self,
        batch_token_ids: Union[List[List[int]], Tensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode batch of sequences back to texts.

        Args:
            batch_token_ids: Batch of token id sequences
            skip_special_tokens: Whether to remove special tokens

        Returns:
            List of decoded text strings
        """
        # Convert tensor to list if needed
        if isinstance(batch_token_ids, Tensor):
            batch_token_ids = batch_token_ids.tolist()

        return [
            self.decode(token_ids, skip_special_tokens) for token_ids in batch_token_ids
        ]

    def get_sequence_length(self, text: str) -> int:
        """
        Get the length of encoded sequence for given text.

        Args:
            text: Input text

        Returns:
            Length of encoded sequence
        """
        sequence = self.vocabulary.text_to_sequence(
            text,
            preprocessor=self.preprocessor,
            add_special_tokens=self.add_special_tokens,
        )
        return len(sequence)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocabulary.vocab_size

    def get_special_tokens_dict(self) -> Dict[str, int]:
        """Get mapping of special tokens to their indices."""
        return {
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "sos_token_id": self.sos_token_id,
            "eos_token_id": self.eos_token_id,
        }


class SequenceCollator:
    """
    Collator for creating batches of sequences with padding for DataLoader.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_length: Optional[int] = None,
        padding: str = "longest",
        return_tensors: str = "pt",
    ):
        """
        Initialize sequence collator.

        Args:
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            padding: Padding strategy
            return_tensors: Tensor format to return
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.return_tensors = return_tensors

    def __call__(self, batch: List[Dict]) -> Dict[str, Tensor]:
        """
        Collate batch of samples into padded tensors.

        Args:
            batch: List of samples, each with 'text' and optionally 'label'

        Returns:
            Batch dictionary with padded tensors
        """
        # If items are already tokenized (they contain 'input_ids'), simply stack them
        if batch and isinstance(batch[0], dict) and "input_ids" in batch[0]:
            input_ids = torch.stack([item["input_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
            batch_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "labels" in batch[0]:
                batch_dict["labels"] = torch.stack([item["labels"] for item in batch])
            return batch_dict

        # Otherwise, extract raw texts and (optionally) labels and perform tokenization here
        texts = [
            item.get("review") or item.get("text")
            for item in batch
            if item is not None and (item.get("review") or item.get("text"))
        ]
        labels = None
        if batch and batch[0] is not None:
            if "sentiment" in batch[0]:
                labels = torch.tensor(
                    [item["sentiment"] for item in batch if item is not None],
                    dtype=torch.long,
                )
            elif "label" in batch[0]:
                labels = torch.tensor(
                    [item["label"] for item in batch if item is not None], dtype=torch.long
                )

        encoded = self.tokenizer.encode(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            return_tensors=self.return_tensors,
        )

        batch_dict = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if labels is not None:
            batch_dict["labels"] = labels
        return batch_dict


def create_tokenizer(
    vocabulary: Vocabulary,
    preprocessor: Optional[TextPreprocessor] = None,
    max_length: int = 512,
    padding: str = "max_length",
    add_special_tokens: bool = True,
) -> Tokenizer:
    """
    Create a tokenizer with common settings for sentiment analysis.

    Args:
        vocabulary: Built vocabulary instance
        preprocessor: Text preprocessor
        max_length: Maximum sequence length
        padding: Padding strategy
        add_special_tokens: Whether to add special tokens

    Returns:
        Configured tokenizer instance
    """
    return Tokenizer(
        vocabulary=vocabulary,
        preprocessor=preprocessor,
        max_length=max_length,
        padding=padding,
        truncation=True,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )


def analyze_sequence_lengths(
    texts: List[str], tokenizer: Tokenizer, percentiles: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Analyze sequence length distribution to help choose max_length.

    Args:
        texts: List of texts to analyze
        tokenizer: Tokenizer to use for encoding
        percentiles: Percentiles to compute (default: [50, 90, 95, 99])

    Returns:
        Dictionary with length statistics
    """
    if percentiles is None:
        percentiles = [50, 90, 95, 99]

    # Get sequence lengths
    lengths = [tokenizer.get_sequence_length(text) for text in texts]

    # Compute statistics
    import numpy as np

    stats = {
        "count": len(lengths),
        "mean": np.mean(lengths),
        "std": np.std(lengths),
        "min": np.min(lengths),
        "max": np.max(lengths),
    }

    # Add percentiles
    for p in percentiles:
        stats[f"percentile_{p}"] = np.percentile(lengths, p)

    return stats


def create_sequences_from_texts(
    texts: List[str],
    labels: Optional[List[int]] = None,
    tokenizer: Optional[Tokenizer] = None,
    vocabulary: Optional[Vocabulary] = None,
    preprocessor: Optional[TextPreprocessor] = None,
    max_length: int = 512,
    padding: str = "max_length",
    add_special_tokens: bool = True,
) -> Dict[str, Tensor]:
    """
    Convert texts to padded sequences ready for model training.

    Args:
        texts: List of input texts
        labels: Optional list of labels
        tokenizer: Pre-configured tokenizer (if not provided, will create one)
        vocabulary: Vocabulary for tokenizer creation
        preprocessor: Text preprocessor
        max_length: Maximum sequence length
        padding: Padding strategy
        add_special_tokens: Whether to add special tokens

    Returns:
        Dictionary with input_ids, attention_mask, and optionally labels
    """
    # Create tokenizer if not provided
    if tokenizer is None:
        if vocabulary is None:
            raise ValueError("Either tokenizer or vocabulary must be provided")
        tokenizer = create_tokenizer(
            vocabulary=vocabulary,
            preprocessor=preprocessor,
            max_length=max_length,
            padding=padding,
            add_special_tokens=add_special_tokens,
        )

    # Encode texts
    encoded = tokenizer.encode(texts, return_tensors="pt")

    # Add labels if provided
    if labels is not None:
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)

    return encoded

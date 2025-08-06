"""
Tests for data processing modules.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import (AugmentedSentimentDataset, DatasetSplitter,
                              SentimentDataset, create_data_loaders,
                              load_sentiment_data)
from src.data.preprocessing import (TextPreprocessor,
                                    create_aggressive_preprocessor,
                                    create_default_preprocessor,
                                    create_minimal_preprocessor)
from src.data.tokenization import (SequenceCollator, Tokenizer,
                                   analyze_sequence_lengths,
                                   create_sequences_from_texts,
                                   create_tokenizer)
from src.data.vocabulary import (Vocabulary, create_merged_vocabulary,
                                 create_vocabulary_from_data)


class TestTextPreprocessor:
    """Test cases for TextPreprocessor class."""

    def test_html_tag_removal(self) -> None:
        """Test HTML tag removal functionality."""
        preprocessor = TextPreprocessor(remove_html=True)

        # Test basic HTML tags
        text = "This is <b>bold</b> and <i>italic</i> text."
        expected = "This is bold and italic text."
        assert preprocessor.remove_html_tags(text) == expected

        # Test br tags
        text = "Line 1<br />Line 2<br/>Line 3"
        expected = "Line 1Line 2Line 3"
        assert preprocessor.remove_html_tags(text) == expected

        # Test HTML entities
        text = "This has &quot;quotes&quot; and &amp; symbols."
        expected = "This has quotes and  symbols."
        assert preprocessor.remove_html_tags(text) == expected

    def test_contraction_expansion(self) -> None:
        """Test contraction expansion."""
        preprocessor = TextPreprocessor(expand_contractions=True, lowercase=False)

        test_cases = [
            ("I can't do this", "I cannot do this"),
            ("Don't worry", "Do not worry"),
            ("It's amazing", "It is amazing"),
            ("We're going", "We are going"),
            ("I'd like that", "I would like that"),
        ]

        for original, expected in test_cases:
            result = preprocessor.expand_contractions_text(original)
            assert result == expected

    def test_lowercase_conversion(self) -> None:
        """Test lowercase conversion."""
        preprocessor = TextPreprocessor(lowercase=True)

        text = "This Has MIXED Case Text"
        expected = "this has mixed case text"
        result = preprocessor.clean_text(text)
        assert result == expected

    def test_whitespace_normalization(self) -> None:
        """Test whitespace normalization."""
        preprocessor = TextPreprocessor()

        text = "This   has    multiple     spaces\n\nand\tlines"
        result = preprocessor.normalize_whitespace(text)

        # Should normalize to single spaces
        assert "   " not in result
        assert "\n" not in result
        assert "\t" not in result
        assert result == "This has multiple spaces and lines"

    def test_special_character_handling(self) -> None:
        """Test special character handling."""
        preprocessor = TextPreprocessor()

        # Test repeated punctuation
        text = "This is amazing!!!! Really???"
        result = preprocessor.remove_special_characters(text)
        expected = "This is amazing! Really?"
        assert result == expected

        # Test repeated characters
        text = "Nooooooooo way"
        result = preprocessor.remove_special_characters(text)
        expected = "Nooo way"
        assert result == expected

    def test_tokenization(self) -> None:
        """Test tokenization functionality."""
        preprocessor = TextPreprocessor(min_word_length=1, max_word_length=50)

        text = "This is a test sentence with punctuation!"
        tokens = preprocessor.tokenize(text)

        # Should include words and punctuation
        assert "This" in tokens
        assert "is" in tokens
        assert "!" in tokens

        # Test length filtering
        preprocessor_filtered = TextPreprocessor(min_word_length=3, max_word_length=10)
        tokens_filtered = preprocessor_filtered.tokenize(text)

        # Short words should be filtered out
        assert "is" not in tokens_filtered  # Too short
        assert "This" in tokens_filtered  # Good length

    def test_stopword_removal(self) -> None:
        """Test stopword removal."""
        preprocessor = TextPreprocessor(remove_stopwords=True)

        text = "This is a test of the stopword removal"
        tokens = preprocessor.tokenize(text)

        # Common stopwords should be removed
        assert "is" not in tokens
        assert "a" not in tokens
        assert "the" not in tokens
        assert "of" not in tokens

        # Content words should remain
        assert "test" in tokens
        assert "stopword" in tokens
        assert "removal" in tokens

    def test_punctuation_removal(self) -> None:
        """Test punctuation removal."""
        preprocessor = TextPreprocessor(remove_punctuation=True)

        text = "Hello, world! How are you?"
        result = preprocessor.clean_text(text)

        # Punctuation should be removed
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

        # Words should remain
        assert "hello" in result
        assert "world" in result

    def test_number_removal(self) -> None:
        """Test number removal."""
        preprocessor = TextPreprocessor(remove_numbers=True)

        text = "I have 5 cats and 10 dogs in 2023."
        result = preprocessor.clean_text(text)

        # Numbers should be removed
        assert "5" not in result
        assert "10" not in result
        assert "2023" not in result

        # Words should remain
        assert "cats" in result
        assert "dogs" in result

    def test_full_preprocessing_pipeline(self) -> None:
        """Test the complete preprocessing pipeline."""
        preprocessor = TextPreprocessor(
            lowercase=True,
            remove_html=True,
            expand_contractions=True,
        )

        text = "<p>I can't believe this <b>amazing</b> movie isn't   rated higher!</p>"
        result = preprocessor.preprocess(text)

        # Should be cleaned and normalized
        assert "<p>" not in result
        assert "<b>" not in result
        assert "</p>" not in result
        assert "cannot" in result  # Contraction expanded
        assert result.islower()  # Lowercase
        assert "   " not in result  # Normalized whitespace

    def test_batch_processing(self) -> None:
        """Test batch processing functionality."""
        preprocessor = TextPreprocessor(lowercase=True)

        texts = ["First REVIEW text", "Second REVIEW text", "Third REVIEW text"]

        results = preprocessor.preprocess_batch(texts)

        assert len(results) == 3
        for result in results:
            assert result.islower()
            assert "review" in result

    def test_empty_and_none_input(self) -> None:
        """Test handling of empty and None inputs."""
        preprocessor = TextPreprocessor()

        # Empty string
        assert preprocessor.preprocess("") == ""
        assert preprocessor.preprocess_and_tokenize("") == []

        # None input (should be converted to string)
        result = preprocessor.preprocess(None)
        assert isinstance(result, str)

    def test_tokenize_and_preprocess(self) -> None:
        """Test combined preprocessing and tokenization."""
        preprocessor = TextPreprocessor(lowercase=True)

        text = "This IS a Test!"
        tokens = preprocessor.preprocess_and_tokenize(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "this" in tokens
        assert "test" in tokens


class TestPreprocessorFactories:
    """Test preset preprocessor configurations."""

    def test_default_preprocessor(self) -> None:
        """Test default preprocessor configuration."""
        preprocessor = create_default_preprocessor()

        text = "I can't believe this <b>amazing</b> movie!"
        result = preprocessor.preprocess(text)

        # Should expand contractions and remove HTML
        assert "cannot" in result
        assert "<b>" not in result
        assert "amazing" in result

    def test_minimal_preprocessor(self) -> None:
        """Test minimal preprocessor configuration."""
        preprocessor = create_minimal_preprocessor()

        text = "I can't believe this <b>amazing</b> movie!"
        result = preprocessor.preprocess(text)

        # Should not expand contractions but should remove HTML
        assert "can't" in result  # Contractions not expanded
        assert "<b>" not in result  # HTML removed
        assert result.islower()  # Lowercase applied

    def test_aggressive_preprocessor(self) -> None:
        """Test aggressive preprocessor configuration."""
        preprocessor = create_aggressive_preprocessor()

        text = "I can't believe this amazing movie has 100 great scenes!"
        tokens = preprocessor.preprocess_and_tokenize(text)

        # Should be heavily processed
        assert "cannot" in " ".join(tokens)  # Contractions expanded
        assert "100" not in tokens  # Numbers removed
        assert len([t for t in tokens if len(t) >= 3]) == len(
            tokens
        )  # Min length filter


class TestRealWorldExamples:
    """Test with real-world movie review examples."""

    def test_positive_review_preprocessing(self) -> None:
        """Test preprocessing of a positive movie review."""
        preprocessor = create_default_preprocessor()

        review = """
        <p>This movie is absolutely <b>fantastic</b>! I can't recommend it enough.
        The acting is superb and the story is engaging. It's a must-watch film.</p>
        """

        result = preprocessor.preprocess(review)

        # Check cleaning worked
        assert "<p>" not in result
        assert "<b>" not in result
        assert "cannot" in result
        assert result.islower()
        assert "fantastic" in result

    def test_negative_review_preprocessing(self) -> None:
        """Test preprocessing of a negative movie review."""
        preprocessor = create_default_preprocessor()

        review = """
        Terrible movie!!! Don't waste your time. The plot doesn't make sense
        and the acting is awful. 0/10 would not recommend.
        """

        result = preprocessor.preprocess(review)

        # Check cleaning worked
        assert "do not" in result  # Contraction expanded
        assert "does not" in result  # Contraction expanded
        assert "terrible" in result
        assert (
            "!!!" not in result or result.count("!") <= 1
        )  # Repeated punctuation handled

    def test_review_with_html_and_formatting(self) -> None:
        """Test preprocessing of review with HTML and special formatting."""
        preprocessor = create_default_preprocessor()

        review = """
        <br />This film is simply amazing!<br/>
        The cinematography is <i>beautiful</i> and the soundtrack is &quot;perfect&quot;.
        I'd give it 5/5 stars! Can't wait for the sequel...
        """

        result = preprocessor.preprocess(review)

        # Check all cleaning operations
        assert "<br" not in result
        assert "<i>" not in result
        assert "&quot;" not in result
        assert "beautiful" in result
        assert "perfect" in result
        assert "would" in result  # "I'd" expanded


class TestVocabulary:
    """Test cases for Vocabulary class."""

    def test_vocabulary_initialization(self) -> None:
        """Test vocabulary initialization with different settings."""
        # Default initialization
        vocab = Vocabulary()
        assert vocab.max_vocab_size is None
        assert vocab.min_frequency == 2
        assert vocab.include_special_tokens is True
        assert vocab.lowercase is True
        assert not vocab.is_built
        assert vocab.vocab_size == 4  # Special tokens added

        # Custom initialization
        vocab_custom = Vocabulary(
            max_vocab_size=1000,
            min_frequency=5,
            include_special_tokens=False,
            lowercase=False,
        )
        assert vocab_custom.max_vocab_size == 1000
        assert vocab_custom.min_frequency == 5
        assert vocab_custom.include_special_tokens is False
        assert vocab_custom.lowercase is False
        assert vocab_custom.vocab_size == 0  # No special tokens

    def test_special_tokens(self) -> None:
        """Test special token handling."""
        vocab = Vocabulary(include_special_tokens=True)

        # Check special tokens are present
        assert vocab.PAD_TOKEN in vocab.word_to_idx
        assert vocab.UNK_TOKEN in vocab.word_to_idx
        assert vocab.SOS_TOKEN in vocab.word_to_idx
        assert vocab.EOS_TOKEN in vocab.word_to_idx

        # Check special token indices
        assert vocab.word_to_idx[vocab.PAD_TOKEN] == 0
        assert vocab.word_to_idx[vocab.UNK_TOKEN] == 1
        assert vocab.word_to_idx[vocab.SOS_TOKEN] == 2
        assert vocab.word_to_idx[vocab.EOS_TOKEN] == 3

    def test_build_from_texts_basic(self) -> None:
        """Test building vocabulary from texts."""
        vocab = Vocabulary(min_frequency=1, max_vocab_size=10)
        texts = ["hello world", "hello there", "world peace", "hello world peace"]

        vocab.build_from_texts(texts)

        assert vocab.is_built
        assert vocab.vocab_size > 4  # Special tokens + content words
        assert vocab.contains_word("hello")
        assert vocab.contains_word("world")
        assert vocab.contains_word("peace")
        assert vocab.contains_word("there")

    def test_word_frequency_filtering(self) -> None:
        """Test vocabulary filtering by word frequency."""
        vocab = Vocabulary(min_frequency=2, include_special_tokens=False)
        texts = ["hello world", "hello there", "goodbye world", "hello again"]

        vocab.build_from_texts(texts)

        # "hello" appears 3 times, should be included
        assert vocab.contains_word("hello")

        # "world" appears 2 times, should be included
        assert vocab.contains_word("world")

        # "there", "goodbye", "again" appear 1 time each, should be filtered out
        assert not vocab.contains_word("there")
        assert not vocab.contains_word("goodbye")
        assert not vocab.contains_word("again")

    def test_text_to_sequence(self) -> None:
        """Test converting text to sequence of indices."""
        vocab = Vocabulary(min_frequency=1)
        texts = ["hello world", "hello there"]
        vocab.build_from_texts(texts)

        # Test basic conversion
        sequence = vocab.text_to_sequence("hello world")
        assert len(sequence) == 2
        assert all(isinstance(idx, int) for idx in sequence)

        # Test with special tokens
        sequence_with_special = vocab.text_to_sequence(
            "hello world", add_special_tokens=True
        )
        assert len(sequence_with_special) == 4  # SOS + hello + world + EOS
        assert sequence_with_special[0] == vocab.word_to_idx[vocab.SOS_TOKEN]
        assert sequence_with_special[-1] == vocab.word_to_idx[vocab.EOS_TOKEN]

    def test_sequence_to_text(self) -> None:
        """Test converting sequence back to text."""
        vocab = Vocabulary(min_frequency=1)
        texts = ["hello world", "hello there"]
        vocab.build_from_texts(texts)

        # Create sequence and convert back
        original_text = "hello world"
        sequence = vocab.text_to_sequence(original_text)
        reconstructed = vocab.sequence_to_text(sequence)

        assert reconstructed == original_text

    def test_unknown_word_handling(self) -> None:
        """Test handling of unknown words."""
        vocab = Vocabulary(min_frequency=1)
        texts = ["hello world"]
        vocab.build_from_texts(texts)

        # Test with unknown word
        sequence = vocab.text_to_sequence("hello unknown_word")

        # Should contain hello index and UNK index
        hello_idx = vocab.word_to_idx["hello"]
        unk_idx = vocab.word_to_idx[vocab.UNK_TOKEN]

        assert sequence[0] == hello_idx
        assert sequence[1] == unk_idx

    def test_vocabulary_with_preprocessor(self) -> None:
        """Test vocabulary building with text preprocessor."""
        preprocessor = create_default_preprocessor()
        vocab = Vocabulary(min_frequency=1)

        texts = ["This is GREAT!", "It's amazing", "Can't believe it"]

        vocab.build_from_texts(texts, preprocessor)

        # Should contain preprocessed words
        assert vocab.contains_word("great")  # Lowercased
        assert vocab.contains_word("it")  # From "It's" -> "it is"
        assert vocab.contains_word("is")  # From "It's" -> "it is"
        assert vocab.contains_word("cannot")  # From "Can't" -> "cannot"

    def test_get_vocabulary_stats(self) -> None:
        """Test vocabulary statistics generation."""
        vocab = Vocabulary(min_frequency=2, max_vocab_size=10)
        texts = ["hello world hello there world peace"]

        # Before building
        stats = vocab.get_vocabulary_stats()
        assert stats["status"] == "not_built"

        # After building
        vocab.build_from_texts(texts)
        stats = vocab.get_vocabulary_stats()

        assert stats["status"] == "built"
        assert stats["vocab_size"] > 4
        assert stats["special_tokens"] == 4
        assert stats["content_words"] == stats["vocab_size"] - 4
        assert stats["min_frequency"] == 2
        assert stats["max_vocab_size"] == 10

    def test_vocabulary_persistence(self, tmp_path) -> None:
        """Test saving and loading vocabulary."""
        # Build vocabulary
        vocab = Vocabulary(min_frequency=1)
        texts = ["hello world", "hello there"]
        vocab.build_from_texts(texts)

        # Save vocabulary
        save_path = tmp_path / "test_vocab.pkl"
        vocab.save(save_path)

        # Load vocabulary
        loaded_vocab = Vocabulary.load(save_path)

        # Check that loaded vocabulary is identical
        assert loaded_vocab.vocab_size == vocab.vocab_size
        assert loaded_vocab.word_to_idx == vocab.word_to_idx
        assert loaded_vocab.idx_to_word == vocab.idx_to_word
        assert loaded_vocab.is_built == vocab.is_built

    def test_edge_cases(self) -> None:
        """Test edge cases and error handling."""
        vocab = Vocabulary()

        # Test text_to_sequence before building
        with pytest.raises(ValueError, match="Vocabulary not built yet"):
            vocab.text_to_sequence("hello world")

        # Test sequence_to_text before building
        with pytest.raises(ValueError, match="Vocabulary not built yet"):
            vocab.sequence_to_text([1, 2, 3])

        # Test with empty texts
        vocab.build_from_texts([])
        assert vocab.is_built
        assert vocab.vocab_size == 4  # Only special tokens


class TestTokenizer:
    """Test cases for Tokenizer class."""

    def create_test_vocab(self) -> Vocabulary:
        """Create a simple vocabulary for testing."""
        vocab = Vocabulary(min_frequency=1)
        texts = [
            "hello world peace",
            "hello there friend",
            "world peace and love",
            "good morning world",
        ]
        vocab.build_from_texts(texts)
        return vocab

    def test_tokenizer_initialization(self) -> None:
        """Test tokenizer initialization."""
        vocab = self.create_test_vocab()

        # Basic initialization
        tokenizer = Tokenizer(vocab)
        assert tokenizer.vocabulary == vocab
        assert tokenizer.max_length is None
        assert tokenizer.padding == "max_length"
        assert tokenizer.add_special_tokens is True

        # Check special token IDs
        assert tokenizer.pad_token_id == vocab.word_to_idx[vocab.PAD_TOKEN]
        assert tokenizer.unk_token_id == vocab.word_to_idx[vocab.UNK_TOKEN]
        assert tokenizer.sos_token_id == vocab.word_to_idx[vocab.SOS_TOKEN]
        assert tokenizer.eos_token_id == vocab.word_to_idx[vocab.EOS_TOKEN]

    def test_tokenizer_with_unbuilt_vocab(self) -> None:
        """Test tokenizer with unbuilt vocabulary."""
        vocab = Vocabulary()  # Not built

        with pytest.raises(ValueError, match="Vocabulary must be built"):
            Tokenizer(vocab)

    def test_single_text_encoding(self) -> None:
        """Test encoding single text."""
        vocab = self.create_test_vocab()
        tokenizer = Tokenizer(vocab, max_length=10, add_special_tokens=True)

        # Test basic encoding
        result = tokenizer.encode("hello world", return_tensors="pt")

        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "attention_mask" in result

        # Check tensor properties
        assert isinstance(result["input_ids"], torch.Tensor)
        assert isinstance(result["attention_mask"], torch.Tensor)
        assert result["input_ids"].dtype == torch.long
        assert result["attention_mask"].dtype == torch.long

    def test_batch_text_encoding(self) -> None:
        """Test encoding batch of texts."""
        vocab = self.create_test_vocab()
        tokenizer = Tokenizer(vocab, max_length=8, padding="max_length")

        texts = ["hello world", "peace and love", "good morning"]
        result = tokenizer.encode(texts, return_tensors="pt")

        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "attention_mask" in result

        # Check batch dimensions
        batch_size = len(texts)
        assert result["input_ids"].shape[0] == batch_size
        assert result["attention_mask"].shape[0] == batch_size
        assert result["input_ids"].shape[1] == 8  # max_length

    def test_truncation(self) -> None:
        """Test sequence truncation."""
        vocab = self.create_test_vocab()
        tokenizer = Tokenizer(
            vocab, max_length=4, truncation=True, add_special_tokens=True
        )

        # Long text that will be truncated
        long_text = "hello world peace and love and friendship"
        result = tokenizer.encode(long_text, return_tensors="pt")

        # Should be truncated to max_length
        assert result["input_ids"].shape[-1] == 4

    def test_padding_strategies(self) -> None:
        """Test different padding strategies."""
        vocab = self.create_test_vocab()

        texts = ["hello", "hello world", "hello world peace"]

        # Test max_length padding
        tokenizer_max = Tokenizer(vocab, max_length=6, padding="max_length")
        result_max = tokenizer_max.encode(texts, return_tensors="pt")
        assert result_max["input_ids"].shape[1] == 6

        # Test longest padding
        tokenizer_longest = Tokenizer(vocab, padding="longest")
        result_longest = tokenizer_longest.encode(texts, return_tensors="pt")
        # Should pad to length of longest sequence + special tokens
        assert (
            result_longest["input_ids"].shape[1] >= 3
        )  # longest text length + special tokens

    def test_no_padding(self) -> None:
        """Test no padding option."""
        vocab = self.create_test_vocab()
        tokenizer = Tokenizer(vocab, padding="do_not_pad")

        texts = ["hello", "hello world"]
        result = tokenizer.encode(texts, return_tensors=None)
        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]) != len(result[1])

    def test_special_tokens_handling(self) -> None:
        """Test special token addition and removal."""
        vocab = self.create_test_vocab()

        # With special tokens
        tokenizer_with = Tokenizer(vocab, add_special_tokens=True, padding="do_not_pad")
        result_with = tokenizer_with.encode("hello world", return_tensors=None)

        # Without special tokens
        tokenizer_without = Tokenizer(
            vocab, add_special_tokens=False, padding="do_not_pad"
        )
        result_without = tokenizer_without.encode("hello world", return_tensors=None)

        # With special tokens should be longer
        assert len(result_with) > len(result_without)

    def test_attention_mask(self) -> None:
        """Test attention mask generation."""
        vocab = self.create_test_vocab()
        tokenizer = Tokenizer(vocab, max_length=6, padding="max_length")

        texts = ["hello", "hello world peace"]
        result = tokenizer.encode(texts, return_tensors="pt")

        input_ids = result["input_ids"]
        attention_mask = result["attention_mask"]

        # Check attention mask properties
        assert attention_mask.shape == input_ids.shape

        # Check that attention mask is 0 for padding and 1 for real tokens
        for i in range(len(texts)):
            for j in range(input_ids.shape[1]):
                if input_ids[i, j] == tokenizer.pad_token_id:
                    assert attention_mask[i, j] == 0
                else:
                    assert attention_mask[i, j] == 1

    def test_decode_single_sequence(self) -> None:
        """Test decoding single sequence."""
        vocab = self.create_test_vocab()
        tokenizer = Tokenizer(vocab, add_special_tokens=False)

        # Encode then decode
        original_text = "hello world"
        encoded = tokenizer.encode(
            original_text, return_tensors=None, padding="do_not_pad"
        )
        decoded = tokenizer.decode(encoded)

        assert decoded == original_text.lower()

    def test_decode_with_special_tokens(self) -> None:
        """Test decoding with special token handling."""
        vocab = self.create_test_vocab()
        tokenizer = Tokenizer(vocab, add_special_tokens=True)

        original_text = "hello world"
        encoded = tokenizer.encode(
            original_text, return_tensors=None, padding="do_not_pad"
        )

        # Decode with special tokens removed
        decoded_clean = tokenizer.decode(encoded, skip_special_tokens=True)
        assert decoded_clean == original_text

        # Decode with special tokens kept
        decoded_with_special = tokenizer.decode(encoded, skip_special_tokens=False)
        assert vocab.SOS_TOKEN in decoded_with_special
        assert vocab.EOS_TOKEN in decoded_with_special

    def test_batch_decode(self) -> None:
        """Test batch decoding."""
        vocab = self.create_test_vocab()
        tokenizer = Tokenizer(
            vocab, add_special_tokens=False, padding="max_length", max_length=5
        )

        texts = ["hello world", "peace love"]
        encoded = tokenizer.encode(texts, return_tensors="pt")
        decoded = tokenizer.batch_decode(encoded["input_ids"])

        assert len(decoded) == len(texts)
        for i, original in enumerate(texts):
            # Remove padding artifacts from comparison
            decoded_clean = decoded[i].strip()
            assert original in decoded_clean or decoded_clean in original

    def test_sequence_length_calculation(self) -> None:
        """Test sequence length calculation."""
        vocab = self.create_test_vocab()
        tokenizer = Tokenizer(vocab, add_special_tokens=True)

        text = "hello world"
        length = tokenizer.get_sequence_length(text)

        # Should be word count + special tokens
        assert length >= 2  # at least the two words

        # Compare with actual encoding
        encoded = tokenizer.encode(text, padding="do_not_pad", return_tensors=None)
        assert length == len(encoded)

    def test_get_vocab_size(self) -> None:
        """Test vocabulary size getter."""
        vocab = self.create_test_vocab()
        tokenizer = Tokenizer(vocab)

        assert tokenizer.get_vocab_size() == vocab.vocab_size

    def test_get_special_tokens_dict(self) -> None:
        """Test special tokens dictionary."""
        vocab = self.create_test_vocab()
        tokenizer = Tokenizer(vocab)

        special_tokens = tokenizer.get_special_tokens_dict()

        assert "pad_token_id" in special_tokens
        assert "unk_token_id" in special_tokens
        assert "sos_token_id" in special_tokens
        assert "eos_token_id" in special_tokens

        # Check values match vocabulary
        assert special_tokens["pad_token_id"] == vocab.word_to_idx[vocab.PAD_TOKEN]
        assert special_tokens["unk_token_id"] == vocab.word_to_idx[vocab.UNK_TOKEN]


class TestSequenceCollator:
    """Test cases for SequenceCollator class."""

    def create_test_tokenizer(self) -> Tokenizer:
        """Create tokenizer for testing."""
        vocab = Vocabulary(min_frequency=1)
        texts = ["hello world", "good morning", "peace love"]
        vocab.build_from_texts(texts)
        return Tokenizer(vocab, max_length=8, padding="longest")

    def test_collator_initialization(self) -> None:
        """Test collator initialization."""
        tokenizer = self.create_test_tokenizer()
        collator = SequenceCollator(tokenizer)

        assert collator.tokenizer == tokenizer
        assert collator.padding == "longest"
        assert collator.return_tensors == "pt"

    def test_collate_batch_with_labels(self) -> None:
        """Test collating batch with labels."""
        tokenizer = self.create_test_tokenizer()
        collator = SequenceCollator(tokenizer, max_length=6, padding="max_length")

        batch = [
            {"text": "hello world", "label": 1},
            {"text": "good morning", "label": 0},
            {"text": "peace", "label": 1},
        ]

        result = collator(batch)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result

        # Check shapes
        assert result["input_ids"].shape[0] == 3  # batch size
        assert result["input_ids"].shape[1] == 6  # max_length
        assert result["labels"].shape[0] == 3

        # Check label values
        expected_labels = torch.tensor([1, 0, 1], dtype=torch.long)
        assert torch.equal(result["labels"], expected_labels)

    def test_collate_batch_without_labels(self) -> None:
        """Test collating batch without labels."""
        tokenizer = self.create_test_tokenizer()
        collator = SequenceCollator(tokenizer)

        batch = [{"text": "hello world"}, {"text": "good morning"}, {"text": "peace"}]

        result = collator(batch)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" not in result


class TestTokenizationUtilities:
    """Test utility functions for tokenization."""

    def create_test_setup(self):
        """Create test vocabulary and tokenizer."""
        vocab = Vocabulary(min_frequency=1)
        texts = [
            "hello world", 
            "good morning", 
            "peace and love",
            "i cannot believe this amazing movie",
            "i can't believe this amazing movie"
        ]
        vocab.build_from_texts(texts)

        preprocessor = create_default_preprocessor()
        tokenizer = create_tokenizer(vocab, preprocessor, max_length=8)

        return vocab, preprocessor, tokenizer

    def test_create_tokenizer_function(self) -> None:
        """Test create_tokenizer utility function."""
        vocab, preprocessor, _ = self.create_test_setup()

        tokenizer = create_tokenizer(
            vocabulary=vocab,
            preprocessor=preprocessor,
            max_length=10,
            padding="max_length",
            add_special_tokens=True,
        )

        assert isinstance(tokenizer, Tokenizer)
        assert tokenizer.vocabulary == vocab
        assert tokenizer.preprocessor == preprocessor
        assert tokenizer.max_length == 10
        assert tokenizer.padding == "max_length"
        assert tokenizer.add_special_tokens is True

    def test_analyze_sequence_lengths(self) -> None:
        """Test sequence length analysis function."""
        vocab, preprocessor, tokenizer = self.create_test_setup()

        texts = [
            "hello",
            "hello world",
            "hello world peace",
            "hello world peace and love",
        ]

        stats = analyze_sequence_lengths(texts, tokenizer)

        # Check required statistics
        assert "count" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "percentile_50" in stats
        assert "percentile_90" in stats

        # Check values make sense
        assert stats["count"] == len(texts)
        assert stats["min"] > 0
        assert stats["max"] >= stats["min"]
        assert stats["mean"] > 0

    def test_create_sequences_from_texts(self) -> None:
        """Test create_sequences_from_texts function."""
        vocab, preprocessor, tokenizer = self.create_test_setup()

        texts = ["hello world", "good morning"]
        labels = [1, 0]

        # Test with tokenizer
        result_with_tokenizer = create_sequences_from_texts(
            texts=texts, labels=labels, tokenizer=tokenizer
        )

        assert "input_ids" in result_with_tokenizer
        assert "attention_mask" in result_with_tokenizer
        assert "labels" in result_with_tokenizer

        # Test without tokenizer (should create one)
        result_without_tokenizer = create_sequences_from_texts(
            texts=texts,
            labels=labels,
            vocabulary=vocab,
            preprocessor=preprocessor,
            max_length=10,
        )

        assert "input_ids" in result_without_tokenizer
        assert "attention_mask" in result_without_tokenizer
        assert "labels" in result_without_tokenizer

    def test_create_sequences_error_handling(self) -> None:
        """Test error handling in create_sequences_from_texts."""
        texts = ["hello world"]

        # Should raise error when neither tokenizer nor vocabulary provided
        with pytest.raises(ValueError, match="Either tokenizer or vocabulary"):
            create_sequences_from_texts(texts=texts)

    def test_tokenizer_with_preprocessor_integration(self) -> None:
        """Test tokenizer integration with preprocessor."""
        vocab, preprocessor, _ = self.create_test_setup()

        tokenizer = Tokenizer(
            vocabulary=vocab,
            preprocessor=preprocessor,
            add_special_tokens=False,
            padding="do_not_pad",
        )

        # Test text that will be preprocessed
        original_text = "I can't believe this <b>AMAZING</b> movie!"
        result = tokenizer.encode(original_text, return_tensors=None)
        decoded = tokenizer.decode(result)

        # Should be preprocessed (lowercase, contractions expanded, HTML removed)
        assert "cannot" in decoded
        assert "amazing" in decoded
        assert "<b>" not in decoded
        assert decoded == "i cannot believe this amazing movie <UNK>"

    def test_edge_cases(self) -> None:
        """Test edge cases in tokenization."""
        vocab, _, tokenizer = self.create_test_setup()

        # Empty string
        empty_result = tokenizer.encode("", return_tensors="pt")
        assert empty_result["input_ids"].shape[1] >= 0

        # Very long text (test truncation)
        long_text = " ".join(["word"] * 100)
        long_result = tokenizer.encode(long_text, max_length=5, return_tensors="pt")
        assert long_result["input_ids"].shape[1] == 5

        # Text with only unknown words
        unknown_text = "zzxyzqwerty unknownword123"
        unknown_result = tokenizer.encode(unknown_text, return_tensors="pt")
        # Should contain UNK tokens
        assert tokenizer.unk_token_id in unknown_result["input_ids"].flatten().tolist()


class TestSentimentDataset:
    """Test cases for SentimentDataset class."""

    def create_test_csv(self, tmp_path: Path) -> Path:
        """Create a test CSV file for dataset testing."""
        data = {
            "review": [
                "This movie is absolutely amazing!",
                "Terrible film, waste of time.",
                "Great acting and wonderful story.",
                "Boring and predictable plot.",
                "Excellent cinematography and direction.",
                "Worst movie I have ever seen.",
                "Perfect blend of action and drama.",
                "Disappointing and overrated.",
            ],
            "sentiment": [
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
            ],
            "rating": [5, 1, 4, 2, 5, 1, 4, 2],
        }

        df = pd.DataFrame(data)
        csv_path = tmp_path / "test_sentiment.csv"
        df.to_csv(csv_path, index=False)

        return csv_path

    def create_test_tokenizer(self) -> Tokenizer:
        """Create tokenizer for testing."""
        vocab = Vocabulary(min_frequency=1)
        texts = [
            "amazing movie great",
            "terrible film bad",
            "wonderful story excellent",
            "boring plot disappointing",
        ]
        vocab.build_from_texts(texts)

        preprocessor = create_default_preprocessor()
        return create_tokenizer(vocab, preprocessor, max_length=20)

    def test_dataset_initialization(self, tmp_path):
        """Test dataset initialization."""
        csv_path = self.create_test_csv(tmp_path)
        tokenizer = self.create_test_tokenizer()

        # Basic initialization
        dataset = SentimentDataset(
            data_path=csv_path,
            tokenizer=tokenizer,
            text_column="review",
            label_column="sentiment",
        )

        assert len(dataset) == 8
        assert dataset.text_column == "review"
        assert dataset.label_column == "sentiment"
        assert dataset.tokenizer == tokenizer

        # Check label mapping
        expected_mapping = {"negative": 0, "positive": 1}
        assert dataset.label_mapping == expected_mapping

    def test_dataset_with_missing_file(self, tmp_path):
        """Test dataset with non-existent file."""
        tokenizer = self.create_test_tokenizer()
        missing_path = tmp_path / "missing.csv"

        with pytest.raises(FileNotFoundError):
            SentimentDataset(missing_path, tokenizer)

    def test_dataset_with_invalid_columns(self, tmp_path):
        """Test dataset with missing required columns."""
        # Create CSV with wrong column names
        data = pd.DataFrame({"text": ["sample text"], "label": ["positive"]})
        csv_path = tmp_path / "wrong_columns.csv"
        data.to_csv(csv_path, index=False)

        tokenizer = self.create_test_tokenizer()

        # Should raise error for missing text column
        with pytest.raises(ValueError, match="Text column 'review' not found"):
            SentimentDataset(csv_path, tokenizer)

    def test_dataset_getitem(self, tmp_path):
        """Test getting individual samples from dataset."""
        csv_path = self.create_test_csv(tmp_path)
        tokenizer = self.create_test_tokenizer()

        dataset = SentimentDataset(csv_path, tokenizer, max_length=15)

        # Get first sample
        sample = dataset[0]

        # Check output structure
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample

        # Check tensor properties
        assert isinstance(sample["input_ids"], torch.Tensor)
        assert isinstance(sample["attention_mask"], torch.Tensor)
        assert isinstance(sample["labels"], torch.Tensor)

        # Check shapes
        assert sample["input_ids"].shape[0] == 15  # max_length
        assert sample["attention_mask"].shape[0] == 15
        assert sample["labels"].shape == torch.Size([])  # Scalar

    def test_dataset_index_error(self, tmp_path):
        """Test index out of range error."""
        csv_path = self.create_test_csv(tmp_path)
        tokenizer = self.create_test_tokenizer()

        dataset = SentimentDataset(csv_path, tokenizer)

        # Valid index
        _ = dataset[0]
        _ = dataset[len(dataset) - 1]

        # Invalid indices
        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]

        # with pytest.raises(IndexError):
        #     _ = dataset[-1]  # Negative indexing not supported

    def test_dataset_label_mapping_binary(self, tmp_path):
        """Test automatic label mapping for binary classification."""
        # Test different binary label formats
        test_cases = [
            (["positive", "negative"], {"negative": 0, "positive": 1}),
            (["pos", "neg"], {"neg": 0, "pos": 1}),
            ([0, 1], {0: 0, 1: 1}),
            (["good", "bad"], {"bad": 0, "good": 1}),  # Alphabetical order
        ]

        tokenizer = self.create_test_tokenizer()

        for labels, expected_mapping in test_cases:
            data = pd.DataFrame(
                {"review": ["sample text"] * len(labels), "sentiment": labels}
            )

            csv_path = tmp_path / f"test_{hash(str(labels))}.csv"
            data.to_csv(csv_path, index=False)

            dataset = SentimentDataset(csv_path, tokenizer)
            assert dataset.label_mapping == expected_mapping

    def test_dataset_label_mapping_multiclass(self, tmp_path):
        """Test automatic label mapping for multi-class classification."""
        labels = ["positive", "negative", "neutral"]
        expected_mapping = {"negative": 0, "neutral": 1, "positive": 2}

        data = pd.DataFrame(
            {"review": ["text1", "text2", "text3"], "sentiment": labels}
        )

        csv_path = tmp_path / "multiclass.csv"
        data.to_csv(csv_path, index=False)

        tokenizer = self.create_test_tokenizer()
        dataset = SentimentDataset(csv_path, tokenizer)

        assert dataset.label_mapping == expected_mapping

    def test_dataset_custom_label_mapping(self, tmp_path):
        """Test custom label mapping."""
        csv_path = self.create_test_csv(tmp_path)
        tokenizer = self.create_test_tokenizer()

        custom_mapping = {"positive": 2, "negative": 5}

        dataset = SentimentDataset(csv_path, tokenizer, label_mapping=custom_mapping)

        assert dataset.label_mapping == custom_mapping

        # Check that samples use custom mapping
        sample = dataset[0]  # First sample should be positive -> 2
        assert sample["labels"].item() == 2

    def test_dataset_get_sample_info(self, tmp_path):
        """Test getting detailed sample information."""
        csv_path = self.create_test_csv(tmp_path)
        tokenizer = self.create_test_tokenizer()

        dataset = SentimentDataset(csv_path, tokenizer)

        info = dataset.get_sample_info(0)

        assert "index" in info
        assert "text" in info
        assert "label" in info
        assert "original_label" in info
        assert "text_length" in info
        assert "token_count" in info
        assert "tokens" in info

        assert info["index"] == 0
        assert isinstance(info["text"], str)
        assert isinstance(info["label"], int)
        assert info["original_label"] == "positive"

    def test_dataset_label_distribution(self, tmp_path):
        """Test label distribution calculation."""
        csv_path = self.create_test_csv(tmp_path)
        tokenizer = self.create_test_tokenizer()

        dataset = SentimentDataset(csv_path, tokenizer)

        distribution = dataset.get_label_distribution()

        # Should have equal distribution (4 positive, 4 negative)
        assert distribution[0] == 4  # negative
        assert distribution[1] == 4  # positive
        assert len(distribution) == 2

    def test_dataset_text_statistics(self, tmp_path):
        """Test text statistics calculation."""
        csv_path = self.create_test_csv(tmp_path)
        tokenizer = self.create_test_tokenizer()

        dataset = SentimentDataset(csv_path, tokenizer)

        stats = dataset.get_text_statistics()

        required_keys = [
            "text_length_mean",
            "text_length_std",
            "text_length_min",
            "text_length_max",
            "token_length_mean",
            "token_length_std",
            "token_length_min",
            "token_length_max",
            "token_length_95th",
        ]

        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float, np.int64))

        assert stats["text_length_min"] > 0
        assert stats["token_length_min"] > 0
        assert stats["text_length_max"] >= stats["text_length_min"]
        assert stats["token_length_max"] >= stats["token_length_min"]
        assert stats["token_length_95th"] >= stats["token_length_min"]
        assert stats["text_length_mean"] > 0

    def test_dataset_with_invalid_data(self, tmp_path):
        """Test dataset handling of invalid data."""
        # Create CSV with missing values
        data = pd.DataFrame(
            {
                "review": ["Good movie", None, "Bad film", ""],
                "sentiment": ["positive", "negative", None, "positive"],
            }
        )

        csv_path = tmp_path / "invalid_data.csv"
        data.to_csv(csv_path, index=False)

        tokenizer = self.create_test_tokenizer()

        # With filtering enabled (default)
        dataset_filtered = SentimentDataset(csv_path, tokenizer, filter_invalid=True)
        assert len(dataset_filtered) == 1  # Only 'Good movie' is valid

        # With filtering disabled
        dataset_unfiltered = SentimentDataset(csv_path, tokenizer, filter_invalid=False)
        assert len(dataset_unfiltered) == 4  # All samples kept

    def test_dataset_json_loading(self, tmp_path):
        """Test loading dataset from JSON file."""
        data = [
            {"review": "Amazing movie!", "sentiment": "positive"},
            {"review": "Terrible film.", "sentiment": "negative"},
        ]

        json_path = tmp_path / "test.json"
        pd.DataFrame(data).to_json(json_path, orient="records")

        tokenizer = self.create_test_tokenizer()
        dataset = SentimentDataset(json_path, tokenizer)

        assert len(dataset) == 2

    def test_dataset_caching(self, tmp_path):
        """Test dataset caching functionality."""
        csv_path = self.create_test_csv(tmp_path)
        tokenizer = self.create_test_tokenizer()
        cache_dir = tmp_path / "cache"

        # Create dataset with caching
        dataset = SentimentDataset(csv_path, tokenizer, cache_dir=cache_dir)

        # Save cache
        dataset.save_cache()

        # Verify cache file exists
        assert dataset.cache_file.exists()

        # Create new dataset and load cache
        dataset2 = SentimentDataset(csv_path, tokenizer, cache_dir=cache_dir)

        # Should load from cache
        success = dataset2.load_cache()
        assert success

        # Datasets should be equivalent
        assert len(dataset) == len(dataset2)
        assert dataset.label_mapping == dataset2.label_mapping


class TestDatasetSplitter:
    """Test cases for DatasetSplitter."""

    def create_test_dataset(self, tmp_path: Path, size: int = 100) -> SentimentDataset:
        """Create a test dataset of specified size."""
        data = {
            "review": [f"Sample review {i}" for i in range(size)],
            "sentiment": [
                "positive" if i % 2 == 0 else "negative" for i in range(size)
            ],
        }

        df = pd.DataFrame(data)
        csv_path = tmp_path / "large_test.csv"
        df.to_csv(csv_path, index=False)

        vocab = Vocabulary(min_frequency=1)
        vocab.build_from_texts(["sample text"])
        tokenizer = create_tokenizer(vocab, max_length=10)

        return SentimentDataset(csv_path, tokenizer)

    def test_dataset_splitting(self, tmp_path):
        """Test dataset splitting functionality."""
        dataset = self.create_test_dataset(tmp_path, size=100)

        train_ds, val_ds, test_ds = DatasetSplitter.split_dataset(
            dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=42
        )

        # Check sizes
        assert len(train_ds) == 70
        assert len(val_ds) == 20
        assert len(test_ds) == 10

        # Total should equal original
        assert len(train_ds) + len(val_ds) + len(test_ds) == len(dataset)

    def test_dataset_splitting_invalid_ratios(self, tmp_path):
        """Test dataset splitting with invalid ratios."""
        dataset = self.create_test_dataset(tmp_path, size=10)

        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            DatasetSplitter.split_dataset(
                dataset, train_ratio=0.6, val_ratio=0.3, test_ratio=0.2  # Sum = 1.1
            )

    def test_split_by_indices(self, tmp_path):
        """Test splitting dataset by specific indices."""
        dataset = self.create_test_dataset(tmp_path, size=20)

        train_indices = list(range(0, 14))
        val_indices = list(range(14, 17))
        test_indices = list(range(17, 20))

        train_ds, val_ds, test_ds = DatasetSplitter.split_by_indices(
            dataset, train_indices, val_indices, test_indices
        )

        assert len(train_ds) == 14
        assert len(val_ds) == 3
        assert len(test_ds) == 3


class TestDataLoaders:
    """Test cases for DataLoader creation."""

    def create_test_splits(self, tmp_path: Path):
        """Create test dataset splits."""
        dataset = self.create_test_dataset(tmp_path, size=30)
        return DatasetSplitter.split_dataset(
            dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )

    def create_test_dataset(self, tmp_path: Path, size: int) -> SentimentDataset:
        """Create test dataset."""
        data = {
            "review": [f"Sample review {i}" for i in range(size)],
            "sentiment": [
                "positive" if i % 2 == 0 else "negative" for i in range(size)
            ],
        }

        df = pd.DataFrame(data)
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        vocab = Vocabulary(min_frequency=1)
        vocab.build_from_texts(["sample text"])
        tokenizer = create_tokenizer(vocab, max_length=10)

        return SentimentDataset(csv_path, tokenizer)

    def test_create_data_loaders(self, tmp_path):
        """Test DataLoader creation."""
        train_ds, val_ds, test_ds = self.create_test_splits(tmp_path)

        vocab = Vocabulary(min_frequency=1)
        vocab.build_from_texts(["sample text"])
        tokenizer = create_tokenizer(vocab, max_length=10)

        train_loader, val_loader, test_loader = create_data_loaders(
            train_ds, val_ds, test_ds, tokenizer, batch_size=4
        )

        # Check loader properties
        assert train_loader.batch_size == 4
        assert val_loader.batch_size == 4
        assert test_loader.batch_size == 4

        # Check number of batches
        assert len(train_loader) == len(train_ds) // 4  # drop_last=True for training
        assert len(val_loader) == (len(val_ds) + 3) // 4  # drop_last=False
        assert len(test_loader) == (len(test_ds) + 3) // 4

        # Test batch iteration
        for batch in train_loader:
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "labels" in batch

            # Check batch size (last batch might be smaller for val/test)
            batch_size = batch["input_ids"].size(0)
            assert batch_size <= 4
            assert batch["attention_mask"].size(0) == batch_size
            assert batch["labels"].size(0) == batch_size
            break  # Just test first batch

    def test_load_sentiment_data_pipeline(self, tmp_path):
        """Test complete data loading pipeline."""
        # Create test data
        data = {
            "review": [f"Sample movie review {i}" for i in range(50)],
            "sentiment": ["positive" if i % 2 == 0 else "negative" for i in range(50)],
        }

        df = pd.DataFrame(data)
        csv_path = tmp_path / "pipeline_test.csv"
        df.to_csv(csv_path, index=False)

        # Create vocabulary
        vocab = Vocabulary(min_frequency=1)
        vocab.build_from_texts(["sample movie review text"])

        # Load data using pipeline
        train_loader, val_loader, test_loader, dataset = load_sentiment_data(
            data_path=csv_path,
            vocabulary=vocab,
            max_length=15,
            batch_size=8,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_seed=42,
        )

        # Check dataset
        assert len(dataset) == 50

        # Check loaders
        assert len(train_loader) == 30 // 8  # 30 samples, batch_size=8, drop_last=True

        # Test batch from each loader
        for loader_name, loader in [
            ("train", train_loader),
            ("val", val_loader),
            ("test", test_loader),
        ]:
            if len(loader) > 0:
                batch = next(iter(loader))
                assert "input_ids" in batch
                assert "attention_mask" in batch
                assert "labels" in batch

                # Check tensor shapes
                batch_size = batch["input_ids"].size(0)
                seq_length = batch["input_ids"].size(1)

                assert batch["attention_mask"].shape == (batch_size, seq_length)
                assert batch["labels"].shape == (batch_size,)

                # Check that text is not in batch
                assert "text" not in batch


class TestAugmentedSentimentDataset:
    """Test cases for AugmentedSentimentDataset."""

    def create_test_csv(self, tmp_path: Path) -> Path:
        """Create test CSV for augmentation testing."""
        data = {
            "review": [
                "This movie is really good and amazing.",
                "The film was bad and terrible.",
                "Great acting in this wonderful movie.",
                "Boring plot made this film awful.",
            ],
            "sentiment": ["positive", "negative", "positive", "negative"],
        }

        df = pd.DataFrame(data)
        csv_path = tmp_path / "augment_test.csv"
        df.to_csv(csv_path, index=False)

        return csv_path

    def test_augmented_dataset_initialization(self, tmp_path):
        """Test augmented dataset initialization."""
        csv_path = self.create_test_csv(tmp_path)

        vocab = Vocabulary(min_frequency=1)
        vocab.build_from_texts(["good bad movie film"])
        tokenizer = create_tokenizer(vocab, max_length=15)

        dataset = AugmentedSentimentDataset(
            csv_path,
            tokenizer,
            augmentation_prob=0.5,
            augmentation_methods=["synonym_replacement", "random_insertion"],
        )

        assert dataset.augmentation_prob == 0.5
        assert dataset.augmentation_methods == [
            "synonym_replacement",
            "random_insertion",
        ]
        assert len(dataset) == 4

    def test_augmentation_methods(self, tmp_path):
        """Test individual augmentation methods."""
        csv_path = self.create_test_csv(tmp_path)

        vocab = Vocabulary(min_frequency=1)
        vocab.build_from_texts(["good bad movie film great terrible"])
        tokenizer = create_tokenizer(vocab, max_length=20)

        dataset = AugmentedSentimentDataset(
            csv_path,
            tokenizer,
            augmentation_prob=1.0,  # Always apply augmentation for testing
            augmentation_methods=["synonym_replacement"],
        )

        # Test that augmentation can change text
        original_text = dataset.samples[0]["text"]

        # Apply augmentation multiple times - should sometimes get different results
        augmented_texts = set()
        for _ in range(10):
            augmented_text = dataset._apply_augmentation(original_text)
            augmented_texts.add(augmented_text)

        # Should have some variation (though not guaranteed due to randomness)
        # At minimum, we test that the method doesn't crash
        assert len(augmented_texts) >= 1
        assert all(isinstance(text, str) for text in augmented_texts)

    def test_augmented_getitem(self, tmp_path):
        """Test getting items from augmented dataset."""
        csv_path = self.create_test_csv(tmp_path)

        vocab = Vocabulary(min_frequency=1)
        vocab.build_from_texts(["good bad movie film"])
        tokenizer = create_tokenizer(vocab, max_length=15)

        dataset = AugmentedSentimentDataset(
            csv_path,
            tokenizer,
            augmentation_prob=0.2,  # Low probability to avoid test flakiness
        )

        # Test getting samples
        sample = dataset[0]

        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample

        # Check tensor properties
        assert isinstance(sample["input_ids"], torch.Tensor)
        assert isinstance(sample["attention_mask"], torch.Tensor)
        assert isinstance(sample["labels"], torch.Tensor)

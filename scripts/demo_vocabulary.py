#!/usr/bin/env python3
"""
Demonstration script for vocabulary building functionality.
Shows how to create vocabularies from movie review data with different settings.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessing import create_default_preprocessor
from src.data.vocabulary import Vocabulary, create_vocabulary_from_data


def demo_basic_vocabulary():
    """Demonstrate basic vocabulary building."""
    print("=" * 80)
    print("BASIC VOCABULARY BUILDING DEMONSTRATION")
    print("=" * 80)

    # Sample movie reviews
    sample_texts = [
        "This movie is absolutely amazing! I loved every minute of it.",
        "The acting was superb and the plot was engaging.",
        "I can't recommend this film enough. It's a masterpiece!",
        "Terrible movie. Don't waste your time watching this.",
        "The story was boring and the characters were poorly developed.",
        "Amazing cinematography but the plot was confusing.",
        "Great performances by all the actors in this film.",
    ]

    print(f"Sample texts: {len(sample_texts)} movie reviews")
    print(f"Sample: '{sample_texts[0]}'")

    # Create vocabulary with different settings
    print("\nCREATING VOCABULARIES WITH DIFFERENT SETTINGS")
    print("-" * 60)

    # Basic vocabulary
    vocab_basic = Vocabulary(min_frequency=1, max_vocab_size=50)
    vocab_basic.build_from_texts(sample_texts)

    print(f"\nBasic Vocabulary Stats:")
    stats = vocab_basic.get_vocabulary_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Vocabulary with preprocessor
    preprocessor = create_default_preprocessor()
    vocab_preprocessed = Vocabulary(min_frequency=1, max_vocab_size=50)
    vocab_preprocessed.build_from_texts(sample_texts, preprocessor)

    print(f"\nPreprocessed Vocabulary Stats:")
    stats_prep = vocab_preprocessed.get_vocabulary_stats()
    for key, value in stats_prep.items():
        print(f"  {key}: {value}")

    # Show most common words
    print(f"\n🔤 Most Common Words (Preprocessed):")
    most_common = vocab_preprocessed.get_most_common_words(10)
    for word, count in most_common:
        print(f"  '{word}': {count} occurrences")

    # Demonstrate text-to-sequence conversion
    print(f"\n🔢 TEXT-TO-SEQUENCE CONVERSION")
    print("-" * 60)

    test_text = "This movie is amazing!"
    print(f"Original text: '{test_text}'")

    # Without preprocessing
    sequence_basic = vocab_basic.text_to_sequence(test_text.lower())
    print(f"Basic sequence: {sequence_basic}")
    reconstructed_basic = vocab_basic.sequence_to_text(sequence_basic)
    print(f"Reconstructed: '{reconstructed_basic}'")

    # With preprocessing
    sequence_prep = vocab_preprocessed.text_to_sequence(test_text, preprocessor)
    print(f"Preprocessed sequence: {sequence_prep}")
    reconstructed_prep = vocab_preprocessed.sequence_to_text(sequence_prep)
    print(f"Reconstructed: '{reconstructed_prep}'")

    # With special tokens
    sequence_special = vocab_preprocessed.text_to_sequence(
        test_text, preprocessor, add_special_tokens=True
    )
    print(f"With special tokens: {sequence_special}")
    print(
        f"  SOS token index: {vocab_preprocessed.word_to_idx[vocab_preprocessed.SOS_TOKEN]}"
    )
    print(
        f"  EOS token index: {vocab_preprocessed.word_to_idx[vocab_preprocessed.EOS_TOKEN]}"
    )

    return vocab_preprocessed


def demo_imdb_vocabulary():
    """Demonstrate vocabulary building with IMDB data."""
    print("\n" + "=" * 80)
    print("IMDB VOCABULARY BUILDING DEMONSTRATION")
    print("=" * 80)

    train_path = "data/processed/imdb_train.csv"
    test_path = "data/processed/imdb_test.csv"

    # Check if data exists
    if not Path(train_path).exists() or not Path(test_path).exists():
        print("IMDB data not found. Please run the data download script first.")
        print("   Run: python src/data/download_data.py")
        return None

    print(f"📁 Building vocabulary from IMDB dataset...")
    print(f"   Training data: {train_path}")
    print(f"   Test data: {test_path}")

    # Create preprocessor
    preprocessor = create_default_preprocessor()

    # Build vocabulary with different configurations
    print(f"\nTESTING DIFFERENT VOCABULARY CONFIGURATIONS")
    print("-" * 60)

    configs = [
        {"name": "Small", "max_vocab_size": 1000, "min_frequency": 5},
        {"name": "Medium", "max_vocab_size": 5000, "min_frequency": 3},
        {"name": "Large", "max_vocab_size": 10000, "min_frequency": 2},
    ]

    vocabularies = {}

    for config in configs:
        print(f"\nBuilding {config['name']} vocabulary...")
        print(
            f"   Max size: {config['max_vocab_size']}, Min frequency: {config['min_frequency']}"
        )

        vocab = create_vocabulary_from_data(
            train_path,
            test_path,
            max_vocab_size=config["max_vocab_size"],
            min_frequency=config["min_frequency"],
            preprocessor=preprocessor,
        )

        vocabularies[config["name"]] = vocab

        # Show statistics
        stats = vocab.get_vocabulary_stats()
        print(f"   Final vocabulary size: {stats['vocab_size']}")
        print(f"   Content words: {stats['content_words']}")
        print(f"   Total unique words seen: {stats['total_unique_words']}")
        print(
            f"   Words filtered: {stats['total_unique_words'] - stats['content_words']}"
        )

    # Compare vocabularies
    print(f"\nVOCABULARY COMPARISON")
    print("-" * 60)
    print(f"{'Config':<8} {'Size':<6} {'Coverage':<10} {'Most Common Word'}")
    print("-" * 60)

    for name, vocab in vocabularies.items():
        stats = vocab.get_vocabulary_stats()
        most_common = stats.get("most_common_word", ("N/A", 0))
        coverage = (
            stats["content_words"] / stats["total_unique_words"] * 100
            if stats["total_unique_words"]
            else 0
        )
        print(
            f"{name:<8} {stats['vocab_size']:<6} {coverage:<10.1f}% {most_common[0]} ({most_common[1]})"
        )

    # Test vocabulary on sample reviews
    print(f"\nTESTING VOCABULARY ON SAMPLE REVIEWS")
    print("-" * 60)

    # Use medium vocabulary for testing
    test_vocab = vocabularies["Medium"]

    sample_reviews = [
        "This movie is absolutely fantastic!",
        "Terrible film, waste of time",
        "Amazing cinematography and excellent acting",
        "The plot was confusing and boring",
    ]

    for i, review in enumerate(sample_reviews, 1):
        print(f"\nReview {i}: '{review}'")

        # Convert to sequence
        sequence = test_vocab.text_to_sequence(review, preprocessor)
        print(f"Sequence length: {len(sequence)}")
        print(f"Sequence: {sequence[:10]}{'...' if len(sequence) > 10 else ''}")

        # Check unknown words
        tokens = preprocessor.preprocess_and_tokenize(review)
        unknown_count = sum(
            1 for token in tokens if not test_vocab.contains_word(token)
        )
        print(
            f"Unknown words: {unknown_count}/{len(tokens)} ({unknown_count/len(tokens)*100:.1f}%)"
        )

    # Save vocabulary
    save_path = "models/vocabulary/imdb_vocab_medium.pkl"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    test_vocab.save(save_path)
    print(f"\n💾 Vocabulary saved to: {save_path}")

    return test_vocab


def demo_vocabulary_features():
    """Demonstrate advanced vocabulary features."""
    print("\n" + "=" * 80)
    print("ADVANCED VOCABULARY FEATURES")
    print("=" * 80)

    # Create a vocabulary for demonstration
    texts = [
        "the movie was excellent and amazing",
        "the acting was superb and excellent",
        "the plot was boring and terrible",
        "excellent cinematography in this film",
        "the movie had excellent special effects",
    ]

    vocab = Vocabulary(min_frequency=2, max_vocab_size=20)
    vocab.build_from_texts(texts)

    print(f"Vocabulary built from {len(texts)} sentences")
    print(f"Vocabulary size: {vocab.vocab_size}")

    # Word frequency analysis
    print(f"\nWORD FREQUENCY ANALYSIS")
    print("-" * 60)

    test_words = ["excellent", "movie", "the", "terrible", "amazing"]
    for word in test_words:
        freq = vocab.get_word_frequency(word)
        in_vocab = vocab.contains_word(word)
        print(f"'{word}': frequency={freq}, in_vocab={in_vocab}")

    # Special token demonstration
    print(f"\nSPECIAL TOKENS")
    print("-" * 60)

    special_tokens = [
        vocab.PAD_TOKEN,
        vocab.UNK_TOKEN,
        vocab.SOS_TOKEN,
        vocab.EOS_TOKEN,
    ]
    for token in special_tokens:
        idx = vocab.word_to_idx.get(token, -1)
        print(f"'{token}': index={idx}")

    # Sequence conversion with special tokens
    print(f"\n🔄 SEQUENCE CONVERSION WITH SPECIAL TOKENS")
    print("-" * 60)

    test_text = "the movie was excellent"

    # Normal sequence
    seq_normal = vocab.text_to_sequence(test_text)
    print(f"Normal: {test_text} -> {seq_normal}")

    # With special tokens
    seq_special = vocab.text_to_sequence(test_text, add_special_tokens=True)
    print(f"Special: {test_text} -> {seq_special}")

    # Convert back
    text_normal = vocab.sequence_to_text(seq_normal)
    text_special = vocab.sequence_to_text(seq_special, remove_special_tokens=True)
    text_with_special = vocab.sequence_to_text(seq_special, remove_special_tokens=False)

    print(f"Back (normal): {seq_normal} -> '{text_normal}'")
    print(f"Back (no special): {seq_special} -> '{text_special}'")
    print(f"Back (with special): {seq_special} -> '{text_with_special}'")

    # Unknown word handling
    print(f"\n❓ UNKNOWN WORD HANDLING")
    print("-" * 60)

    unknown_text = "this film has extraordinary cinematography"
    seq_unknown = vocab.text_to_sequence(unknown_text)
    text_reconstructed = vocab.sequence_to_text(seq_unknown)

    print(f"Unknown text: '{unknown_text}'")
    print(f"Sequence: {seq_unknown}")
    print(f"Reconstructed: '{text_reconstructed}'")
    print(
        f"UNK token: '{vocab.UNK_TOKEN}' (index: {vocab.word_to_idx[vocab.UNK_TOKEN]})"
    )


def main():
    """Run all vocabulary demonstrations."""
    print("VOCABULARY BUILDING SYSTEM DEMONSTRATION")
    print("This script demonstrates the vocabulary building capabilities")
    print("    for sentiment analysis with movie review data.")

    # Basic vocabulary demo
    vocab_basic = demo_basic_vocabulary()

    # IMDB vocabulary demo
    vocab_imdb = demo_imdb_vocabulary()

    # Advanced features demo
    demo_vocabulary_features()

    print("\n" + "=" * 80)
    print("VOCABULARY DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("• Vocabulary building from text data")
    print("• Frequency-based word filtering")
    print("• Vocabulary size limiting")
    print("• Text preprocessing integration")
    print("• Text-to-sequence conversion")
    print("• Special token handling (PAD, UNK, SOS, EOS)")
    print("• Unknown word management")
    print("• Vocabulary persistence (save/load)")
    print("• Real-world IMDB dataset processing")

    print("\nReady for tokenization and PyTorch Dataset creation!")


if __name__ == "__main__":
    main()

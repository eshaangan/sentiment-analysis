#!/usr/bin/env python3
"""
Demonstration script for tokenization and text-to-sequence conversion.
Shows how to convert text to tensors for PyTorch model training.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.data.preprocessing import create_default_preprocessor
from src.data.tokenization import (SequenceCollator, Tokenizer,
                                   analyze_sequence_lengths,
                                   create_sequences_from_texts,
                                   create_tokenizer)
from src.data.vocabulary import Vocabulary


def demo_basic_tokenization():
    """Demonstrate basic tokenization functionality."""
    print("=" * 80)
    print("BASIC TOKENIZATION DEMONSTRATION")
    print("=" * 80)

    # Create sample vocabulary
    print("Creating vocabulary from movie reviews...")
    vocab = Vocabulary(min_frequency=1, max_vocab_size=100)
    sample_texts = [
        "This movie is absolutely amazing and fantastic!",
        "The acting was superb and the plot engaging.",
        "I can't recommend this film enough.",
        "Terrible movie, complete waste of time.",
        "The story was boring and poorly written.",
        "Amazing cinematography but confusing plot.",
        "Great performances by all the actors.",
    ]

    preprocessor = create_default_preprocessor()
    vocab.build_from_texts(sample_texts, preprocessor)

    print(f"Vocabulary created with {vocab.vocab_size} words")

    # Create tokenizer
    print(f"\nCreating tokenizer with max_length=20...")
    tokenizer = Tokenizer(
        vocabulary=vocab,
        preprocessor=preprocessor,
        max_length=20,
        padding="max_length",
        add_special_tokens=True,
    )

    print(f"Tokenizer created")
    print(f"   Vocab size: {tokenizer.get_vocab_size()}")
    print(f"   Special tokens: {tokenizer.get_special_tokens_dict()}")

    return vocab, preprocessor, tokenizer


def demo_single_text_encoding(tokenizer):
    """Demonstrate encoding single text."""
    print(f"\nSINGLE TEXT ENCODING")
    print("-" * 60)

    test_texts = [
        "This movie is great!",
        "I can't believe how amazing this film is.",
        "Terrible acting and boring plot.",
        "Best movie ever made!",
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\nExample {i}: '{text}'")

        # Get sequence length
        seq_length = tokenizer.get_sequence_length(text)
        print(f"   Sequence length: {seq_length}")

        # Encode without tensors
        tokens_list = tokenizer.encode(text, return_tensors=None)
        print(f"   Token IDs: {tokens_list}")

        # Encode with tensors
        encoded = tokenizer.encode(text, return_tensors="pt")
        print(f"   Input IDs shape: {encoded['input_ids'].shape}")
        print(f"   Input IDs: {encoded['input_ids']}")
        print(f"   Attention mask: {encoded['attention_mask']}")

        # Decode back
        decoded = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
        print(f"   Decoded: '{decoded}'")

        # Show with special tokens
        decoded_with_special = tokenizer.decode(
            encoded["input_ids"][0], skip_special_tokens=False
        )
        print(f"   With special tokens: '{decoded_with_special}'")


def demo_batch_encoding(tokenizer):
    """Demonstrate batch encoding."""
    print(f"\n📦 BATCH ENCODING")
    print("-" * 60)

    batch_texts = [
        "Great movie!",
        "This film is absolutely amazing and I loved every minute.",
        "Boring.",
        "The cinematography was stunning but the story was weak.",
    ]

    print(f"Encoding batch of {len(batch_texts)} texts:")
    for i, text in enumerate(batch_texts):
        print(f"   {i+1}. '{text}' (length: {len(text.split())} words)")

    # Encode batch
    encoded_batch = tokenizer.encode(batch_texts, return_tensors="pt")

    print(f"\nBatch encoding results:")
    print(f"   Input IDs shape: {encoded_batch['input_ids'].shape}")
    print(f"   Attention mask shape: {encoded_batch['attention_mask'].shape}")

    print(f"\nDetailed batch analysis:")
    for i, text in enumerate(batch_texts):
        input_ids = encoded_batch["input_ids"][i]
        attention_mask = encoded_batch["attention_mask"][i]

        # Count real tokens (non-padding)
        real_tokens = (attention_mask == 1).sum().item()
        padding_tokens = (attention_mask == 0).sum().item()

        print(
            f"   Text {i+1}: {real_tokens} real tokens, {padding_tokens} padding tokens"
        )
        print(
            f"      Input IDs: {input_ids[:10]}{'...' if len(input_ids) > 10 else ''}"
        )

    # Decode batch
    decoded_batch = tokenizer.batch_decode(
        encoded_batch["input_ids"], skip_special_tokens=True
    )
    print(f"\n🔄 Batch decoding:")
    for i, (original, decoded) in enumerate(zip(batch_texts, decoded_batch)):
        print(f"   {i+1}. Original: '{original}'")
        print(f"      Decoded:  '{decoded.strip()}'")


def demo_padding_strategies(tokenizer):
    """Demonstrate different padding strategies."""
    print(f"\n📏 PADDING STRATEGIES")
    print("-" * 60)

    test_texts = [
        "Short",
        "Medium length text here",
        "This is a much longer text that will demonstrate padding behavior",
    ]

    print(f"Test texts:")
    for i, text in enumerate(test_texts):
        length = tokenizer.get_sequence_length(text)
        print(f"   {i+1}. '{text}' (length: {length})")

    strategies = [
        ("max_length", {"padding": "max_length", "max_length": 15}),
        ("longest", {"padding": "longest"}),
        ("do_not_pad", {"padding": "do_not_pad"}),
    ]

    for strategy_name, params in strategies:
        print(f"\n📐 Strategy: {strategy_name}")

        if strategy_name == "do_not_pad":
            # For no padding, return as lists
            result = tokenizer.encode(test_texts, return_tensors=None, **params)
            print(f"   Result type: {type(result)}")
            for i, seq in enumerate(result):
                print(f"   Text {i+1}: length {len(seq)}")
        else:
            # For padding strategies, return as tensors
            result = tokenizer.encode(test_texts, return_tensors="pt", **params)
            print(f"   Shape: {result['input_ids'].shape}")
            for i in range(len(test_texts)):
                real_tokens = (result["attention_mask"][i] == 1).sum().item()
                padding_tokens = (result["attention_mask"][i] == 0).sum().item()
                print(
                    f"   Text {i+1}: {real_tokens} real + {padding_tokens} padding = {real_tokens + padding_tokens} total"
                )


def demo_truncation(tokenizer):
    """Demonstrate text truncation."""
    print(f"\nTEXT TRUNCATION")
    print("-" * 60)

    long_text = "This is an extremely long movie review that goes on and on about every single detail of the film including the acting performances cinematography music direction plot character development special effects and much more than anyone really wants to read in a single review"

    print(f"Original text: '{long_text}'")
    print(f"   Word count: {len(long_text.split())} words")
    print(f"   Full sequence length: {tokenizer.get_sequence_length(long_text)}")

    max_lengths = [10, 20, 30]

    for max_len in max_lengths:
        print(f"\n📏 Truncating to max_length={max_len}:")

        # Create tokenizer with specific max length
        truncated_result = tokenizer.encode(
            long_text, max_length=max_len, truncation=True, return_tensors="pt"
        )

        print(f"   Result shape: {truncated_result['input_ids'].shape}")

        # Decode to see what was kept
        decoded = tokenizer.decode(
            truncated_result["input_ids"][0], skip_special_tokens=True
        )
        print(f"   Truncated text: '{decoded}'")

        # Show special token preservation
        decoded_with_special = tokenizer.decode(
            truncated_result["input_ids"][0], skip_special_tokens=False
        )
        print(f"   With special tokens: '{decoded_with_special}'")


def demo_sequence_collator():
    """Demonstrate SequenceCollator for DataLoader."""
    print(f"\n🔗 SEQUENCE COLLATOR FOR DATALOADER")
    print("-" * 60)

    # Create simple vocabulary and tokenizer
    vocab = Vocabulary(min_frequency=1)
    texts = ["good movie", "bad film", "amazing cinema", "terrible acting"]
    vocab.build_from_texts(texts)

    tokenizer = Tokenizer(vocab, max_length=8, padding="longest")
    collator = SequenceCollator(tokenizer, max_length=10, padding="max_length")

    # Create sample batch (simulating DataLoader)
    sample_batch = [
        {"text": "This movie is great!", "label": 1},
        {"text": "Terrible film", "label": 0},
        {"text": "Amazing", "label": 1},
        {"text": "The worst movie I've ever seen", "label": 0},
    ]

    print(f"📦 Sample batch:")
    for i, item in enumerate(sample_batch):
        print(f"   {i+1}. Text: '{item['text']}', Label: {item['label']}")

    # Collate batch
    collated = collator(sample_batch)

    print(f"\nCollated batch:")
    print(f"   Input IDs shape: {collated['input_ids'].shape}")
    print(f"   Attention mask shape: {collated['attention_mask'].shape}")
    print(f"   Labels shape: {collated['labels'].shape}")
    print(f"   Labels: {collated['labels']}")

    # Show each sample in the batch
    for i in range(len(sample_batch)):
        input_ids = collated["input_ids"][i]
        attention_mask = collated["attention_mask"][i]
        label = collated["labels"][i]

        real_tokens = (attention_mask == 1).sum().item()

        print(f"\n   Sample {i+1}:")
        print(f"      Input IDs: {input_ids}")
        print(f"      Real tokens: {real_tokens}")
        print(f"      Label: {label.item()}")


def demo_sequence_analysis():
    """Demonstrate sequence length analysis."""
    print(f"\nSEQUENCE LENGTH ANALYSIS")
    print("-" * 60)

    # Create vocabulary and tokenizer
    vocab = Vocabulary(min_frequency=1)
    sample_texts = [
        "Great!",
        "This movie is good.",
        "I really enjoyed watching this film.",
        "The cinematography was absolutely stunning and beautiful.",
        "This is an extremely long review that goes into great detail about every aspect.",
        "Epic movie with amazing special effects, great acting, wonderful story, and perfect direction.",
    ]
    vocab.build_from_texts(sample_texts)

    preprocessor = create_default_preprocessor()
    tokenizer = create_tokenizer(vocab, preprocessor, max_length=50)

    # Analyze sequence lengths
    print(f"Analyzing {len(sample_texts)} sample texts...")

    stats = analyze_sequence_lengths(sample_texts, tokenizer)

    print(f"\nSequence Length Statistics:")
    print(f"   Count: {stats['count']}")
    print(f"   Mean: {stats['mean']:.2f}")
    print(f"   Std: {stats['std']:.2f}")
    print(f"   Min: {stats['min']}")
    print(f"   Max: {stats['max']}")
    print(f"   50th percentile: {stats['percentile_50']:.1f}")
    print(f"   90th percentile: {stats['percentile_90']:.1f}")
    print(f"   95th percentile: {stats['percentile_95']:.1f}")
    print(f"   99th percentile: {stats['percentile_99']:.1f}")

    # Show individual lengths
    print(f"\nIndividual sequence lengths:")
    for i, text in enumerate(sample_texts):
        length = tokenizer.get_sequence_length(text)
        print(f"   {i+1}. Length {length:2d}: '{text}'")

    # Recommendation for max_length
    recommended_max = int(stats["percentile_95"])
    print(f"\nRecommended max_length: {recommended_max} (covers 95% of sequences)")


def demo_utility_functions():
    """Demonstrate utility functions."""
    print(f"\nUTILITY FUNCTIONS")
    print("-" * 60)

    # Create test data
    vocab = Vocabulary(min_frequency=1)
    texts = ["great movie", "terrible film", "amazing story"]
    vocab.build_from_texts(texts)

    preprocessor = create_default_preprocessor()

    # Test create_sequences_from_texts
    print(f"Testing create_sequences_from_texts...")

    sample_texts = ["This is great!", "Terrible movie"]
    sample_labels = [1, 0]

    sequences = create_sequences_from_texts(
        texts=sample_texts,
        labels=sample_labels,
        vocabulary=vocab,
        preprocessor=preprocessor,
        max_length=10,
        padding="max_length",
    )

    print(f"   Input IDs shape: {sequences['input_ids'].shape}")
    print(f"   Attention mask shape: {sequences['attention_mask'].shape}")
    print(f"   Labels shape: {sequences['labels'].shape}")
    print(f"   Labels: {sequences['labels']}")

    # Show tensor details
    print(f"\nTensor details:")
    print(f"   Input IDs dtype: {sequences['input_ids'].dtype}")
    print(f"   Attention mask dtype: {sequences['attention_mask'].dtype}")
    print(f"   Labels dtype: {sequences['labels'].dtype}")

    # Test tensor operations
    print(f"\n⚡ Testing PyTorch operations:")

    # Example: computing sequence lengths from attention mask
    seq_lengths = sequences["attention_mask"].sum(dim=1)
    print(f"   Sequence lengths: {seq_lengths}")

    # Example: masking operations
    masked_input = sequences["input_ids"] * sequences["attention_mask"]
    print(f"   Masked input shape: {masked_input.shape}")

    # Example: batch statistics
    batch_size = sequences["input_ids"].size(0)
    seq_length = sequences["input_ids"].size(1)
    vocab_size = vocab.vocab_size

    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_length}")
    print(f"   Vocabulary size: {vocab_size}")


def main():
    """Run all tokenization demonstrations."""
    print("TOKENIZATION AND TEXT-TO-SEQUENCE DEMONSTRATION")
    print(
        "This script demonstrates converting text to tensors for PyTorch training."
    )
    print()

    try:
        # Basic tokenization setup
        vocab, preprocessor, tokenizer = demo_basic_tokenization()

        # Single text encoding
        demo_single_text_encoding(tokenizer)

        # Batch encoding
        demo_batch_encoding(tokenizer)

        # Padding strategies
        demo_padding_strategies(tokenizer)

        # Truncation
        demo_truncation(tokenizer)

        # Sequence collator
        demo_sequence_collator()

        # Sequence analysis
        demo_sequence_analysis()

        # Utility functions
        demo_utility_functions()

        print(f"\n" + "=" * 80)
        print("TOKENIZATION DEMONSTRATION COMPLETE")
        print("=" * 80)
        print()
        print("Key Features Demonstrated:")
        print("• Text-to-sequence conversion with vocabulary")
        print("• Sequence padding and truncation strategies")
        print("• Batch processing with attention masks")
        print("• Special token handling (SOS, EOS, PAD, UNK)")
        print("• PyTorch tensor integration")
        print("• SequenceCollator for DataLoader compatibility")
        print("• Sequence length analysis and optimization")
        print("• Encoding/decoding round-trip fidelity")
        print("• Preprocessing integration")
        print()
        print("Ready for PyTorch Dataset and DataLoader creation!")

    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

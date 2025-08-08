#!/usr/bin/env python3
"""
Demonstration script for PyTorch Dataset functionality.
Shows how to load sentiment analysis data for model training.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (AugmentedSentimentDataset, DatasetSplitter,
                              SentimentDataset, create_data_loaders,
                              load_sentiment_data)
from src.data.preprocessing import create_default_preprocessor
from src.data.tokenization import create_tokenizer
from src.data.vocabulary import Vocabulary, create_merged_vocabulary


def demo_basic_dataset():
    """Demonstrate basic dataset functionality."""
    print("=" * 80)
    print("BASIC DATASET DEMONSTRATION")
    print("=" * 80)

    # Create sample data
    print("Creating sample movie review dataset...")
    sample_data = {
        "review": [
            "This movie is absolutely fantastic! Amazing acting and great story.",
            "Terrible film. Waste of time and money. Completely boring.",
            "Great cinematography and excellent direction. Loved every minute.",
            "Disappointing plot and poor character development. Not recommended.",
            "Outstanding performances by all actors. A masterpiece of cinema.",
            "Awful script and terrible acting. One of the worst films ever.",
            "Beautiful visuals and compelling storyline. Highly entertaining.",
            "Boring and predictable. Nothing new or interesting to offer.",
            "Incredible film with amazing special effects and great music.",
            "Complete disaster. Poor writing and unconvincing performances.",
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
            "positive",
            "negative",
        ],
        "rating": [5, 1, 4, 2, 5, 1, 4, 2, 5, 1],
    }

    # Save to temporary CSV
    df = pd.DataFrame(sample_data)
    csv_path = Path("data/processed/demo_sample.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(f"Created dataset with {len(df)} samples")
    print(f"   Saved to: {csv_path}")

    # Create vocabulary and tokenizer
    print("\nCreating vocabulary and tokenizer...")
    vocab = Vocabulary(min_frequency=1, max_vocab_size=1000)
    preprocessor = create_default_preprocessor()
    vocab.build_from_texts(sample_data["review"], preprocessor)

    tokenizer = create_tokenizer(
        vocabulary=vocab, preprocessor=preprocessor, max_length=50, padding="max_length"
    )

    print(f"Vocabulary created with {vocab.vocab_size} words")
    print(f"Tokenizer configured with max_length=50")

    return csv_path, tokenizer, vocab


def demo_dataset_creation(csv_path, tokenizer):
    """Demonstrate dataset creation and basic operations."""
    print(f"\nDATASET CREATION AND OPERATIONS")
    print("-" * 60)

    # Create dataset
    dataset = SentimentDataset(
        data_path=csv_path,
        tokenizer=tokenizer,
        text_column="review",
        label_column="sentiment",
        max_length=50,
    )

    print(f"Dataset created with {len(dataset)} samples")
    print(f"   Text column: '{dataset.text_column}'")
    print(f"   Label column: '{dataset.label_column}'")
    print(f"   Label mapping: {dataset.label_mapping}")

    # Get sample information
    print(f"\nSample Information:")
    for i in range(min(3, len(dataset))):
        info = dataset.get_sample_info(i)
        print(f"   Sample {i+1}:")
        print(
            f"      Text: '{info['text'][:50]}{'...' if len(info['text']) > 50 else ''}'"
        )
        print(f"      Label: {info['label']} (original: '{info['original_label']}')")
        print(f"      Text length: {info['text_length']} chars")
        print(f"      Token count: {info['token_count']} tokens")

    # Get dataset statistics
    print(f"\nDataset Statistics:")
    label_dist = dataset.get_label_distribution()
    print(f"   Label distribution:")
    for label, count in label_dist.items():
        label_name = [k for k, v in dataset.label_mapping.items() if v == label][0]
        print(f"      {label} ({label_name}): {count} samples")

    text_stats = dataset.get_text_statistics()
    print(f"   Text statistics:")
    print(f"      Mean text length: {text_stats['text_length_mean']:.1f} chars")
    print(f"      Mean token length: {text_stats['token_length_mean']:.1f} tokens")
    print(f"      Max token length: {text_stats['token_length_max']} tokens")
    print(f"      95th percentile: {text_stats['token_length_95th']:.1f} tokens")

    return dataset


def demo_dataset_samples(dataset):
    """Demonstrate getting samples from dataset."""
    print(f"\nDATASET SAMPLE ACCESS")
    print("-" * 60)

    print(f"Examining first 3 samples:")

    for i in range(min(3, len(dataset))):
        print(f"\nSample {i+1}:")

        # Get raw sample info
        info = dataset.get_sample_info(i)
        print(
            f"   Original text: '{info['text'][:40]}{'...' if len(info['text']) > 40 else ''}'"
        )
        print(f"   Label: {info['label']} ({info['original_label']})")

        # Get processed sample (as would be used by DataLoader)
        sample = dataset[i]
        print(f"   Processed sample:")
        print(f"      Input IDs shape: {sample['input_ids'].shape}")
        print(f"      Attention mask shape: {sample['attention_mask'].shape}")
        print(f"      Label tensor: {sample['labels']}")
        print(f"      First 10 tokens: {sample['input_ids'][:10].tolist()}")

        # Show real vs padding tokens
        real_tokens = (sample["attention_mask"] == 1).sum().item()
        padding_tokens = (sample["attention_mask"] == 0).sum().item()
        print(f"      Real tokens: {real_tokens}, Padding tokens: {padding_tokens}")

        # Decode back to text
        decoded = dataset.tokenizer.decode(
            sample["input_ids"], skip_special_tokens=True
        )
        print(
            f"      Decoded text: '{decoded[:40]}{'...' if len(decoded) > 40 else ''}'"
        )


def demo_dataset_splitting(dataset):
    """Demonstrate dataset splitting."""
    print(f"\nDATASET SPLITTING")
    print("-" * 60)

    print(f"Splitting dataset ({len(dataset)} total samples):")
    print(f"   Training: 70% | Validation: 20% | Test: 10%")

    # Split dataset
    train_dataset, val_dataset, test_dataset = DatasetSplitter.split_dataset(
        dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=42
    )

    print(f"\nSplit results:")
    print(f"   Training set: {len(train_dataset)} samples")
    print(f"   Validation set: {len(val_dataset)} samples")
    print(f"   Test set: {len(test_dataset)} samples")

    # Verify splits don't overlap
    print(f"\nVerifying split integrity:")

    # Get indices for each split (if they're Subset objects)
    if hasattr(train_dataset, "indices"):
        train_indices = set(train_dataset.indices)
        val_indices = set(val_dataset.indices)
        test_indices = set(test_dataset.indices)

        # Check for overlaps
        train_val_overlap = train_indices.intersection(val_indices)
        train_test_overlap = train_indices.intersection(test_indices)
        val_test_overlap = val_indices.intersection(test_indices)

        print(f"   Train-Val overlap: {len(train_val_overlap)} (should be 0)")
        print(f"   Train-Test overlap: {len(train_test_overlap)} (should be 0)")
        print(f"   Val-Test overlap: {len(val_test_overlap)} (should be 0)")
        print(
            f"   Total unique indices: {len(train_indices.union(val_indices, test_indices))}"
        )

    return train_dataset, val_dataset, test_dataset


def demo_data_loaders(train_dataset, val_dataset, test_dataset, tokenizer):
    """Demonstrate DataLoader creation and usage."""
    print(f"\n📦 DATALOADER CREATION")
    print("-" * 60)

    print(f"Creating DataLoaders with batch_size=4:")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        tokenizer,
        batch_size=4,
        num_workers=0,
        shuffle_train=True,
    )

    print(f"DataLoaders created:")
    print(f"   Training: {len(train_loader)} batches")
    print(f"   Validation: {len(val_loader)} batches")
    print(f"   Test: {len(test_loader)} batches")

    # Demonstrate batch iteration
    print(f"\n🔄 Examining batches:")

    loaders = [
        ("Training", train_loader),
        ("Validation", val_loader),
        ("Test", test_loader),
    ]

    for loader_name, loader in loaders:
        if len(loader) > 0:
            print(f"\n   {loader_name} Loader:")
            batch = next(iter(loader))

            print(f"      Batch keys: {list(batch.keys())}")
            print(f"      Input IDs shape: {batch['input_ids'].shape}")
            print(f"      Attention mask shape: {batch['attention_mask'].shape}")
            print(f"      Labels shape: {batch['labels'].shape}")
            print(f"      Labels: {batch['labels'].tolist()}")

            # Show sequence length analysis
            seq_lengths = batch["attention_mask"].sum(dim=1)
            print(f"      Sequence lengths in batch: {seq_lengths.tolist()}")

            # Show a decoded sample from the batch
            sample_idx = 0
            decoded = tokenizer.decode(
                batch["input_ids"][sample_idx], skip_special_tokens=True
            )
            print(
                f"      Sample text: '{decoded[:50]}{'...' if len(decoded) > 50 else ''}'"
            )

    return train_loader, val_loader, test_loader


def demo_complete_pipeline():
    """Demonstrate complete data loading pipeline."""
    print(f"\nCOMPLETE DATA LOADING PIPELINE")
    print("-" * 60)

    # Check if IMDB data exists
    imdb_train_path = Path("data/processed/imdb_train.csv")
    imdb_test_path = Path("data/processed/imdb_test.csv")

    if imdb_train_path.exists() and imdb_test_path.exists():
        print(f"📁 Using IMDB dataset:")
        print(f"   Training: {imdb_train_path}")
        print(f"   Test: {imdb_test_path}")

        # Create vocabulary from both files
        print(f"\nCreating vocabulary from IMDB data...")
        vocab = create_merged_vocabulary(
            train_csv_path=imdb_train_path,
            test_csv_path=imdb_test_path,
            max_vocab_size=5000,
            min_frequency=5,
        )

        print(f"Vocabulary created with {vocab.vocab_size} words")

        # Use complete pipeline
        train_loader, val_loader, test_loader, full_dataset = load_sentiment_data(
            data_path=imdb_train_path,
            vocabulary=vocab,
            max_length=256,
            batch_size=16,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            cache_dir="data/cache",
            random_seed=42,
        )

        print(f"\nPipeline Results:")
        print(f"   Full dataset: {len(full_dataset)} samples")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")

        # Show sample batch
        print(f"\nSample training batch:")
        batch = next(iter(train_loader))
        print(f"   Batch size: {batch['input_ids'].size(0)}")
        print(f"   Sequence length: {batch['input_ids'].size(1)}")
        print(
            f"   Label distribution in batch: {torch.bincount(batch['labels']).tolist()}"
        )

        # Show text statistics
        stats = full_dataset.get_text_statistics()
        print(f"\nDataset statistics:")
        print(f"   Mean token length: {stats['token_length_mean']:.1f}")
        print(f"   95th percentile: {stats['token_length_95th']:.1f}")
        print(f"   Recommended max_length: {int(stats['token_length_95th'])}")

        return train_loader, val_loader, test_loader, full_dataset

    else:
        print(f"IMDB data not found. Please run data download script first:")
        print(f"   python src/data/download_data.py")
        return None, None, None, None


def demo_augmented_dataset(csv_path, tokenizer):
    """Demonstrate augmented dataset functionality."""
    print(f"\n🔄 AUGMENTED DATASET DEMONSTRATION")
    print("-" * 60)

    print(f"🎭 Creating augmented dataset with data augmentation:")

    # Create augmented dataset
    aug_dataset = AugmentedSentimentDataset(
        data_path=csv_path,
        tokenizer=tokenizer,
        augmentation_prob=0.5,
        augmentation_methods=[
            "synonym_replacement",
            "random_insertion",
            "random_deletion",
        ],
    )

    print(f"Augmented dataset created:")
    print(f"   Base samples: {len(aug_dataset)}")
    print(f"   Augmentation probability: {aug_dataset.augmentation_prob}")
    print(f"   Augmentation methods: {aug_dataset.augmentation_methods}")

    # Compare original vs augmented
    print(f"\nComparing original vs augmented samples:")

    for i in range(min(3, len(aug_dataset))):
        original_text = aug_dataset.samples[i]["text"]
        print(f"\nSample {i+1}:")
        print(f"   Original: '{original_text}'")

        # Generate several augmented versions
        augmented_versions = []
        for _ in range(3):
            augmented = aug_dataset._apply_augmentation(original_text)
            augmented_versions.append(augmented)

        for j, aug_text in enumerate(augmented_versions):
            if aug_text != original_text:
                print(f"   Augmented {j+1}: '{aug_text}'")
            else:
                print(f"   Augmented {j+1}: (no change)")

    # Test augmented data loader
    print(f"\n📦 Creating DataLoader with augmented dataset:")

    from torch.utils.data import DataLoader

    from src.data.tokenization import SequenceCollator

    collator = SequenceCollator(tokenizer, padding="longest")
    aug_loader = DataLoader(
        aug_dataset, batch_size=4, shuffle=True, collate_fn=collator
    )

    print(f"Augmented DataLoader created with {len(aug_loader)} batches")

    # Show augmented batch
    if len(aug_loader) > 0:
        batch = next(iter(aug_loader))
        print(f"   Batch shape: {batch['input_ids'].shape}")

        # Decode first sample to show augmentation effect
        decoded = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
        print(
            f"   Sample from batch: '{decoded[:60]}{'...' if len(decoded) > 60 else ''}'"
        )


def demo_performance_optimization():
    """Demonstrate performance optimization techniques."""
    print(f"\n⚡ PERFORMANCE OPTIMIZATION")
    print("-" * 60)

    print(f"Performance Tips:")
    print(f"   • Use caching for processed datasets")
    print(f"   • Set num_workers > 0 for faster data loading")
    print(f"   • Use pin_memory=True for GPU training")
    print(f"   • Choose optimal batch_size for your GPU memory")
    print(f"   • Consider sequence length distribution for max_length")
    print(f"   • Use drop_last=True for training stability")

    print(f"\nMemory Usage Estimates:")
    batch_sizes = [16, 32, 64, 128]
    seq_length = 256
    vocab_size = 10000

    for batch_size in batch_sizes:
        # Rough memory calculation for input tensors
        input_ids_mem = batch_size * seq_length * 4  # 4 bytes per int32
        attention_mask_mem = batch_size * seq_length * 4
        labels_mem = batch_size * 4
        total_mb = (input_ids_mem + attention_mask_mem + labels_mem) / (1024 * 1024)

        print(f"   Batch size {batch_size:3d}: ~{total_mb:.1f}MB per batch")

    print(f"\nRecommended Settings:")
    print(f"   • Small GPU (4-8GB): batch_size=16-32, max_length=256")
    print(f"   • Medium GPU (8-16GB): batch_size=32-64, max_length=512")
    print(f"   • Large GPU (16GB+): batch_size=64-128, max_length=512")


def main():
    """Run all dataset demonstrations."""
    print("PYTORCH DATASET DEMONSTRATION")
    print("This script demonstrates PyTorch Dataset functionality")
    print("    for sentiment analysis data loading and preprocessing.")
    print()

    try:
        # Basic dataset setup
        csv_path, tokenizer, vocab = demo_basic_dataset()

        # Dataset creation and operations
        dataset = demo_dataset_creation(csv_path, tokenizer)

        # Sample access
        demo_dataset_samples(dataset)

        # Dataset splitting
        train_dataset, val_dataset, test_dataset = demo_dataset_splitting(dataset)

        # DataLoader creation
        train_loader, val_loader, test_loader = demo_data_loaders(
            train_dataset, val_dataset, test_dataset, tokenizer
        )

        # Complete pipeline (if IMDB data available)
        demo_complete_pipeline()

        # Augmented dataset
        demo_augmented_dataset(csv_path, tokenizer)

        # Performance optimization
        demo_performance_optimization()

        print(f"\n" + "=" * 80)
        print("DATASET DEMONSTRATION COMPLETE")
        print("=" * 80)
        print()
        print("Key Features Demonstrated:")
        print("• PyTorch Dataset class for sentiment analysis")
        print("• Automatic text preprocessing and tokenization")
        print("• Flexible label mapping (binary/multi-class)")
        print("• Dataset splitting (train/validation/test)")
        print("• DataLoader integration with batching")
        print("• Caching for improved performance")
        print("• Data augmentation capabilities")
        print("• Complete data loading pipeline")
        print("• Memory usage optimization")
        print("• Integration with existing vocabulary/tokenizer")
        print()
        print("Ready for model training with PyTorch!")

        # Clean up demo file
        if csv_path.exists():
            csv_path.unlink()
            print(f"🧹 Cleaned up demo file: {csv_path}")

    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

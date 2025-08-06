"""
Custom PyTorch Dataset classes for sentiment analysis.
Handles data loading, preprocessing, and tokenization for model training.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from src.data.preprocessing import TextPreprocessor
from src.data.tokenization import SequenceCollator, Tokenizer
from src.data.vocabulary import Vocabulary


class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment analysis with integrated tokenization.

    Features:
    - Automatic text preprocessing and tokenization
    - Memory-efficient data loading from CSV/JSON files
    - Flexible label handling (binary, multi-class, string labels)
    - Caching support for processed data
    - Integration with custom Vocabulary and Tokenizer
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Tokenizer,
        text_column: str = "review",
        label_column: str = "sentiment",
        max_length: Optional[int] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        label_mapping: Optional[Dict[Any, int]] = None,
        filter_invalid: bool = True,
        transform: Optional[callable] = None,
    ):
        """
        Initialize sentiment dataset.

        Args:
            data_path: Path to CSV or JSON data file
            tokenizer: Tokenizer instance for text processing
            text_column: Name of column containing text data
            label_column: Name of column containing labels
            max_length: Maximum sequence length (overrides tokenizer setting)
            cache_dir: Directory to cache processed data
            label_mapping: Custom mapping from labels to integers
            filter_invalid: Whether to filter out invalid samples
            transform: Optional transform function applied to samples
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length or tokenizer.max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.filter_invalid = filter_invalid
        self.transform = transform

        # Load and process data
        self.data = self._load_data()
        self.label_mapping = label_mapping or self._create_label_mapping()
        self.samples = self._process_data()

        # Setup caching
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._setup_cache()

    def _load_data(self) -> pd.DataFrame:
        """Load data from file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        print(f"📁 Loading data from: {self.data_path}")

        if self.data_path.suffix.lower() == ".csv":
            data = pd.read_csv(self.data_path)
        elif self.data_path.suffix.lower() == ".json":
            data = pd.read_json(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        print(f"✅ Loaded {len(data)} samples")

        # Validate required columns
        if self.text_column not in data.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in data")
        if self.label_column not in data.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in data")

        return data

    def _create_label_mapping(self) -> Dict[Any, int]:
        """Create mapping from string labels to integers."""
        unique_labels = self.data[self.label_column].dropna().unique()

        # Handle common sentiment analysis label formats
        if len(unique_labels) == 2:
            # Binary classification
            if all(label in ["positive", "negative"] for label in unique_labels):
                mapping = {"negative": 0, "positive": 1}
            elif all(label in ["pos", "neg"] for label in unique_labels):
                mapping = {"neg": 0, "pos": 1}
            elif all(label in [0, 1] for label in unique_labels):
                mapping = {0: 0, 1: 1}
            else:
                # Default binary mapping
                mapping = {sorted(unique_labels)[0]: 0, sorted(unique_labels)[1]: 1}
        else:
            # Multi-class classification
            mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}

        print(f"📋 Label mapping created: {mapping}")
        return mapping

    def _process_data(self) -> List[Dict[str, Any]]:
        """Process raw data into samples."""
        print("🔄 Processing data samples...")

        samples = []
        invalid_count = 0

        for idx, row in tqdm(
            self.data.iterrows(), total=len(self.data), desc="Processing"
        ):
            try:
                # Extract text and label
                text = row[self.text_column]
                label = row[self.label_column]

                # Validate sample
                if pd.isna(text) or pd.isna(label):
                    if self.filter_invalid:
                        invalid_count += 1
                        continue
                    else:
                        text = str(text) if not pd.isna(text) else ""
                        label = 0 if pd.isna(label) else label

                # Convert text to string if needed
                text = str(text).strip()
                if not text and self.filter_invalid:
                    invalid_count += 1
                    continue

                # Map label to integer
                if label not in self.label_mapping:
                    if self.filter_invalid:
                        invalid_count += 1
                        continue
                    else:
                        # Assign unknown label to class 0
                        numeric_label = 0
                else:
                    numeric_label = self.label_mapping[label]

                # Create sample
                sample = {
                    "text": text,
                    "label": numeric_label,
                    "original_label": label,
                    "index": idx,
                }

                samples.append(sample)

            except Exception as e:
                if self.filter_invalid:
                    invalid_count += 1
                    print(f"⚠️ Error processing sample {idx}: {e}")
                    continue
                else:
                    raise

        if invalid_count > 0:
            print(f"⚠️ Filtered out {invalid_count} invalid samples")

        print(f"✅ Processed {len(samples)} valid samples")
        return samples

    def _setup_cache(self):
        """Setup caching system for processed data."""
        # Create cache key based on data file and tokenizer settings
        cache_key = f"{self.data_path.stem}_{hash(str(self.tokenizer.get_special_tokens_dict()))}"
        self.cache_file = self.cache_dir / f"{cache_key}.pkl"

        print(f"💾 Cache file: {self.cache_file}")

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with tokenized text and label tensors
        """
        if idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.samples)}"
            )

        sample = self.samples[idx]

        # Tokenize text
        encoded = self.tokenizer.encode(
            sample["text"],
            max_length=self.max_length,
            padding="max_length" if self.max_length else "do_not_pad",
            truncation=True,
            return_tensors="pt",
        )

        # Prepare output
        output = {
            "input_ids": encoded["input_ids"].squeeze(0),  # Remove batch dimension
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample["label"], dtype=torch.long),
        }

        # Apply transform if provided
        if self.transform:
            output = self.transform(output)

        return output

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get detailed information about a specific sample."""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range")

        sample = self.samples[idx]
        tokenized = self.tokenizer.encode(sample["text"], return_tensors=None)

        return {
            "index": idx,
            "text": sample["text"],
            "label": sample["label"],
            "original_label": sample["original_label"],
            "text_length": len(sample["text"]),
            "token_count": len(tokenized),
            "tokens": (
                tokenized[:10] if len(tokenized) > 10 else tokenized
            ),  # First 10 tokens
        }

    def get_label_distribution(self) -> Dict[int, int]:
        """Get distribution of labels in the dataset."""
        label_counts = {}
        for sample in self.samples:
            label = sample["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts

    def get_text_statistics(self) -> Dict[str, float]:
        """Get text length statistics."""
        text_lengths = [len(sample["text"]) for sample in self.samples]
        token_lengths = [
            len(self.tokenizer.encode(sample["text"], return_tensors=None))
            for sample in self.samples
        ]

        import numpy as np

        return {
            "text_length_mean": np.mean(text_lengths),
            "text_length_std": np.std(text_lengths),
            "text_length_min": np.min(text_lengths),
            "text_length_max": np.max(text_lengths),
            "token_length_mean": np.mean(token_lengths),
            "token_length_std": np.std(token_lengths),
            "token_length_min": np.min(token_lengths),
            "token_length_max": np.max(token_lengths),
            "token_length_95th": np.percentile(token_lengths, 95),
        }

    def save_cache(self, file_path: Optional[Union[str, Path]] = None):
        """Save processed samples to cache file."""
        cache_path = Path(file_path) if file_path else self.cache_file
        if cache_path:
            print(f"💾 Saving cache to: {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "samples": self.samples,
                        "label_mapping": self.label_mapping,
                        "data_path": str(self.data_path),
                        "text_column": self.text_column,
                        "label_column": self.label_column,
                    },
                    f,
                )
            print("✅ Cache saved successfully")

    def load_cache(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        """Load processed samples from cache file."""
        cache_path = Path(file_path) if file_path else self.cache_file
        if cache_path and cache_path.exists():
            print(f"📁 Loading cache from: {cache_path}")
            try:
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)

                # Validate cache compatibility
                if (
                    cached_data["data_path"] == str(self.data_path)
                    and cached_data["text_column"] == self.text_column
                    and cached_data["label_column"] == self.label_column
                ):

                    self.samples = cached_data["samples"]
                    self.label_mapping = cached_data["label_mapping"]
                    print("✅ Cache loaded successfully")
                    return True
                else:
                    print("⚠️ Cache incompatible with current settings")
                    return False
            except Exception as e:
                print(f"❌ Error loading cache: {e}")
                return False
        return False


class DatasetSplitter:
    """Utility class for splitting datasets into train/validation/test sets."""

    @staticmethod
    def split_dataset(
        dataset: SentimentDataset,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split dataset into train/validation/test sets.

        Args:
            dataset: Dataset to split
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_seed: Random seed for reproducible splits

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

        print(f"📊 Splitting dataset: {total_size} total samples")
        print(f"   Training: {train_size} samples ({train_ratio:.1%})")
        print(f"   Validation: {val_size} samples ({val_ratio:.1%})")
        print(f"   Test: {test_size} samples ({test_ratio:.1%})")

        # Set random seed for reproducible splits
        generator = torch.Generator().manual_seed(random_seed)

        return random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )

    @staticmethod
    def split_by_indices(
        dataset: SentimentDataset,
        train_indices: List[int],
        val_indices: List[int],
        test_indices: List[int],
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Split dataset using specific indices."""
        from torch.utils.data import Subset

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        return train_dataset, val_dataset, test_dataset


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    tokenizer: Tokenizer,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train/validation/test datasets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        tokenizer: Tokenizer for sequence collation
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create collator
    collator = SequenceCollator(
        tokenizer=tokenizer, padding="longest", return_tensors="pt"
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch for training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print(f"📦 Created DataLoaders:")
    print(f"   Training: {len(train_loader)} batches")
    print(f"   Validation: {len(val_loader)} batches")
    print(f"   Test: {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def load_sentiment_data(
    data_path: Union[str, Path],
    vocabulary: Vocabulary,
    preprocessor: Optional[TextPreprocessor] = None,
    text_column: str = "review",
    label_column: str = "sentiment",
    max_length: int = 512,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    cache_dir: Optional[Union[str, Path]] = None,
    num_workers: int = 0,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, SentimentDataset]:
    """
    Complete pipeline to load sentiment analysis data.

    Args:
        data_path: Path to data file
        vocabulary: Vocabulary for tokenization
        preprocessor: Text preprocessor
        text_column: Name of text column
        label_column: Name of label column
        max_length: Maximum sequence length
        batch_size: Batch size for DataLoaders
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        cache_dir: Cache directory for processed data
        num_workers: Number of data loading workers
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader, full_dataset)
    """
    from src.data.tokenization import create_tokenizer

    print("🚀 Loading sentiment analysis data pipeline...")

    # Create tokenizer
    tokenizer = create_tokenizer(
        vocabulary=vocabulary,
        preprocessor=preprocessor,
        max_length=max_length,
        padding="max_length",
        add_special_tokens=True,
    )

    # Create dataset
    dataset = SentimentDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
        cache_dir=cache_dir,
    )

    # Split dataset
    train_dataset, val_dataset, test_dataset = DatasetSplitter.split_dataset(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print("✅ Data loading pipeline complete!")

    return train_loader, val_loader, test_loader, dataset


class AugmentedSentimentDataset(SentimentDataset):
    """
    Extended sentiment dataset with data augmentation capabilities.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Tokenizer,
        augmentation_prob: float = 0.1,
        augmentation_methods: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize augmented dataset.

        Args:
            data_path: Path to data file
            tokenizer: Tokenizer instance
            augmentation_prob: Probability of applying augmentation
            augmentation_methods: List of augmentation methods to use
            **kwargs: Additional arguments for base dataset
        """
        super().__init__(data_path, tokenizer, **kwargs)
        self.augmentation_prob = augmentation_prob
        self.augmentation_methods = augmentation_methods or ["synonym_replacement"]

        print(f"🔄 Augmentation enabled with probability {augmentation_prob}")
        print(f"   Methods: {self.augmentation_methods}")

    def _apply_augmentation(self, text: str) -> str:
        """Apply random augmentation to text."""
        import random

        if random.random() > self.augmentation_prob:
            return text

        # Simple augmentation methods
        if "synonym_replacement" in self.augmentation_methods:
            text = self._synonym_replacement(text)

        if "random_insertion" in self.augmentation_methods:
            text = self._random_insertion(text)

        if "random_deletion" in self.augmentation_methods:
            text = self._random_deletion(text)

        return text

    def _synonym_replacement(self, text: str) -> str:
        """Simple synonym replacement (placeholder implementation)."""
        # This is a simplified version - in practice, you'd use a proper synonym library
        synonyms = {
            "good": ["great", "excellent", "wonderful"],
            "bad": ["terrible", "awful", "horrible"],
            "movie": ["film", "picture", "cinema"],
            "amazing": ["incredible", "fantastic", "outstanding"],
        }

        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in synonyms:
                import random

                if random.random() < 0.3:  # 30% chance to replace
                    words[i] = random.choice(synonyms[word.lower()])

        return " ".join(words)

    def _random_insertion(self, text: str) -> str:
        """Insert random common words."""
        import random

        words = text.split()
        if len(words) < 3:
            return text

        insert_words = ["really", "very", "quite", "extremely", "absolutely"]
        if random.random() < 0.2:  # 20% chance
            insert_pos = random.randint(1, len(words) - 1)
            words.insert(insert_pos, random.choice(insert_words))

        return " ".join(words)

    def _random_deletion(self, text: str) -> str:
        """Randomly delete words."""
        import random

        words = text.split()
        if len(words) <= 3:
            return text

        # Delete up to 10% of words
        delete_count = max(1, int(0.1 * len(words)))
        for _ in range(delete_count):
            if len(words) > 3 and random.random() < 0.1:  # 10% chance per word
                del_idx = random.randint(0, len(words) - 1)
                words.pop(del_idx)

        return " ".join(words)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample with potential augmentation."""
        sample = self.samples[idx]

        # Apply augmentation
        text = self._apply_augmentation(sample["text"])

        # Tokenize augmented text
        encoded = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding="max_length" if self.max_length else "do_not_pad",
            truncation=True,
            return_tensors="pt",
        )

        output = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample["label"], dtype=torch.long),
        }

        if self.transform:
            output = self.transform(output)

        return output

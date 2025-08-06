#!/usr/bin/env python3
"""
Download and process IMDB movie reviews dataset for sentiment analysis.
This script downloads the dataset and prepares it for training.
"""

import os
import ssl
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

# Handle SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context


class DownloadProgressBar(tqdm):
    """Progress bar for file downloads."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = None) -> None:
        """Update progress bar."""
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, desc: str = None) -> None:
    """Download a file with progress bar."""
    print(f"Downloading {desc or 'file'}...")

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_tar(tar_path: Path, extract_to: Path) -> None:
    """Extract a tar file."""
    print(f"Extracting {tar_path.name}...")

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_to)


def read_imdb_files(data_dir: Path, split: str) -> List[Tuple[str, int]]:
    """Read IMDB review files and return list of (review, label) tuples."""
    reviews = []

    # Read positive reviews
    pos_dir = data_dir / split / "pos"
    if pos_dir.exists():
        for file_path in pos_dir.glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                review = f.read().strip()
                reviews.append((review, 1))  # 1 for positive

    # Read negative reviews
    neg_dir = data_dir / split / "neg"
    if neg_dir.exists():
        for file_path in neg_dir.glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                review = f.read().strip()
                reviews.append((review, 0))  # 0 for negative

    return reviews


def create_csv_dataset(reviews: List[Tuple[str, int]], output_path: Path) -> None:
    """Create CSV dataset from reviews."""
    df = pd.DataFrame(reviews, columns=["review", "sentiment"])

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} reviews to {output_path}")


def main() -> None:
    """Main function to download and process IMDB dataset."""
    # URLs and paths
    IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    # Create directories
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset
    tar_path = raw_dir / "aclImdb_v1.tar.gz"

    if not tar_path.exists():
        download_file(IMDB_URL, tar_path, "IMDB dataset")
    else:
        print(f"Dataset already downloaded: {tar_path}")

    # Extract dataset
    extract_dir = raw_dir / "aclImdb"
    if not extract_dir.exists():
        extract_tar(tar_path, raw_dir)
    else:
        print(f"Dataset already extracted: {extract_dir}")

    # Process training data
    train_csv = processed_dir / "imdb_train.csv"
    if not train_csv.exists():
        print("Processing training data...")
        train_reviews = read_imdb_files(extract_dir, "train")
        create_csv_dataset(train_reviews, train_csv)
    else:
        print(f"Training data already processed: {train_csv}")

    # Process test data
    test_csv = processed_dir / "imdb_test.csv"
    if not test_csv.exists():
        print("Processing test data...")
        test_reviews = read_imdb_files(extract_dir, "test")
        create_csv_dataset(test_reviews, test_csv)
    else:
        print(f"Test data already processed: {test_csv}")

    # Print summary
    if train_csv.exists() and test_csv.exists():
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        print("\n" + "=" * 50)
        print("DATASET SUMMARY")
        print("=" * 50)
        print(f"Training samples: {len(train_df):,}")
        print(f"Test samples: {len(test_df):,}")
        print(f"Training positive: {(train_df['sentiment'] == 1).sum():,}")
        print(f"Training negative: {(train_df['sentiment'] == 0).sum():,}")
        print(f"Test positive: {(test_df['sentiment'] == 1).sum():,}")
        print(f"Test negative: {(test_df['sentiment'] == 0).sum():,}")
        print("=" * 50)

        # Show sample reviews
        print("\nSample reviews:")
        print(
            f"Positive: {train_df[train_df['sentiment'] == 1].iloc[0]['review'][:100]}..."
        )
        print(
            f"Negative: {train_df[train_df['sentiment'] == 0].iloc[0]['review'][:100]}..."
        )


if __name__ == "__main__":
    main()

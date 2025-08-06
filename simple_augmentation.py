#!/usr/bin/env python3
"""Simple data augmentation for sentiment analysis without NLTK dependencies."""

import random
import re
import pandas as pd
from pathlib import Path


class SimpleTextAugmenter:
    """Simple text augmentation for sentiment analysis."""
    
    def __init__(self, augmentation_prob=0.3):
        self.augmentation_prob = augmentation_prob
        
        # Simple synonym dictionaries for sentiment words
        self.positive_synonyms = {
            'good': ['great', 'excellent', 'fantastic', 'amazing', 'wonderful'],
            'great': ['excellent', 'fantastic', 'amazing', 'wonderful', 'outstanding'],
            'excellent': ['fantastic', 'amazing', 'wonderful', 'outstanding', 'superb'],
            'fantastic': ['amazing', 'wonderful', 'outstanding', 'superb', 'brilliant'],
            'amazing': ['wonderful', 'outstanding', 'superb', 'brilliant', 'incredible'],
            'wonderful': ['outstanding', 'superb', 'brilliant', 'incredible', 'marvelous'],
            'bad': ['terrible', 'awful', 'horrible', 'dreadful', 'atrocious'],
            'terrible': ['awful', 'horrible', 'dreadful', 'atrocious', 'abysmal'],
            'awful': ['horrible', 'dreadful', 'atrocious', 'abysmal', 'appalling'],
            'horrible': ['dreadful', 'atrocious', 'abysmal', 'appalling', 'disgusting'],
            'love': ['adore', 'enjoy', 'like', 'appreciate', 'cherish'],
            'hate': ['dislike', 'loathe', 'despise', 'abhor', 'detest'],
            'best': ['finest', 'greatest', 'top', 'superior', 'premium'],
            'worst': ['poorest', 'lowest', 'inferior', 'substandard', 'terrible'],
        }
    
    def augment_text(self, text: str, label: int):
        """Augment a single text sample."""
        augmented_samples = [(text, label)]  # Keep original
        
        # Apply different augmentation techniques
        if random.random() < self.augmentation_prob:
            augmented_samples.append((self._synonym_replacement(text), label))
        
        if random.random() < self.augmentation_prob:
            augmented_samples.append((self._random_insertion(text), label))
        
        if random.random() < self.augmentation_prob:
            augmented_samples.append((self._random_deletion(text), label))
        
        if random.random() < self.augmentation_prob:
            augmented_samples.append((self._random_swap(text), label))
        
        return augmented_samples
    
    def _simple_tokenize(self, text: str):
        """Simple tokenization without NLTK."""
        # Split on whitespace and punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _synonym_replacement(self, text: str, n=1):
        """Replace n words with synonyms."""
        words = self._simple_tokenize(text)
        n = min(n, len(words))
        
        for _ in range(n):
            if len(words) == 0:
                break
            
            # Find sentiment words first
            sentiment_words = [word for word in words if word in self.positive_synonyms]
            
            if sentiment_words:
                word = random.choice(sentiment_words)
                synonyms = self.positive_synonyms.get(word, [])
                if synonyms:
                    synonym = random.choice(synonyms)
                    # Replace the word in the original text (case-insensitive)
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    text = pattern.sub(synonym, text, count=1)
        
        return text
    
    def _random_insertion(self, text: str, n=1):
        """Insert n random words."""
        words = self._simple_tokenize(text)
        n = min(n, len(words))
        
        for _ in range(n):
            if len(words) == 0:
                break
            
            # Insert a random word from the text
            random_word = random.choice(words)
            # Find a random position to insert
            insert_pos = random.randint(0, len(text.split()))
            text_parts = text.split()
            text_parts.insert(insert_pos, random_word)
            text = ' '.join(text_parts)
        
        return text
    
    def _random_deletion(self, text: str, p=0.1):
        """Randomly delete words with probability p."""
        words = text.split()
        
        # Keep important words (sentiment words)
        important_words = set(self.positive_synonyms.keys())
        
        remaining_words = []
        for word in words:
            if word.lower() in important_words or random.random() > p:
                remaining_words.append(word)
        
        return ' '.join(remaining_words) if remaining_words else text
    
    def _random_swap(self, text: str, n=1):
        """Randomly swap n pairs of adjacent words."""
        words = text.split()
        n = min(n, len(words) - 1)
        
        for _ in range(n):
            if len(words) < 2:
                break
            
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        
        return ' '.join(words)


def create_augmented_csv(input_file: str, output_file: str, augmentation_prob: float = 0.3):
    """Create augmented dataset CSV file."""
    # Read original data
    df = pd.read_csv(input_file)
    
    # Create augmenter
    augmenter = SimpleTextAugmenter(augmentation_prob=augmentation_prob)
    
    # Augment data
    augmented_texts = []
    augmented_labels = []
    
    print(f"Augmenting {len(df)} samples...")
    
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 1000 == 0:
            print(f"Processing sample {i}/{len(df)}")
        
        text = row['review']
        label = row['sentiment']
        
        augmented_samples = augmenter.augment_text(text, label)
        for aug_text, aug_label in augmented_samples:
            augmented_texts.append(aug_text)
            augmented_labels.append(aug_label)
    
    # Create new dataframe
    augmented_df = pd.DataFrame({
        'review': augmented_texts,
        'sentiment': augmented_labels
    })
    
    # Save to file
    augmented_df.to_csv(output_file, index=False)
    print(f"✅ Augmented dataset saved to {output_file}")
    print(f"   Original samples: {len(df)}")
    print(f"   Augmented samples: {len(augmented_df)}")
    print(f"   Increase: {len(augmented_df) / len(df):.1f}x")


if __name__ == "__main__":
    # Create augmented dataset
    create_augmented_csv(
        "data/processed/imdb_train.csv",
        "data/processed/imdb_train_augmented.csv",
        augmentation_prob=0.3
    ) 
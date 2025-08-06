"""Data augmentation techniques for sentiment analysis."""

import random
import re
from typing import List, Tuple
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextAugmenter:
    """Text augmentation for sentiment analysis."""
    
    def __init__(self, augmentation_prob=0.3):
        self.augmentation_prob = augmentation_prob
        
        # Synonym dictionaries for sentiment words
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
        }
    
    def augment_text(self, text: str, label: int) -> List[Tuple[str, int]]:
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
    
    def _synonym_replacement(self, text: str, n=1) -> str:
        """Replace n words with synonyms."""
        # Use simple word splitting instead of NLTK tokenization
        words = text.split()
        n = min(n, len(words))
        
        for _ in range(n):
            if len(words) == 0:
                break
            
            # Find sentiment words first
            sentiment_words = [word for word in words if word.lower() in self.positive_synonyms]
            
            if sentiment_words:
                word = random.choice(sentiment_words)
                synonyms = self.positive_synonyms.get(word.lower(), [])
                if synonyms:
                    synonym = random.choice(synonyms)
                    words = [synonym if w == word else w for w in words]
            else:
                # Use WordNet for general synonyms
                idx = random.randint(0, len(words) - 1)
                word = words[idx]
                synonyms = self._get_synonyms(word)
                if synonyms:
                    words[idx] = random.choice(synonyms)
        
        return ' '.join(words)
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms using WordNet."""
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms.append(lemma.name())
        return list(set(synonyms))[:5]  # Limit to 5 synonyms
    
    def _random_insertion(self, text: str, n=1) -> str:
        """Insert n random words."""
        words = text.split()
        n = min(n, len(words))
        
        for _ in range(n):
            if len(words) == 0:
                break
            
            # Insert a random word from the text
            random_word = random.choice(words)
            random_idx = random.randint(0, len(words))
            words.insert(random_idx, random_word)
        
        return ' '.join(words)
    
    def _random_deletion(self, text: str, p=0.1) -> str:
        """Randomly delete words with probability p."""
        words = text.split()
        
        # Keep important words (sentiment words)
        important_words = set(self.positive_synonyms.keys())
        
        remaining_words = []
        for word in words:
            if word.lower() in important_words or random.random() > p:
                remaining_words.append(word)
        
        return ' '.join(remaining_words) if remaining_words else text
    
    def _random_swap(self, text: str, n=1) -> str:
        """Randomly swap n pairs of adjacent words."""
        words = text.split()
        n = min(n, len(words) - 1)
        
        for _ in range(n):
            if len(words) < 2:
                break
            
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        
        return ' '.join(words)


def augment_dataset(texts: List[str], labels: List[int], augmenter: TextAugmenter = None) -> Tuple[List[str], List[int]]:
    """Augment entire dataset."""
    if augmenter is None:
        augmenter = TextAugmenter()
    
    augmented_texts = []
    augmented_labels = []
    
    for text, label in zip(texts, labels):
        augmented_samples = augmenter.augment_text(text, label)
        for aug_text, aug_label in augmented_samples:
            augmented_texts.append(aug_text)
            augmented_labels.append(aug_label)
    
    return augmented_texts, augmented_labels


def create_augmented_csv(input_file: str, output_file: str, augmentation_prob: float = 0.3):
    """Create augmented dataset CSV file."""
    import pandas as pd
    
    # Read original data
    df = pd.read_csv(input_file)
    
    # Create augmenter
    augmenter = TextAugmenter(augmentation_prob=augmentation_prob)
    
    # Augment data
    augmented_texts = []
    augmented_labels = []
    
    for _, row in df.iterrows():
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
    # Example usage
    create_augmented_csv(
        "data/processed/imdb_train.csv",
        "data/processed/imdb_train_augmented.csv",
        augmentation_prob=0.3
    ) 
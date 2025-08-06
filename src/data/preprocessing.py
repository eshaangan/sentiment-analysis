"""
Text preprocessing utilities for sentiment analysis.
Handles cleaning, normalization, and tokenization of text data.
"""

import re
import ssl
import string
from typing import List, Optional, Union

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Handle SSL certificate issues for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


class TextPreprocessor:
    """
    Comprehensive text preprocessing for sentiment analysis.

    Features:
    - HTML tag removal
    - Special character handling
    - Text normalization
    - Tokenization
    - Stop word removal (optional)
    - Contraction expansion
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_html: bool = True,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        remove_stopwords: bool = False,
        expand_contractions: bool = True,
        min_word_length: int = 2,
        max_word_length: int = 50,
    ):
        """
        Initialize the text preprocessor.

        Args:
            lowercase: Convert text to lowercase
            remove_html: Remove HTML tags
            remove_punctuation: Remove punctuation marks
            remove_numbers: Remove numeric characters
            remove_stopwords: Remove English stop words
            expand_contractions: Expand contractions (e.g., "don't" -> "do not")
            min_word_length: Minimum word length to keep
            max_word_length: Maximum word length to keep
        """
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.expand_contractions = expand_contractions
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length

        # Initialize NLTK components
        self._download_nltk_data()

        if self.remove_stopwords:
            if self._has_stopwords:
                self.stop_words = set(stopwords.words("english"))
            else:
                # Fallback stopwords list
                self.stop_words = {
                    "i",
                    "me",
                    "my",
                    "myself",
                    "we",
                    "our",
                    "ours",
                    "ourselves",
                    "you",
                    "your",
                    "yours",
                    "yourself",
                    "yourselves",
                    "he",
                    "him",
                    "his",
                    "himself",
                    "she",
                    "her",
                    "hers",
                    "herself",
                    "it",
                    "its",
                    "itself",
                    "they",
                    "them",
                    "their",
                    "theirs",
                    "themselves",
                    "what",
                    "which",
                    "who",
                    "whom",
                    "this",
                    "that",
                    "these",
                    "those",
                    "am",
                    "is",
                    "are",
                    "was",
                    "were",
                    "be",
                    "been",
                    "being",
                    "have",
                    "has",
                    "had",
                    "having",
                    "do",
                    "does",
                    "did",
                    "doing",
                    "a",
                    "an",
                    "the",
                    "and",
                    "but",
                    "if",
                    "or",
                    "because",
                    "as",
                    "until",
                    "while",
                    "of",
                    "at",
                    "by",
                    "for",
                    "with",
                    "through",
                    "during",
                    "before",
                    "after",
                    "above",
                    "below",
                    "up",
                    "down",
                    "out",
                    "off",
                    "on",
                    "over",
                    "under",
                    "again",
                    "further",
                    "then",
                    "once",
                }
        else:
            self.stop_words = set()

        # Contraction mapping
        self.contractions = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "couldn't": "could not",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'll": "he will",
            "he's": "he is",
            "i'd": "i would",
            "i'll": "i will",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'll": "it will",
            "it's": "it is",
            "let's": "let us",
            "might've": "might have",
            "mustn't": "must not",
            "shan't": "shall not",
            "she'd": "she would",
            "she'll": "she will",
            "she's": "she is",
            "shouldn't": "should not",
            "that's": "that is",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "we'd": "we would",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what's": "what is",
            "where's": "where is",
            "who's": "who is",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have",
        }

    def _download_nltk_data(self) -> None:
        """Download required NLTK data with fallback handling."""
        # Try to download punkt tokenizer
        try:
            nltk.data.find("tokenizers/punkt")
            self._has_punkt = True
        except LookupError:
            try:
                print("Downloading NLTK punkt tokenizer...")
                nltk.download("punkt", quiet=True)
                self._has_punkt = True
            except Exception as e:
                print(f"Warning: Could not download punkt tokenizer: {e}")
                self._has_punkt = False

        # Try to download stopwords
        try:
            nltk.data.find("corpora/stopwords")
            self._has_stopwords = True
        except LookupError:
            try:
                print("Downloading NLTK stopwords...")
                nltk.download("stopwords", quiet=True)
                self._has_stopwords = True
            except Exception as e:
                print(f"Warning: Could not download stopwords: {e}")
                self._has_stopwords = False

    def remove_html_tags(self, text: str) -> str:
        """
        Remove HTML tags from text.

        Args:
            text: Input text with potential HTML tags

        Returns:
            Text with HTML tags removed
        """
        if not self.remove_html:
            return text

        # Remove HTML tags
        clean_text = re.sub(r"<[^>]+>", "", text)

        # Remove HTML entities
        clean_text = re.sub(r"&[a-zA-Z]+;", "", clean_text)

        return clean_text

    def expand_contractions_text(self, text: str) -> str:
        """
        Expand contractions in text.

        Args:
            text: Input text with contractions

        Returns:
            Text with expanded contractions
        """
        if not self.expand_contractions:
            return text

        # Create pattern for contractions
        contractions_pattern = re.compile(
            "({})".format("|".join(re.escape(key) for key in self.contractions.keys())),
            flags=re.IGNORECASE | re.DOTALL,
        )

        def expand_match(contraction: re.Match[str]) -> str:
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = self.contractions.get(match.lower(), match)

            # Preserve original case
            if first_char.isupper():
                expanded_contraction = expanded_contraction.capitalize()

            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)
        return expanded_text

    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.

        Args:
            text: Input text

        Returns:
            Text with normalized whitespace
        """
        # Replace multiple whitespace with single space
        text = re.sub(r"\s+", " ", text)

        # Strip leading and trailing whitespace
        text = text.strip()

        return text

    def remove_special_characters(self, text: str) -> str:
        """
        Remove or normalize special characters.

        Args:
            text: Input text

        Returns:
            Text with special characters handled
        """
        # Remove extra punctuation (multiple consecutive punctuation marks)
        text = re.sub(r"[^\w\s" + string.punctuation + "]+", "", text)

        # Handle repeated punctuation
        text = re.sub(r"([.!?]){2,}", r"\1", text)

        # Remove very long sequences of the same character
        text = re.sub(r"(.)\1{4,}", r"\1\1\1", text)

        return text

    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning operations to text.

        Args:
            text: Raw input text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)  # type: ignore  # Defensive programming

        # Remove HTML tags
        text = self.remove_html_tags(text)

        # Expand contractions
        text = self.expand_contractions_text(text)

        # Remove special characters
        text = self.remove_special_characters(text)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove numbers if specified
        if self.remove_numbers:
            text = re.sub(r"\d+", "", text)

        # Remove punctuation if specified
        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # Normalize whitespace
        text = self.normalize_whitespace(text)

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Use NLTK tokenizer if available, otherwise use simple tokenizer
        if self._has_punkt:
            try:
                tokens = word_tokenize(text)
            except LookupError:
                # Fallback to simple tokenization
                tokens = self._simple_tokenize(text)
        else:
            tokens = self._simple_tokenize(text)

        # Filter tokens by length
        tokens = [
            token
            for token in tokens
            if self.min_word_length <= len(token) <= self.max_word_length
        ]

        # Remove stop words if specified
        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]

        return tokens

    def _simple_tokenize(self, text: str) -> List[str]:
        """
        Simple fallback tokenizer using regex.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Split on whitespace and punctuation, but keep punctuation as separate tokens
        tokens = []

        # Find all words and punctuation
        pattern = r"\w+|[^\w\s]"
        matches = re.findall(pattern, text)

        for match in matches:
            if match.strip():  # Only add non-empty tokens
                tokens.append(match)

        return tokens

    def preprocess(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to text.

        Args:
            text: Raw input text

        Returns:
            Preprocessed text as string
        """
        # Clean the text
        cleaned_text = self.clean_text(text)

        return cleaned_text

    def preprocess_and_tokenize(self, text: str) -> List[str]:
        """
        Apply full preprocessing pipeline and return tokens.

        Args:
            text: Raw input text

        Returns:
            List of preprocessed tokens
        """
        # Clean the text
        cleaned_text = self.clean_text(text)

        # Tokenize
        tokens = self.tokenize(cleaned_text)

        return tokens

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of raw texts

        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]

    def preprocess_and_tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess and tokenize a batch of texts.

        Args:
            texts: List of raw texts

        Returns:
            List of lists of tokens
        """
        return [self.preprocess_and_tokenize(text) for text in texts]


def create_default_preprocessor() -> TextPreprocessor:
    """
    Create a default preprocessor with sensible settings for sentiment analysis.

    Returns:
        Configured TextPreprocessor instance
    """
    return TextPreprocessor(
        lowercase=True,
        remove_html=True,
        remove_punctuation=False,  # Keep punctuation for sentiment
        remove_numbers=False,  # Numbers might be relevant
        remove_stopwords=False,  # Stop words can carry sentiment
        expand_contractions=True,
        min_word_length=2,
        max_word_length=50,
    )


def create_minimal_preprocessor() -> TextPreprocessor:
    """
    Create a minimal preprocessor that only does basic cleaning.

    Returns:
        Configured TextPreprocessor instance
    """
    return TextPreprocessor(
        lowercase=True,
        remove_html=True,
        remove_punctuation=False,
        remove_numbers=False,
        remove_stopwords=False,
        expand_contractions=False,
        min_word_length=1,
        max_word_length=100,
    )


def create_aggressive_preprocessor() -> TextPreprocessor:
    """
    Create an aggressive preprocessor for maximum cleaning.

    Returns:
        Configured TextPreprocessor instance
    """
    return TextPreprocessor(
        lowercase=True,
        remove_html=True,
        remove_punctuation=True,
        remove_numbers=True,
        remove_stopwords=True,
        expand_contractions=True,
        min_word_length=3,
        max_word_length=30,
    )

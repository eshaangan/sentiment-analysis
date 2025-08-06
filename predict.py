#!/usr/bin/env python3
"""Simple prediction script for sentiment analysis."""

import argparse
import logging
from pathlib import Path

import torch

from src.data.preprocessing import create_default_preprocessor
from src.data.vocabulary import create_vocabulary_from_data
from src.data.tokenization import create_tokenizer
from src.models.lstm_model import LSTMConfig, LSTMModel
from src.training.utils import get_device

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path, vocabulary, device):
    """Load the trained model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = LSTMConfig(
        vocab_size=vocabulary.vocab_size,
        embed_dim=128,
        hidden_dim=256,
        output_dim=2,
        bidirectional=True,
        pooling="attention",
    )
    model = LSTMModel(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def predict_sentiment(text, model, tokenizer, device):
    """Predict sentiment for a single text."""
    model.eval()
    
    with torch.no_grad():
        # Tokenize the text
        encoded = tokenizer.encode(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        
        # Get prediction
        logits = model(input_ids)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1)
        
        # Get confidence
        confidence = torch.max(probabilities, dim=1)[0]
        
        return {
            "sentiment": "positive" if prediction.item() == 1 else "negative",
            "confidence": confidence.item(),
            "probabilities": {
                "negative": probabilities[0][0].item(),
                "positive": probabilities[0][1].item()
            }
        }


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Predict sentiment for text")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--text", help="Text to analyze")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    # Set up data pipeline
    logger.info("Setting up data pipeline...")
    preprocessor = create_default_preprocessor()
    
    # Create vocabulary (using train data)
    vocabulary = create_vocabulary_from_data(
        "data/processed/imdb_train.csv",
        "data/processed/imdb_test.csv",
        text_column="review",
        max_vocab_size=10000,
        min_frequency=2,
        preprocessor=preprocessor,
    )

    # Create tokenizer
    tokenizer = create_tokenizer(
        vocabulary=vocabulary, 
        max_length=512
    )

    # Load model
    logger.info("Loading model...")
    model = load_model(checkpoint_path, vocabulary, device)

    if args.interactive:
        print("\n🎬 Sentiment Analysis Interactive Mode")
        print("Enter text to analyze (or 'quit' to exit):")
        print("-" * 50)
        
        while True:
            try:
                text = input("\nEnter text: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                if not text:
                    continue
                
                result = predict_sentiment(text, model, tokenizer, device)
                print(f"\n📊 Analysis Results:")
                print(f"   Sentiment: {result['sentiment'].upper()}")
                print(f"   Confidence: {result['confidence']:.2%}")
                print(f"   Negative: {result['probabilities']['negative']:.2%}")
                print(f"   Positive: {result['probabilities']['positive']:.2%}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\n👋 Goodbye!")
        
    elif args.text:
        result = predict_sentiment(args.text, model, tokenizer, device)
        print(f"\n📊 Analysis Results for: '{args.text}'")
        print(f"   Sentiment: {result['sentiment'].upper()}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Negative: {result['probabilities']['negative']:.2%}")
        print(f"   Positive: {result['probabilities']['positive']:.2%}")
        
    else:
        print("Please provide either --text or --interactive flag")


if __name__ == "__main__":
    main() 
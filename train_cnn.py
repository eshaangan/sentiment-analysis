#!/usr/bin/env python3
"""Train CNN model for sentiment analysis."""

import argparse
import logging
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.preprocessing import create_default_preprocessor
from src.data.vocabulary import create_vocabulary_from_data
from src.data.tokenization import create_tokenizer
from src.data.dataset import load_sentiment_data
from src.models.cnn_model import CNNConfig, CNNModel
from src.training.trainer import Trainer
from src.training.schedulers import create_scheduler
from src.training.early_stopping import EarlyStopping
from src.training.utils import get_device

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main training function for CNN model."""
    parser = argparse.ArgumentParser(description="Train CNN model for sentiment analysis")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num-filters", type=int, default=100, help="Number of filters")
    parser.add_argument("--filter-sizes", nargs="+", type=int, default=[2, 3, 4, 5], 
                       help="Filter sizes for CNN")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--max-vocab-size", type=int, default=10000, help="Maximum vocabulary size")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints", help="Checkpoint directory")
    
    args = parser.parse_args()
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up data pipeline
    logger.info("Setting up data pipeline...")
    preprocessor = create_default_preprocessor()
    
    # Create vocabulary
    vocabulary = create_vocabulary_from_data(
        "data/processed/imdb_train.csv",
        "data/processed/imdb_test.csv",
        text_column="review",
        max_vocab_size=args.max_vocab_size,
        min_frequency=2,
        preprocessor=preprocessor,
    )
    
    # Create tokenizer
    tokenizer = create_tokenizer(
        vocabulary=vocabulary,
        max_length=args.max_length
    )
    
    # Load datasets
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader, _ = load_sentiment_data(
        data_path="data/processed/imdb_train.csv",
        vocabulary=vocabulary,
        preprocessor=preprocessor,
        text_column="review",
        label_column="sentiment",
        max_length=args.max_length,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
        batch_size=args.batch_size,
        random_seed=42
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Create CNN model
    logger.info("Creating CNN model...")
    config = CNNConfig(
        vocab_size=vocabulary.vocab_size,
        embed_dim=args.embed_dim,
        num_filters=args.num_filters,
        filter_sizes=args.filter_sizes,
        output_dim=2,
        dropout=args.dropout,
    )
    model = CNNModel(config)
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Set up training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = create_scheduler(optimizer, name="step", step_size=5, gamma=0.5)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, mode="max")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        progress_bar=True
    )
    
    # Train the model
    logger.info("Starting training...")
    history = trainer.fit(
        epochs=args.epochs,
        early_stopper=early_stopping
    )
    
    # Save final model
    checkpoint_path = checkpoint_dir / "cnn_final.pt"
    trainer.save_checkpoint(checkpoint_path, epoch=args.epochs, metrics=history)
    logger.info(f"Training completed! Final model saved to {checkpoint_path}")
    
    # Save training history
    history_path = checkpoint_dir / "cnn_final_history.yaml"
    with open(history_path, "w") as f:
        yaml.dump(history, f)
    logger.info(f"Training history saved to {history_path}")
    
    # Print final results
    final_train_acc = history["train_accuracy"][-1]
    final_val_acc = history["val_accuracy"][-1]
    logger.info(f"Final Training Accuracy: {final_train_acc:.4f}")
    logger.info(f"Final Validation Accuracy: {final_val_acc:.4f}")


if __name__ == "__main__":
    main() 
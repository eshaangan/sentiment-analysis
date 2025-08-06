#!/usr/bin/env python3
"""Train a better sentiment analysis model."""

import argparse
import logging
import yaml
from pathlib import Path

import torch

from src.data.preprocessing import create_default_preprocessor
from src.data.vocabulary import create_vocabulary_from_data
from src.data.tokenization import create_tokenizer
from src.data.dataset import load_sentiment_data
from src.models.lstm_model import LSTMConfig, LSTMModel
from src.training.trainer import Trainer
from src.training.early_stopping import EarlyStopping
from src.training.utils import get_device

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train a better sentiment analysis model")
    parser.add_argument("--model-type", default="lstm", choices=["lstm", "cnn"], help="Model type")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Create preprocessor and vocabulary
    logger.info("Setting up data pipeline...")
    preprocessor = create_default_preprocessor()
    
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
        max_length=256  # Shorter sequences for faster training
    )

    # Load data
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader, _ = load_sentiment_data(
        data_path="data/processed/imdb_train.csv",
        vocabulary=vocabulary,
        preprocessor=preprocessor,
        text_column="review",
        label_column="sentiment",
        max_length=256,
        batch_size=args.batch_size,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
        num_workers=0,  # Use 0 for MPS compatibility
        random_seed=args.seed,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    # Create model
    logger.info(f"Creating {args.model_type} model...")
    if args.model_type == "lstm":
        config = LSTMConfig(
            vocab_size=vocabulary.vocab_size,
            embed_dim=128,
            hidden_dim=256,
            output_dim=2,
            bidirectional=True,
            pooling="mean",  # Use mean pooling for better performance
            dropout=0.3,
        )
        model = LSTMModel(config)
    else:
        from src.models.cnn_model import CNNConfig, CNNModel
        config = CNNConfig(
            vocab_size=vocabulary.vocab_size,
            embed_dim=128,
            num_filters=100,
            filter_sizes=[3, 4, 5],
            dropout=0.5,
        )
        model = CNNModel(config)

    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Create optimizer and loss function
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Create early stopping
    early_stopper = EarlyStopping(
        patience=5,
        min_delta=0.001,
        mode="max",  # Monitor validation accuracy
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        grad_clip=1.0,
        progress_bar=True,
    )

    # Create checkpoint directory
    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train model
    logger.info("Starting training...")
    trainer.fit(epochs=args.epochs, early_stopper=early_stopper)

    # Save final model
    final_path = checkpoint_dir / f"{args.model_type}_better.pt"
    trainer.save_checkpoint(
        final_path,
        epoch=len(trainer.history),
        metrics=trainer.history[-1] if trainer.history else {}
    )

    # Save training history
    history_path = checkpoint_dir / f"{args.model_type}_better_history.yaml"
    with open(history_path, "w") as f:
        yaml.dump(trainer.history, f)

    logger.info(f"Training completed! Final model saved to {final_path}")
    logger.info(f"Training history saved to {history_path}")

    # Print final metrics
    if trainer.history:
        final_metrics = trainer.history[-1]
        logger.info(f"Final Training Accuracy: {final_metrics['train_acc']:.4f}")
        logger.info(f"Final Validation Accuracy: {final_metrics['val_acc']:.4f}")


if __name__ == "__main__":
    main() 
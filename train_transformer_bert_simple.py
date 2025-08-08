#!/usr/bin/env python3
"""Train Transformer model with simplified BERT-like embeddings for sentiment analysis."""

import argparse
import logging
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.preprocessing import create_default_preprocessor
from src.data.vocabulary import create_vocabulary_from_data
from src.data.tokenization import create_tokenizer
from src.data.dataset import SentimentDataset
from src.models.transformer_model import TransformerConfig, TransformerModel
from src.training.trainer import Trainer
from src.training.utils import get_device

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_bert_embeddings(vocab_size: int, embed_dim: int, max_length: int = 512):
    """Create simplified BERT-like embeddings without slow positional encoding."""
    # Initialize embeddings with better starting values
    embeddings = torch.randn(vocab_size, embed_dim) * 0.02  # Small random initialization
    
    # Create simple positional embeddings (much faster)
    position_embeddings = torch.zeros(max_length, embed_dim)
    for pos in range(max_length):
        for i in range(0, embed_dim, 2):
            if i + 1 < embed_dim:
                # Simplified positional encoding
                position_embeddings[pos, i] = torch.sin(torch.tensor(pos / (10000 ** (i / embed_dim))))
                position_embeddings[pos, i + 1] = torch.cos(torch.tensor(pos / (10000 ** (i / embed_dim))))
    
    return embeddings, position_embeddings


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Transformer model with simplified BERT-like embeddings")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Feed-forward hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max-vocab-size", type=int, default=10000, help="Maximum vocabulary size")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing factor")

    args = parser.parse_args()

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
    train_dataset = SentimentDataset(
        data_path="data/processed/imdb_train.csv",
        tokenizer=tokenizer,
        text_column="review",
        label_column="sentiment",
        max_length=args.max_length,
    )
    
    val_dataset = SentimentDataset(
        data_path="data/processed/imdb_test.csv",
        tokenizer=tokenizer,
        text_column="review",
        label_column="sentiment",
        max_length=args.max_length,
    )

    # Create data loaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        pin_memory=True
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    # Create Transformer model
    logger.info("Creating Transformer model with simplified BERT-like embeddings...")
    config = TransformerConfig(
        vocab_size=vocabulary.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        max_seq_length=args.max_length,
        output_dim=2,
        dropout=args.dropout,
    )
    model = TransformerModel(config)
    
    # Initialize with simplified BERT-like embeddings
    logger.info("Initializing with simplified BERT-like embeddings...")
    embeddings, position_embeddings = create_simple_bert_embeddings(
        vocab_size=vocabulary.vocab_size,
        embed_dim=args.embed_dim,
        max_length=args.max_length
    )
    
    # Set the embeddings
    with torch.no_grad():
        model.embedding.weight.copy_(embeddings)
        if hasattr(model, 'position_embedding'):
            model.position_embedding.weight.copy_(position_embeddings[:args.max_length])
    
    model.to(device)
    logger.info("Simplified BERT-like embeddings initialized")

    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Cosine annealing scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6
    )

    # Create loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
    )

    # Create checkpoint directory
    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train model
    logger.info("Starting simplified BERT-like Transformer training...")
    history = trainer.fit(epochs=args.epochs)

    # Save final model
    checkpoint_path = checkpoint_dir / "transformer_bert_simple.pt"
    trainer.save_checkpoint(checkpoint_path, epoch=args.epochs, metrics=history)
    logger.info(f"Training completed! Final model saved to {checkpoint_path}")

    # Save training history
    history_path = checkpoint_dir / "transformer_bert_simple_history.yaml"
    with open(history_path, "w") as f:
        yaml.dump(history, f)
    logger.info(f"Training history saved to {history_path}")

    # Print final results
    if history and "train_accuracy" in history and history["train_accuracy"]:
        final_train_acc = history["train_accuracy"][-1]
        final_val_acc = history["val_accuracy"][-1]
        logger.info(f"Final Training Accuracy: {final_train_acc:.4f}")
        logger.info(f"Final Validation Accuracy: {final_val_acc:.4f}")
    else:
        logger.info("Training completed successfully!")


if __name__ == "__main__":
    main() 
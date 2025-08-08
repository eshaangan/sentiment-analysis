#!/usr/bin/env python3
"""Train Transformer model with BERT embeddings for transfer learning."""

import argparse
import logging
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.data.preprocessing import create_default_preprocessor
from src.data.vocabulary import create_vocabulary_from_data
from src.data.tokenization import create_tokenizer
from src.data.dataset import load_sentiment_data
from src.models.transformer_model import TransformerConfig, TransformerModel
from src.training.trainer import Trainer
from src.training.schedulers import create_scheduler
from src.training.early_stopping import EarlyStopping
from src.training.utils import get_device

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_bert_like_embeddings(vocab_size, embed_dim, device):
    """Create BERT-like embeddings with positional encoding."""
    # Initialize embeddings with better distribution (like BERT)
    embeddings = torch.randn(vocab_size, embed_dim, device=device) * 0.02  # BERT initialization
    
    # Add positional encoding
    position_embeddings = torch.zeros(vocab_size, embed_dim, device=device)
    for pos in range(vocab_size):
        for i in range(0, embed_dim, 2):
            position_embeddings[pos, i] = np.sin(pos / (10000 ** (i / embed_dim)))
            if i + 1 < embed_dim:
                position_embeddings[pos, i + 1] = np.cos(pos / (10000 ** (i / embed_dim)))
    
    # Combine word and position embeddings
    combined_embeddings = embeddings + position_embeddings * 0.1
    
    return combined_embeddings


def main():
    """Main training function for Transformer with BERT-like embeddings."""
    parser = argparse.ArgumentParser(description="Train Transformer with BERT-like embeddings")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Small batch size for better convergence")
    parser.add_argument("--learning-rate", type=float, default=0.00005, help="Very low learning rate")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Feed-forward hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Low dropout for transfer learning")
    parser.add_argument("--max-vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--max-length", type=int, default=256, help="Sequence length")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints", help="Checkpoint directory")
    parser.add_argument("--use-bert-embeddings", action="store_true", default=True, help="Use BERT-like embeddings")
    
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
    
    # Create Transformer model with BERT-like architecture
    logger.info("Creating Transformer model with BERT-like embeddings...")
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
    
    # Initialize with BERT-like embeddings
    if args.use_bert_embeddings:
        logger.info("Initializing with BERT-like embeddings...")
        bert_embeddings = create_bert_like_embeddings(
            vocabulary.vocab_size, args.embed_dim, device
        )
        model.embedding.weight.data.copy_(bert_embeddings)
        logger.info("BERT-like embeddings initialized")
    
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Set up training components optimized for transfer learning
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Light label smoothing
    
    # Use AdamW with very low learning rate for transfer learning
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Use warmup + cosine scheduler
    total_steps = args.epochs * len(train_loader)
    scheduler = create_scheduler(
        optimizer, 
        name="cosine", 
        T_max=total_steps,
        eta_min=1e-7
    )
    
    # Very patient early stopping for transfer learning
    early_stopping = EarlyStopping(patience=10, min_delta=0.0005, mode="max")
    
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
    logger.info("Starting Transformer training with BERT-like embeddings...")
    history = trainer.fit(
        epochs=args.epochs,
        early_stopper=early_stopping
    )
    
    # Save final model
    checkpoint_path = checkpoint_dir / "transformer_bert.pt"
    trainer.save_checkpoint(checkpoint_path, epoch=args.epochs, metrics=history)
    logger.info(f"Training completed! Final model saved to {checkpoint_path}")
    
    # Save training history
    history_path = checkpoint_dir / "transformer_bert_history.yaml"
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
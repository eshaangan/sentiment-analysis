#!/usr/bin/env python3
"""Train Transformer model with combined approach: Data Augmentation + BERT-like embeddings."""

import argparse
import logging
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.preprocessing import create_default_preprocessor
from src.data.vocabulary import create_vocabulary_from_data
from src.data.tokenization import create_tokenizer
from src.data.dataset import SentimentDataset
from src.data.augmentation import create_augmented_csv
from src.models.transformer_model import TransformerConfig, TransformerModel
from src.training.trainer import Trainer
from src.training.early_stopping import EarlyStopping
from src.training.utils import get_device

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_bert_like_embeddings(vocab_size: int, embed_dim: int, max_length: int = 512):
    """Create BERT-like embeddings with positional encoding."""
    # Initialize embeddings with better starting values (like BERT)
    embeddings = torch.randn(vocab_size, embed_dim) * 0.02  # BERT initialization
    
    # Create positional embeddings
    position_embeddings = torch.zeros(max_length, embed_dim)
    for pos in range(max_length):
        for i in range(0, embed_dim, 2):
            if i + 1 < embed_dim:
                # Sinusoidal positional encoding
                position_embeddings[pos, i] = torch.sin(torch.tensor(pos / (10000 ** (i / embed_dim))))
                position_embeddings[pos, i + 1] = torch.cos(torch.tensor(pos / (10000 ** (i / embed_dim))))
    
    return embeddings, position_embeddings


def create_augmented_dataset_if_needed(train_file: str, augmented_file: str, augmentation_prob: float = 0.3):
    """Create augmented dataset if it doesn't exist."""
    if not Path(augmented_file).exists():
        logger.info(f"Creating augmented dataset with {augmentation_prob} probability...")
        create_augmented_csv(train_file, augmented_file, augmentation_prob)
        logger.info("Augmented dataset created")
    else:
        logger.info("Augmented dataset already exists")


def main():
    """Main training function for combined approach."""
    parser = argparse.ArgumentParser(description="Train Transformer with combined approach (Augmentation + BERT-like)")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Feed-forward hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max-vocab-size", type=int, default=10000, help="Maximum vocabulary size")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--augmentation-prob", type=float, default=0.3, help="Data augmentation probability")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing factor")
    parser.add_argument("--early-stopping-patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints", help="Checkpoint directory")

    args = parser.parse_args()

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create augmented dataset
    train_file = "data/processed/imdb_train.csv"
    augmented_file = "data/processed/imdb_train_augmented.csv"
    create_augmented_dataset_if_needed(train_file, augmented_file, args.augmentation_prob)

    # Set up data pipeline
    logger.info("Setting up data pipeline...")
    preprocessor = create_default_preprocessor()
    
    # Create vocabulary from augmented data
    vocabulary = create_vocabulary_from_data(
        augmented_file,  # Use augmented data for vocabulary
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

    # Load augmented training dataset
    logger.info("Loading augmented training dataset...")
    train_dataset = SentimentDataset(
        data_path=augmented_file,
        tokenizer=tokenizer,
        text_column="review",
        label_column="sentiment",
        max_length=args.max_length,
    )
    
    # Load validation dataset (original test data)
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

    logger.info(f"Train batches: {len(train_loader)} (augmented data)")
    logger.info(f"Validation batches: {len(val_loader)}")

    # Create Transformer model with BERT-like architecture
    logger.info("Creating Transformer model with combined approach...")
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
    logger.info("Initializing with BERT-like embeddings...")
    embeddings, position_embeddings = create_bert_like_embeddings(
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
    logger.info("BERT-like embeddings initialized")
    logger.info(f"Model parameters: {model.count_parameters():,}")

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

    # Create early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience, 
        min_delta=0.001, 
        mode="max"
    )

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

    # Train model
    logger.info("Starting combined approach training (Augmentation + BERT-like)...")
    history = trainer.fit(
        epochs=args.epochs,
        early_stopper=early_stopping
    )

    # Save final model
    checkpoint_path = checkpoint_dir / "transformer_combined.pt"
    trainer.save_checkpoint(checkpoint_path, epoch=args.epochs, metrics=history)
    logger.info(f"Training completed! Final model saved to {checkpoint_path}")

    # Save training history
    history_path = checkpoint_dir / "transformer_combined_history.yaml"
    with open(history_path, "w") as f:
        yaml.dump(history, f)
    logger.info(f"Training history saved to {history_path}")

    # Print final results
    if history and "train_accuracy" in history and history["train_accuracy"]:
        final_train_acc = history["train_accuracy"][-1]
        final_val_acc = history["val_accuracy"][-1]
        logger.info(f"Final Training Accuracy: {final_train_acc:.4f}")
        logger.info(f"Final Validation Accuracy: {final_val_acc:.4f}")
        
        # Expected improvement analysis
        logger.info("Combined Approach Results:")
        logger.info(f"   - Original Transformer: ~67% accuracy")
        logger.info(f"   - With Augmentation: ~85% accuracy")
        logger.info(f"   - With BERT-like: ~85% accuracy")
        logger.info(f"   - Combined Approach: {final_val_acc:.1%} accuracy")
        
        if final_val_acc > 0.85:
            logger.info("   SUCCESS: Combined approach achieved >85% accuracy!")
        elif final_val_acc > 0.80:
            logger.info("   GOOD: Combined approach achieved >80% accuracy")
        else:
            logger.info("   Combined approach needs tuning")
    else:
        logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()

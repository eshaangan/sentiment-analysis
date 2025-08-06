#!/usr/bin/env python3
"""
Training script for sentiment analysis models.
This script handles the complete training pipeline from data loading to model training.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import load_sentiment_data
from src.data.preprocessing import create_default_preprocessor
from src.data.tokenization import create_tokenizer
from src.data.vocabulary import create_vocabulary_from_data
from src.models.factory import create_model_from_yaml
from src.training.early_stopping import EarlyStopping
from src.training.schedulers import create_scheduler
from src.training.trainer import Trainer
from src.training.utils import get_device, set_seed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument(
        "--model-config",
        default="config/model_config.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--training-config",
        default="config/training_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--model-type",
        default="lstm",
        choices=["lstm", "cnn", "transformer"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load configurations
    logger.info("Loading configurations...")
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Check if data exists
    train_path = Path(training_config["dataset"]["train_path"])
    test_path = Path(training_config["dataset"]["test_path"])

    if not train_path.exists() or not test_path.exists():
        logger.error(f"Data files not found. Please run data download first:")
        logger.error(f"python src/data/download_data.py")
        return

    # Create preprocessor
    logger.info("Creating text preprocessor...")
    preprocessor = create_default_preprocessor()

    # Build vocabulary
    logger.info("Building vocabulary...")
    vocabulary = create_vocabulary_from_data(
        train_path,
        test_path,
        text_column=training_config["dataset"]["text_column"],
        max_vocab_size=training_config["dataset"]["max_vocab_size"],
        min_frequency=training_config["dataset"]["min_frequency"],
        preprocessor=preprocessor,
    )

    logger.info(f"Vocabulary size: {vocabulary.vocab_size}")

    # Create tokenizer
    tokenizer = create_tokenizer(
        vocabulary=vocabulary, max_length=training_config["dataset"]["max_length"]
    )

    # Load data
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader, _ = load_sentiment_data(
        data_path=train_path,
        vocabulary=vocabulary,
        preprocessor=preprocessor,
        text_column=training_config["dataset"]["text_column"],
        label_column=training_config["dataset"]["label_column"],
        max_length=training_config["dataset"]["max_length"],
        batch_size=training_config["training"]["batch_size"],
        train_ratio=training_config["dataset"]["train_split"],
        val_ratio=training_config["dataset"]["val_split"],
        test_ratio=1.0 - training_config["dataset"]["train_split"] - training_config["dataset"]["val_split"],
        num_workers=training_config["hardware"]["num_workers"],
        random_seed=args.seed,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # Update model config with actual vocab size
    model_config[args.model_type]["vocab_size"] = vocabulary.vocab_size
    model_config[args.model_type]["num_classes"] = training_config.get("num_classes", 2)

    # Create model
    logger.info(f"Creating {args.model_type} model...")
    if args.model_type == "lstm":
        from src.models.lstm_model import LSTMConfig, LSTMModel
        config = LSTMConfig(**model_config[args.model_type])
        model = LSTMModel(config)
    elif args.model_type == "cnn":
        from src.models.cnn_model import CNNConfig, CNNModel
        config = CNNConfig(**model_config[args.model_type])
        model = CNNModel(config)
    elif args.model_type == "transformer":
        from src.models.transformer_model import TransformerConfig, TransformerModel
        config = TransformerConfig(**model_config[args.model_type])
        model = TransformerModel(config)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Create optimizer
    if training_config["training"]["optimizer"].lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config["training"]["learning_rate"],
            weight_decay=training_config["training"]["weight_decay"],
        )
    elif training_config["training"]["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config["training"]["learning_rate"],
            weight_decay=training_config["training"]["weight_decay"],
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=training_config["training"]["learning_rate"],
            weight_decay=training_config["training"]["weight_decay"],
        )

    # Create loss function
    if training_config["training"]["loss_function"] == "label_smoothing":
        from src.models.regularization import LabelSmoothingCrossEntropy

        criterion = LabelSmoothingCrossEntropy(
            smoothing=training_config["training"]["label_smoothing"]
        )
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Create scheduler
    scheduler = None
    if "scheduler" in training_config["training"]:
        scheduler = create_scheduler(
            optimizer,
            name=training_config["training"]["scheduler"]["type"],
            **training_config["training"]["scheduler"],
        )

    # Create early stopping
    early_stopper = None
    if training_config["early_stopping"]["enabled"]:
        early_stopper = EarlyStopping(
            patience=training_config["early_stopping"]["patience"],
            min_delta=training_config["early_stopping"]["min_delta"],
            mode=training_config["early_stopping"]["mode"],
        )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        grad_clip=model_config["regularization"].get("gradient_clip"),
        progress_bar=True,
    )

    # Create checkpoint directory
    checkpoint_dir = Path(training_config["checkpointing"]["save_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train model
    logger.info("Starting training...")
    trainer.fit(
        epochs=training_config["training"]["epochs"], early_stopper=early_stopper
    )

    # Save final model
    final_checkpoint_path = checkpoint_dir / f"{args.model_type}_final.pt"
    trainer.save_checkpoint(
        final_checkpoint_path,
        epoch=len(trainer.history),
        metrics=trainer.history[-1] if trainer.history else {},
    )

    logger.info(f"Training completed! Final model saved to {final_checkpoint_path}")

    # Save training history
    history_path = checkpoint_dir / f"{args.model_type}_history.yaml"
    with open(history_path, "w") as f:
        yaml.dump(trainer.history, f)

    logger.info(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()

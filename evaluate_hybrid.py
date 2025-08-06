#!/usr/bin/env python3
"""Evaluate hybrid CNN+LSTM model for sentiment analysis."""

import argparse
import logging
import yaml
from pathlib import Path

import torch
import numpy as np

from src.data.preprocessing import create_default_preprocessor
from src.data.vocabulary import create_vocabulary_from_data
from src.data.tokenization import create_tokenizer
from src.data.dataset import SentimentDataset
from src.models.hybrid_model import HybridConfig, HybridModel
from src.evaluation.metrics import compute_classification_metrics, get_confusion_matrix, get_classification_report
from src.training.utils import get_device

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model, dataset, device, batch_size=32):
    """Evaluate model on test data and return predictions and labels."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    # Create a simple dataloader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
            else:
                input_ids, labels = batch[0].to(device), batch[1].to(device)

            # Forward pass
            logits = model(input_ids)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return all_predictions, all_labels, all_probabilities


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate hybrid CNN+LSTM sentiment analysis model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--test-data", default="data/processed/imdb_test.csv", help="Path to test data")
    parser.add_argument("--output-dir", default="results", help="Directory to save evaluation results")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum number of samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--max-vocab-size", type=int, default=10000, help="Maximum vocabulary size")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num-filters", type=int, default=64, help="Number of CNN filters")
    parser.add_argument("--filter-sizes", nargs="+", type=int, default=[3, 4, 5], help="Filter sizes")
    parser.add_argument("--lstm-hidden-dim", type=int, default=128, help="LSTM hidden dimension")
    parser.add_argument("--lstm-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--pooling", type=str, default="attention", help="Pooling method")

    args = parser.parse_args()

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check if test data exists
    test_path = Path(args.test_data)
    if not test_path.exists():
        logger.error(f"Test data not found: {test_path}")
        return

    # Create preprocessor and vocabulary
    logger.info("Setting up data pipeline...")
    preprocessor = create_default_preprocessor()
    
    # Use the same vocabulary settings as training
    vocabulary = create_vocabulary_from_data(
        "data/processed/imdb_train.csv",
        str(test_path),
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

    # Load test dataset directly
    logger.info("Loading test dataset...")
    test_dataset = SentimentDataset(
        data_path=test_path,
        tokenizer=tokenizer,
        text_column="review",
        label_column="sentiment",
        max_length=args.max_length,
    )

    # Limit the number of samples for faster evaluation
    if len(test_dataset) > args.max_samples:
        logger.info(f"Limiting evaluation to {args.max_samples} samples")
        test_dataset = torch.utils.data.Subset(test_dataset, range(args.max_samples))

    # Load hybrid model with correct configuration
    logger.info("Loading hybrid CNN+LSTM model...")
    config = HybridConfig(
        vocab_size=vocabulary.vocab_size,
        embed_dim=args.embed_dim,
        num_filters=args.num_filters,
        filter_sizes=args.filter_sizes,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
        lstm_dropout=0.2,
        cnn_dropout=0.3,
        output_dim=2,
        bidirectional=True,
        pooling=args.pooling,
    )
    model = HybridModel(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    # Evaluate model
    logger.info("Evaluating model...")
    predictions, labels, probabilities = evaluate_model(model, test_dataset, device, args.batch_size)

    # Calculate metrics
    metrics = compute_classification_metrics(labels, predictions, average="weighted")
    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]

    logger.info(f"Test Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")

    # Create confusion matrix
    cm = get_confusion_matrix(labels, predictions)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Generate classification report
    class_names = ["negative", "positive"]
    report = get_classification_report(labels, predictions, target_names=class_names)
    logger.info(f"Classification Report:\n{report}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "num_samples": len(labels),
        "model_type": "Hybrid_CNN_LSTM",
        "model_parameters": model.count_parameters(),
    }

    results_path = output_dir / "hybrid_evaluation_results.yaml"
    with open(results_path, "w") as f:
        yaml.dump(results, f)

    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main() 
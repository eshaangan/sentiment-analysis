#!/usr/bin/env python3
"""
Evaluation script for trained sentiment analysis models.
This script loads a trained model and evaluates it on test data.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import load_sentiment_data
from src.data.preprocessing import create_default_preprocessor
from src.data.tokenization import create_tokenizer
from src.data.vocabulary import create_vocabulary_from_data
from src.evaluation.metrics import (calculate_accuracy, calculate_f1,
                                    calculate_precision, calculate_recall,
                                    calculate_roc_auc, create_confusion_matrix,
                                    generate_classification_report,
                                    plot_roc_curve)
from src.training.utils import get_device

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_model(model, test_loader, device):
    """Evaluate model on test data and return predictions and labels."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in test_loader:
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
    parser = argparse.ArgumentParser(description="Evaluate sentiment analysis model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--training-config",
        default="config/training_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--output-dir", default="results", help="Directory to save evaluation results"
    )

    args = parser.parse_args()

    # Load configuration
    logger.info("Loading configuration...")
    training_config = load_config(args.training_config)

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

    # Recreate data pipeline (needed for evaluation)
    train_path = Path(training_config["dataset"]["train_path"])
    test_path = Path(training_config["dataset"]["test_path"])

    if not train_path.exists() or not test_path.exists():
        logger.error("Data files not found. Please ensure data is downloaded.")
        return

    # Create preprocessor and vocabulary
    logger.info("Recreating data pipeline...")
    preprocessor = create_default_preprocessor()
    vocabulary = create_vocabulary_from_data(
        train_path,
        test_path,
        text_column=training_config["dataset"]["text_column"],
        max_vocab_size=training_config["dataset"]["max_vocab_size"],
        min_frequency=training_config["dataset"]["min_frequency"],
        preprocessor=preprocessor,
    )

    # Create tokenizer
    tokenizer = create_tokenizer(
        vocabulary=vocabulary, max_length=training_config["dataset"]["max_length"]
    )

    # Load test data
    _, _, test_loader = load_sentiment_data(
        train_csv_path=train_path,
        test_csv_path=test_path,
        tokenizer=tokenizer,
        batch_size=training_config["training"]["val_batch_size"],
        num_workers=training_config["hardware"]["num_workers"],
    )

    # Load model (this would need to be adapted based on the actual model class)
    # For now, this is a placeholder - you'd need to determine the model type from checkpoint
    logger.info("Loading model...")
    # model = load_model_from_checkpoint(checkpoint_path)  # This needs implementation
    logger.warning("Model loading not implemented - this is a template")
    return

    # Evaluate model
    logger.info("Evaluating model...")
    predictions, labels, probabilities = evaluate_model(model, test_loader, device)

    # Calculate metrics
    accuracy = calculate_accuracy(labels, predictions)
    precision = calculate_precision(labels, predictions, average="weighted")
    recall = calculate_recall(labels, predictions, average="weighted")
    f1 = calculate_f1(labels, predictions, average="weighted")

    logger.info(f"Test Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")

    # Create confusion matrix
    cm = create_confusion_matrix(labels, predictions)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Generate classification report
    class_names = ["negative", "positive"]
    report = generate_classification_report(labels, predictions, class_names)
    logger.info(f"Classification Report:\n{report}")

    # Calculate AUC (for binary classification)
    if len(set(labels)) == 2:
        # Get probabilities for positive class
        pos_probs = [prob[1] for prob in probabilities]
        auc = calculate_roc_auc(labels, pos_probs)
        logger.info(f"AUC: {auc:.4f}")

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
    }

    if len(set(labels)) == 2:
        results["auc"] = float(auc)

    results_path = output_dir / "evaluation_results.yaml"
    with open(results_path, "w") as f:
        yaml.dump(results, f)

    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()

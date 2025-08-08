#!/usr/bin/env python3
"""
Demonstration script for LSTM-based sentiment analysis models.
Shows various LSTM configurations, attention mechanisms, and pooling strategies.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.models.lstm_model import (AttentionLSTM, BidirectionalLSTM,
                                   LSTMConfig, LSTMModel,
                                   create_attention_lstm,
                                   create_bidirectional_lstm,
                                   create_lstm_model)


def demo_lstm_configurations():
    """Demonstrate different LSTM configurations."""
    print("=" * 80)
    print("LSTM MODEL CONFIGURATIONS DEMONSTRATION")
    print("=" * 80)

    print("Creating different LSTM configurations...")

    # Basic LSTM configuration
    basic_config = LSTMConfig(
        vocab_size=5000,
        embed_dim=128,
        hidden_dim=256,
        output_dim=2,
        num_layers=1,
        dropout=0.3,
        bidirectional=False,
        pooling="last",
    )

    # Bidirectional LSTM configuration
    bi_config = LSTMConfig(
        vocab_size=5000,
        embed_dim=128,
        hidden_dim=256,
        output_dim=2,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        pooling="mean",
    )

    # Attention LSTM configuration
    attention_config = LSTMConfig(
        vocab_size=5000,
        embed_dim=128,
        hidden_dim=256,
        output_dim=2,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        attention=True,
        attention_heads=8,
        pooling="attention",
    )

    configs = {
        "Basic LSTM": basic_config,
        "Bidirectional LSTM": bi_config,
        "Attention LSTM": attention_config,
    }

    for name, config in configs.items():
        print(f"\n{name} Configuration:")
        print(f"   Vocabulary size: {config.vocab_size:,}")
        print(f"   Embedding dimension: {config.embed_dim}")
        print(f"   Hidden dimension: {config.hidden_dim}")
        print(f"   Number of layers: {config.num_layers}")
        print(f"   Bidirectional: {config.bidirectional}")
        print(f"   Attention: {config.attention}")
        print(f"   Pooling strategy: {config.pooling}")
        if config.attention:
            print(f"   Attention heads: {config.attention_heads}")

    return configs


def demo_lstm_model_creation(configs):
    """Demonstrate LSTM model creation and properties."""
    print(f"\nLSTM MODEL CREATION AND PROPERTIES")
    print("-" * 60)

    models = {}

    for name, config in configs.items():
        print(f"\n🔨 Creating {name}...")
        model = LSTMModel(config)
        models[name] = model

        # Get parameter summary
        param_summary = model.get_parameter_summary()

        print(f"{name} created:")
        print(f"   Total parameters: {param_summary['total_parameters']:,}")
        print(f"   Trainable parameters: {param_summary['trainable_parameters']:,}")
        print(f"   Model size: {param_summary['model_size_mb']:.2f} MB")
        print(f"   Device: {model.device}")

        # Show LSTM-specific properties
        print(f"   LSTM input size: {model.lstm.input_size}")
        print(f"   LSTM hidden size: {model.lstm.hidden_size}")
        print(f"   LSTM layers: {model.lstm.num_layers}")
        print(f"   LSTM bidirectional: {model.lstm.bidirectional}")

        if model.attention is not None:
            print(f"   Attention heads: {model.attention.num_heads}")
            print(f"   Attention dropout: {model.attention.dropout.p}")

    return models


def demo_forward_pass_and_pooling(models):
    """Demonstrate forward pass with different pooling strategies."""
    print(f"\nFORWARD PASS AND POOLING STRATEGIES")
    print("-" * 60)

    # Create sample input data
    batch_size = 3
    seq_length = 15

    print(f"Creating sample input:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_length}")

    for name, model in configs.items():
        print(f"\nTesting {name}:")

        # Create input for this model's vocabulary
        input_ids = torch.randint(1, model.config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        # Forward pass
        model.eval()
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probabilities = model.predict(input_ids, attention_mask)

        print(f"   Input shape: {input_ids.shape}")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Probabilities shape: {probabilities.shape}")
        print(f"   Pooling strategy: {model.config.pooling}")
        print(f"   Sample logits: {logits[0].cpu().numpy()}")
        print(f"   Sample probabilities: {probabilities[0].cpu().numpy()}")

        # Test single prediction
        single_input = input_ids[0]  # Take first sample
        pred_class, confidence = model.predict_single(single_input)
        print(f"   Single prediction: class={pred_class}, confidence={confidence:.4f}")


def demo_attention_mechanism(models):
    """Demonstrate attention mechanism functionality."""
    print(f"\nATTENTION MECHANISM DEMONSTRATION")
    print("-" * 60)

    # Find attention model
    attention_model = None
    for name, model in models.items():
        if model.attention is not None:
            attention_model = model
            break

    if attention_model is None:
        print("No attention model found in the created models.")
        return

    print(f"Working with {name} attention model:")
    print(f"   Attention heads: {attention_model.attention.num_heads}")
    print(f"   Attention dropout: {attention_model.attention.dropout.p}")

    # Create sample input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(
        1, attention_model.config.vocab_size, (batch_size, seq_len)
    )
    attention_mask = torch.ones(batch_size, seq_len)

    print(f"\nSample input:")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Input tokens: {input_ids[0].cpu().numpy()}")

    # Get attention weights
    attention_weights = attention_model.get_attention_weights(input_ids, attention_mask)

    if attention_weights is not None:
        print(f"\nAttention weights:")
        print(f"   Weights shape: {attention_weights.shape}")
        print(
            f"   Weights sum per sequence: {attention_weights.sum(dim=1).cpu().numpy()}"
        )
        print(f"   Sample attention distribution: {attention_weights[0].cpu().numpy()}")

        # Show which tokens get most attention
        for i in range(batch_size):
            weights = attention_weights[i].cpu().numpy()
            max_attention_idx = weights.argmax()
            print(
                f"   Sequence {i}: Token {max_attention_idx} has highest attention ({weights[max_attention_idx]:.4f})"
            )
    else:
        print("Could not retrieve attention weights")


def demo_pooling_strategies():
    """Demonstrate different pooling strategies."""
    print(f"\n🏊 POOLING STRATEGIES COMPARISON")
    print("-" * 60)

    vocab_size, embed_dim, hidden_dim, output_dim = 1000, 64, 128, 2
    batch_size, seq_len = 2, 8

    pooling_strategies = ["last", "mean", "max", "attention"]

    print(f"Testing different pooling strategies:")
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Embedding dimension: {embed_dim}")
    print(f"   Hidden dimension: {hidden_dim}")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")

    for pooling in pooling_strategies:
        print(f"\n{pooling.upper()} Pooling:")

        # Create configuration
        config = LSTMConfig(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            pooling=pooling,
            attention=(pooling == "attention"),
        )

        # Create model
        model = LSTMModel(config)
        model.eval()

        # Create input
        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Forward pass
        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        print(f"   Model parameters: {model.count_parameters():,}")
        print(f"   Output shape: {logits.shape}")
        print(f"   Sample output: {logits[0].cpu().numpy()}")


def demo_factory_functions():
    """Demonstrate factory functions for creating LSTM models."""
    print(f"\n🏭 FACTORY FUNCTIONS DEMONSTRATION")
    print("-" * 60)

    vocab_size = 5000
    embed_dim = 128
    hidden_dim = 256
    output_dim = 2

    print(f"Using factory functions to create LSTM models:")
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Embedding dimension: {embed_dim}")
    print(f"   Hidden dimension: {hidden_dim}")
    print(f"   Output dimension: {output_dim}")

    # Test different factory functions
    factory_configs = [
        (
            "Basic LSTM",
            lambda: create_lstm_model(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                bidirectional=False,
                attention=False,
                pooling="last",
            ),
        ),
        (
            "Bidirectional LSTM",
            lambda: create_bidirectional_lstm(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
            ),
        ),
        (
            "Attention LSTM",
            lambda: create_attention_lstm(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                attention_heads=4,
            ),
        ),
    ]

    for name, factory_func in factory_configs:
        print(f"\nCreating {name}...")
        model = factory_func()

        print(f"{name} created:")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Parameters: {model.count_parameters():,}")
        print(f"   Bidirectional: {model.config.bidirectional}")
        print(f"   Attention: {model.config.attention}")
        print(f"   Pooling: {model.config.pooling}")

        # Test forward pass
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))

        model.eval()
        with torch.no_grad():
            logits = model(input_ids)

        print(f"   Forward pass successful: {logits.shape}")


def demo_model_comparison():
    """Compare different LSTM model variants."""
    print(f"\nMODEL COMPARISON")
    print("-" * 60)

    vocab_size = 5000
    embed_dim = 128
    hidden_dim = 256
    output_dim = 2

    print(f"Comparing LSTM model variants:")

    # Create different model variants
    variants = {
        "Unidirectional LSTM": create_lstm_model(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            bidirectional=False,
            attention=False,
            pooling="last",
        ),
        "Bidirectional LSTM": create_bidirectional_lstm(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        ),
        "Attention LSTM": create_attention_lstm(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        ),
    }

    print(
        f"\n{'Model Type':<25} {'Parameters':<12} {'Size (MB)':<10} {'Bidirectional':<15} {'Attention':<10}"
    )
    print("-" * 80)

    for name, model in variants.items():
        param_count = model.count_parameters()
        size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32

        print(
            f"{name:<25} {param_count:<12,} {size_mb:<10.2f} {str(model.config.bidirectional):<15} {str(model.config.attention):<10}"
        )

    print(f"\nKey Differences:")
    print(f"   • Unidirectional LSTM: Simple, fast, fewer parameters")
    print(f"   • Bidirectional LSTM: Better context understanding, 2x parameters")
    print(f"   • Attention LSTM: Interpretable, focuses on important tokens")


def demo_training_readiness():
    """Demonstrate that models are ready for training."""
    print(f"\n🎓 TRAINING READINESS DEMONSTRATION")
    print("-" * 60)

    # Create a model
    model = create_lstm_model(
        vocab_size=1000, embed_dim=64, hidden_dim=128, output_dim=2, bidirectional=True
    )

    print(f"Model training setup:")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Trainable parameters: {model.count_parameters():,}")

    # Create dummy training data
    batch_size, seq_len = 4, 10
    input_ids = torch.randint(1, 1000, (batch_size, seq_len))
    targets = torch.randint(0, 2, (batch_size,))

    print(f"\nTraining data:")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Target shape: {targets.shape}")

    # Set up training
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining step simulation:")

    # Forward pass
    logits = model(input_ids)
    loss = criterion(logits, targets)

    print(f"   Loss: {loss.item():.6f}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    total_norm = model.apply_gradient_clipping(max_norm=1.0)
    print(f"   Gradient norm: {total_norm:.6f}")

    # Optimizer step
    optimizer.step()
    print(f"   Training step completed successfully")


def main():
    """Run all LSTM model demonstrations."""
    print("🧠 LSTM-BASED SENTIMENT ANALYSIS MODEL DEMONSTRATION")
    print(
        "This script demonstrates the features of LSTM models for sentiment analysis."
    )
    print(
        "    Including various configurations, attention mechanisms, and pooling strategies."
    )
    print()

    try:
        # Configuration demonstration
        configs = demo_lstm_configurations()

        # Model creation and properties
        models = demo_lstm_model_creation(configs)

        # Forward pass and pooling
        demo_forward_pass_and_pooling(models)

        # Attention mechanism
        demo_attention_mechanism(models)

        # Pooling strategies
        demo_pooling_strategies()

        # Factory functions
        demo_factory_functions()

        # Model comparison
        demo_model_comparison()

        # Training readiness
        demo_training_readiness()

        print(f"\n" + "=" * 80)
        print("LSTM MODEL DEMONSTRATION COMPLETE")
        print("=" * 80)
        print()
        print("Key Features Demonstrated:")
        print("• Multiple LSTM configurations (unidirectional, bidirectional)")
        print("• Attention mechanism with multi-head attention")
        print("• Various pooling strategies (last, mean, max, attention)")
        print("• Factory functions for easy model creation")
        print("• Parameter counting and model comparison")
        print("• Training readiness with gradient clipping")
        print("• Attention weight extraction for interpretability")
        print("• Integration with base model functionality")
        print()
        print("Ready for CNN and Transformer model implementation!")

    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

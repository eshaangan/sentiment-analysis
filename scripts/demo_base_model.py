#!/usr/bin/env python3
"""
Demonstration script for base model functionality.
Shows the common features and capabilities of the BaseModel class.
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

from src.models.base_model import BaseModel, ModelConfig


class DemoLSTMModel(BaseModel):
    """Simple LSTM model for demonstration purposes."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

        # Update classifier for bidirectional LSTM
        lstm_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim),
        )

    def forward(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass through LSTM model."""
        # Embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        embedded = self.dropout(embedded)

        # LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)

        # Use last hidden state for classification
        if self.config.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        # Classification
        logits = self.classifier(hidden)
        return logits


def demo_model_configuration():
    """Demonstrate model configuration features."""
    print("=" * 80)
    print("MODEL CONFIGURATION DEMONSTRATION")
    print("=" * 80)

    print("Creating different model configurations...")

    # Create different configurations
    configs = {
        "small": ModelConfig(
            vocab_size=5000,
            embed_dim=64,
            hidden_dim=128,
            output_dim=2,
            num_layers=1,
            dropout=0.2,
            bidirectional=False,
            model_type="lstm",
        ),
        "medium": ModelConfig(
            vocab_size=10000,
            embed_dim=128,
            hidden_dim=256,
            output_dim=2,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            model_type="lstm",
        ),
        "large": ModelConfig(
            vocab_size=20000,
            embed_dim=256,
            hidden_dim=512,
            output_dim=2,
            num_layers=3,
            dropout=0.4,
            bidirectional=True,
            model_type="lstm",
        ),
    }

    for name, config in configs.items():
        print(f"\n{name.upper()} Model Configuration:")
        print(f"   Vocabulary size: {config.vocab_size:,}")
        print(f"   Embedding dimension: {config.embed_dim}")
        print(f"   Hidden dimension: {config.hidden_dim}")
        print(f"   Number of layers: {config.num_layers}")
        print(f"   Dropout: {config.dropout}")
        print(f"   Bidirectional: {config.bidirectional}")
        print(f"   Output classes: {config.output_dim}")

    return configs


def demo_model_creation_and_summary(configs):
    """Demonstrate model creation and summary features."""
    print(f"\nMODEL CREATION AND SUMMARY")
    print("-" * 60)

    models = {}

    for name, config in configs.items():
        print(f"\nCreating {name} model...")
        model = DemoLSTMModel(config)
        models[name] = model

        # Get parameter summary
        param_summary = model.get_parameter_summary()

        print(f"{name.capitalize()} model created:")
        print(f"   Total parameters: {param_summary['total_parameters']:,}")
        print(f"   Trainable parameters: {param_summary['trainable_parameters']:,}")
        print(f"   Model size: {param_summary['model_size_mb']:.2f} MB")
        print(f"   Device: {model.device}")

        # Show layer breakdown for medium model
        if name == "medium":
            print(f"\n   Layer breakdown:")
            for layer_type, info in param_summary["layer_breakdown"].items():
                print(
                    f"      {layer_type}: {info['count']} layers, {info['params']:,} parameters"
                )

    return models


def demo_forward_pass(models):
    """Demonstrate model forward pass and prediction."""
    print(f"\n🔄 FORWARD PASS AND PREDICTION")
    print("-" * 60)

    # Create sample input data
    batch_size = 4
    seq_length = 20

    print(f"Creating sample input:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_length}")

    for name, model in models.items():
        print(f"\nTesting {name} model:")

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
        print(f"   Sample logits: {logits[0].cpu().numpy()}")
        print(f"   Sample probabilities: {probabilities[0].cpu().numpy()}")

        # Test single prediction
        single_input = input_ids[0]  # Take first sample
        pred_class, confidence = model.predict_single(single_input)
        print(f"   Single prediction: class={pred_class}, confidence={confidence:.4f}")


def demo_embedding_operations(models):
    """Demonstrate embedding layer operations."""
    print(f"\n🔤 EMBEDDING OPERATIONS")
    print("-" * 60)

    model = models["medium"]  # Use medium model for demo

    print(f"Working with medium model embeddings:")
    print(f"   Embedding shape: {model.embedding.weight.shape}")

    # Get current embedding weights
    original_weights = model.get_embedding_weights()
    print(f"   Original embedding mean: {original_weights.mean().item():.6f}")
    print(f"   Original embedding std: {original_weights.std().item():.6f}")

    # Test freeze/unfreeze
    print(f"\nTesting freeze/unfreeze:")
    print(f"   Initial requires_grad: {model.embedding.weight.requires_grad}")

    model.freeze_embedding()
    print(f"   After freeze: {model.embedding.weight.requires_grad}")

    model.unfreeze_embedding()
    print(f"   After unfreeze: {model.embedding.weight.requires_grad}")

    # Test setting new weights
    print(f"\n🔄 Testing weight replacement:")
    new_weights = torch.randn_like(original_weights) * 0.1
    model.set_embedding_weights(new_weights)

    updated_weights = model.get_embedding_weights()
    print(f"   New embedding mean: {updated_weights.mean().item():.6f}")
    print(f"   New embedding std: {updated_weights.std().item():.6f}")
    print(
        f"   Weights changed: {not torch.allclose(original_weights, updated_weights)}"
    )


def demo_model_persistence(models):
    """Demonstrate model saving and loading."""
    print(f"\n💾 MODEL PERSISTENCE")
    print("-" * 60)

    model = models["small"]  # Use small model for demo

    # Create temporary save path
    save_dir = Path("models/demo")
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / "demo_model.pt"
    config_path = save_dir / "demo_config.yaml"

    print(f"💾 Saving model and configuration:")
    print(f"   Model path: {model_path}")
    print(f"   Config path: {config_path}")

    # Save model
    model.save_model(model_path, save_config=True)

    # Verify files exist
    if model_path.exists():
        print(f"   Model file saved ({model_path.stat().st_size} bytes)")
    if config_path.exists():
        print(f"   Config file saved ({config_path.stat().st_size} bytes)")

    # Load and verify checkpoint structure
    checkpoint = torch.load(model_path, map_location="cpu")
    print(f"\nCheckpoint contents:")
    for key in checkpoint.keys():
        if key == "model_state_dict":
            print(f"   {key}: {len(checkpoint[key])} state dict entries")
        else:
            print(f"   {key}: {type(checkpoint[key])}")


def demo_device_management(models):
    """Demonstrate device management features."""
    print(f"\nDEVICE MANAGEMENT")
    print("-" * 60)

    model = models["small"]

    print(f"Device information:")
    print(f"   Current device: {model.device}")
    print(f"   CUDA available: {torch.cuda.is_available()}")

    if hasattr(torch.backends, "mps"):
        print(f"   MPS available: {torch.backends.mps.is_available()}")

    # Show where model parameters are located
    sample_param = next(model.parameters())
    print(f"   Model parameters device: {sample_param.device}")
    print(f"   Model parameters dtype: {sample_param.dtype}")

    # Test with different input devices
    print(f"\n🔄 Testing input device handling:")

    # Create input on CPU
    input_ids = torch.randint(1, model.config.vocab_size, (2, 10))
    print(f"   Input device: {input_ids.device}")

    # Model should handle device mismatch
    with torch.no_grad():
        try:
            # Move input to model's device
            input_ids = input_ids.to(model.device)
            output = model(input_ids)
            print(f"   Output device: {output.device}")
            print(f"   Forward pass successful")
        except Exception as e:
            print(f"   Forward pass failed: {e}")


def demo_gradient_operations(models):
    """Demonstrate gradient clipping and training utilities."""
    print(f"\n🎓 GRADIENT OPERATIONS")
    print("-" * 60)

    model = models["small"]
    model.train()

    print(f"Testing gradient operations:")

    # Create dummy training step
    batch_size, seq_len = 3, 15
    input_ids = torch.randint(1, model.config.vocab_size, (batch_size, seq_len))
    target = torch.randint(0, model.config.output_dim, (batch_size,))

    # Forward pass
    logits = model(input_ids)
    loss = nn.CrossEntropyLoss()(logits, target)

    print(f"   Loss: {loss.item():.6f}")

    # Backward pass
    loss.backward()

    # Test gradient clipping
    max_norms = [0.5, 1.0, 5.0]
    for max_norm in max_norms:
        # Re-compute gradients for each test
        model.zero_grad()
        loss.backward()

        total_norm = model.apply_gradient_clipping(max_norm)
        print(f"   Gradient norm (max={max_norm}): {total_norm:.6f}")


def demo_model_summary(models):
    """Demonstrate comprehensive model summary."""
    print(f"\nCOMPREHENSIVE MODEL SUMMARY")
    print("-" * 60)

    for name, model in models.items():
        print(f"\n{'=' * 20} {name.upper()} MODEL {'=' * 20}")
        print(model.get_model_summary())


def main():
    """Run all base model demonstrations."""
    print("BASE MODEL FUNCTIONALITY DEMONSTRATION")
    print("This script demonstrates the features of the BaseModel class")
    print("    and common functionality for all sentiment analysis models.")
    print()

    try:
        # Configuration demonstration
        configs = demo_model_configuration()

        # Model creation and summary
        models = demo_model_creation_and_summary(configs)

        # Forward pass and prediction
        demo_forward_pass(models)

        # Embedding operations
        demo_embedding_operations(models)

        # Model persistence
        demo_model_persistence(models)

        # Device management
        demo_device_management(models)

        # Gradient operations
        demo_gradient_operations(models)

        # Comprehensive summary
        demo_model_summary(models)

        print(f"\n" + "=" * 80)
        print("BASE MODEL DEMONSTRATION COMPLETE")
        print("=" * 80)
        print()
        print("Key Features Demonstrated:")
        print("• Automatic device detection and management")
        print("• Flexible model configuration system")
        print("• Common embedding and classification layers")
        print("• Parameter counting and model summarization")
        print("• Model saving and loading capabilities")
        print("• Embedding weight management")
        print("• Gradient clipping and training utilities")
        print("• Forward pass and prediction methods")
        print("• Integration with PyTorch training ecosystem")
        print()
        print("Ready for specific model architecture implementation!")

    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
Base model class for sentiment analysis with common functionality.
Provides a foundation for all model architectures (LSTM, CNN, Transformer).
"""

import abc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConfig:
    """
    Configuration class for model hyperparameters and settings.
    Handles loading from YAML files and provides validation.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 2,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        model_type: str = "lstm",
        **kwargs,
    ):
        """
        Initialize model configuration.

        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            output_dim: Number of output classes (2 for binary sentiment)
            num_layers: Number of layers in the model
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional layers (for LSTM/GRU)
            model_type: Type of model ("lstm", "cnn", "transformer")
            **kwargs: Additional model-specific parameters
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.model_type = model_type.lower()

        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Validate configuration
        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if self.model_type not in ["lstm", "gru", "cnn", "transformer"]:
            logger.warning(f"Unknown model_type: {self.model_type}")

    @classmethod
    def from_yaml(
        cls, config_path: Union[str, Path], model_type: str = "lstm"
    ) -> "ModelConfig":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
            model_type: Type of model to load config for

        Returns:
            ModelConfig instance
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Get model-specific config
        if model_type in config_data:
            model_config = config_data[model_type]
        else:
            # Try to find in nested structure
            model_config = config_data.get("models", {}).get(model_type, {})

        if not model_config:
            raise ValueError(
                f"No configuration found for model type '{model_type}' in {config_path}"
            )

        # Add model_type to config
        model_config["model_type"] = model_type

        return cls(**model_config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith("_") and not callable(getattr(self, attr))
        }

    def save(self, save_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    def __repr__(self) -> str:
        return f"ModelConfig({self.model_type}: vocab={self.vocab_size}, embed={self.embed_dim}, hidden={self.hidden_dim})"


class BaseModel(nn.Module, abc.ABC):
    """
    Abstract base class for sentiment analysis models.

    Provides common functionality for all model architectures:
    - Device management and automatic GPU detection
    - Common forward pass structure
    - Model saving and loading
    - Parameter counting and model summary
    - Integration with configuration system
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize base model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.device = self._get_device()

        # Common layers that all models will have
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=0,  # Assuming PAD token is at index 0
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)

        # Classification head - will be common across all models
        self.classifier = self._build_classifier()

        # Initialize weights
        self._init_weights()

        # Move to appropriate device
        self.to(self.device)

        logger.info(
            f"Initialized {self.__class__.__name__} with {self.count_parameters():,} parameters"
        )

    def _get_device(self) -> torch.device:
        """Automatically detect and return appropriate device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device

    def _build_classifier(self) -> nn.Module:
        """
        Build the classification head.
        This is common across all model types.
        """
        # The input dimension depends on the model architecture
        # For now, use hidden_dim, but subclasses can override
        classifier_input_dim = self._get_classifier_input_dim()

        return nn.Sequential(
            nn.Linear(classifier_input_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim // 2, self.config.output_dim),
        )

    def _get_classifier_input_dim(self) -> int:
        """
        Get the input dimension for the classifier.
        Should be overridden by subclasses based on their architecture.
        """
        return self.config.hidden_dim

    def _init_weights(self) -> None:
        """Initialize model weights using appropriate strategies."""
        for name, param in self.named_parameters():
            if "weight" in name:
                if len(param.shape) >= 2:  # Linear layers, convolutions
                    nn.init.xavier_uniform_(param)
                else:  # Bias vectors, batch norm, etc.
                    nn.init.zeros_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Special initialization for embedding layer
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        # Set padding token embedding to zero
        if (
            hasattr(self.embedding, "padding_idx")
            and self.embedding.padding_idx is not None
        ):
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

    @abc.abstractmethod
    def forward(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Token indices [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Logits [batch_size, output_dim]
        """
        pass

    def predict(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Make predictions with the model.

        Args:
            input_ids: Token indices [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Predicted class probabilities [batch_size, output_dim]
        """
        self.eval()
        with torch.no_grad():
            input_tensor = input_ids.to(self.device)
            logits = self.forward(input_tensor, attention_mask)
            probabilities = F.softmax(logits, dim=-1)
        return probabilities

    def predict_single(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tuple[int, float]:
        """
        Make a single prediction and return class and confidence.

        Args:
            input_ids: Token indices [1, seq_len] or [seq_len]
            attention_mask: Attention mask [1, seq_len] or [seq_len]

        Returns:
            Tuple of (predicted_class, confidence_score)
        """
        # Ensure batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        probs = self.predict(input_ids, attention_mask)
        predicted_class = probs.argmax(dim=-1).item()
        confidence = probs.max(dim=-1).values.item()

        return predicted_class, confidence

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_summary(self) -> Dict[str, Any]:
        """Get detailed parameter summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Get parameter breakdown by layer type
        layer_breakdown = {}
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module_params = sum(p.numel() for p in module.parameters())
                if module_params > 0:
                    layer_type = module.__class__.__name__
                    if layer_type not in layer_breakdown:
                        layer_breakdown[layer_type] = {"count": 0, "params": 0}
                    layer_breakdown[layer_type]["count"] += 1
                    layer_breakdown[layer_type]["params"] += module_params

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "layer_breakdown": layer_breakdown,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }

    def save_model(self, save_path: Union[str, Path], save_config: bool = True) -> None:
        """
        Save model state dict and configuration.

        Args:
            save_path: Path to save the model
            save_config: Whether to save the model configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "model_class": self.__class__.__name__,
                "config": self.config.to_dict() if save_config else None,
            },
            save_path,
        )

        logger.info(f"Model saved to {save_path}")

        # Save config separately if requested
        if save_config:
            config_path = save_path.parent / f"{save_path.stem}_config.yaml"
            self.config.save(config_path)

    @classmethod
    def load_model(
        cls, model_path: Union[str, Path], config: Optional[ModelConfig] = None
    ) -> "BaseModel":
        """
        Load model from saved state.

        Args:
            model_path: Path to saved model
            config: Model configuration (if not saved with model)

        Returns:
            Loaded model instance
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")

        # Get configuration
        if config is None:
            if checkpoint.get("config") is None:
                raise ValueError(
                    "No configuration provided and none found in checkpoint"
                )
            config = ModelConfig(**checkpoint["config"])

        # Create model instance
        # Note: This is a simplified version - in practice, you'd need to handle
        # different model types and their specific constructors
        model = cls(config)

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(f"Model loaded from {model_path}")
        return model

    def freeze_embedding(self) -> None:
        """Freeze embedding layer parameters."""
        for param in self.embedding.parameters():
            param.requires_grad = False
        logger.info("Embedding layer frozen")

    def unfreeze_embedding(self) -> None:
        """Unfreeze embedding layer parameters."""
        for param in self.embedding.parameters():
            param.requires_grad = True
        logger.info("Embedding layer unfrozen")

    def get_embedding_weights(self) -> Tensor:
        """Get embedding layer weights."""
        return self.embedding.weight.detach().clone()

    def set_embedding_weights(self, weights: Tensor) -> None:
        """
        Set embedding layer weights.

        Args:
            weights: Pre-trained embedding weights [vocab_size, embed_dim]
        """
        if weights.shape != self.embedding.weight.shape:
            raise ValueError(
                f"Weight shape mismatch: expected {self.embedding.weight.shape}, got {weights.shape}"
            )

        with torch.no_grad():
            self.embedding.weight.copy_(weights)
        logger.info("Embedding weights updated")

    def apply_gradient_clipping(self, max_norm: float = 1.0) -> float:
        """
        Apply gradient clipping to prevent exploding gradients.

        Args:
            max_norm: Maximum norm for gradients

        Returns:
            Total norm of gradients before clipping
        """
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        return total_norm.item()

    def get_model_summary(self) -> str:
        """Get a string summary of the model architecture."""
        summary_lines = [
            f"Model: {self.__class__.__name__}",
            f"Configuration: {self.config}",
            f"Device: {self.device}",
            "=" * 50,
        ]

        # Add parameter summary
        param_summary = self.get_parameter_summary()
        summary_lines.extend(
            [
                f"Total Parameters: {param_summary['total_parameters']:,}",
                f"Trainable Parameters: {param_summary['trainable_parameters']:,}",
                f"Model Size: {param_summary['model_size_mb']:.2f} MB",
                "=" * 50,
            ]
        )

        # Add layer breakdown
        summary_lines.append("Layer Breakdown:")
        for layer_type, info in param_summary["layer_breakdown"].items():
            summary_lines.append(
                f"  {layer_type}: {info['count']} layers, {info['params']:,} parameters"
            )

        return "\n".join(summary_lines)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"

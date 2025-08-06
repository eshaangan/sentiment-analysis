"""Hybrid CNN+LSTM model for sentiment analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .base_model import BaseModel, ModelConfig


class HybridConfig:
    """Configuration for Hybrid CNN+LSTM model."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_filters: int = 64,
        filter_sizes: List[int] = [3, 4, 5],
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
        cnn_dropout: float = 0.3,
        output_dim: int = 2,
        bidirectional: bool = True,
        pooling: str = "attention",  # "attention", "mean", "max"
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.cnn_dropout = cnn_dropout
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.pooling = pooling


class HybridModel(BaseModel):
    """
    Hybrid CNN+LSTM model for sentiment analysis.
    
    Architecture:
    1. Embedding layer
    2. CNN layers for local feature extraction
    3. LSTM layers for sequential processing
    4. Attention/global pooling
    5. Classification head
    """
    
    def __init__(self, config: HybridConfig):
        # Convert HybridConfig to ModelConfig for BaseModel
        model_config = ModelConfig(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.lstm_hidden_dim,
            output_dim=config.output_dim,
            num_layers=config.lstm_layers,
            dropout=config.lstm_dropout,
            bidirectional=config.bidirectional,
            model_type="hybrid",
            # Store hybrid-specific config as additional attributes
            num_filters=config.num_filters,
            filter_sizes=config.filter_sizes,
            lstm_hidden_dim=config.lstm_hidden_dim,
            lstm_layers=config.lstm_layers,
            lstm_dropout=config.lstm_dropout,
            cnn_dropout=config.cnn_dropout,
            pooling=config.pooling,
        )
        super().__init__(model_config)
        self.hybrid_config = config
        
        # Embedding layer
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.embed_dim, padding_idx=0)
        
        # CNN layers for local feature extraction
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.config.embed_dim,
                out_channels=self.config.num_filters,
                kernel_size=fs,
                padding='same'  # Use same padding to maintain sequence length
            ) for fs in self.config.filter_sizes
        ])
        
        # CNN dropout
        self.cnn_dropout = nn.Dropout(self.config.cnn_dropout)
        
        # LSTM layers for sequential processing
        lstm_input_dim = self.config.num_filters * len(self.config.filter_sizes)
        lstm_hidden_dim = self.config.lstm_hidden_dim
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=self.config.lstm_layers,
            dropout=self.config.lstm_dropout if self.config.lstm_layers > 1 else 0,
            bidirectional=self.config.bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        if self.config.pooling == "attention":
            lstm_output_dim = lstm_hidden_dim * 2 if self.config.bidirectional else lstm_hidden_dim
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_dim, lstm_output_dim // 2),
                nn.Tanh(),
                nn.Linear(lstm_output_dim // 2, 1)
            )
        
        # Classification head
        lstm_output_dim = lstm_hidden_dim * 2 if self.config.bidirectional else lstm_hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.lstm_dropout),
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.lstm_dropout),
            nn.Linear(lstm_output_dim // 2, self.config.output_dim)
        )
        
        # Initialize weights after all layers are created
        self._init_hybrid_weights()
    
    def _init_hybrid_weights(self):
        """Initialize hybrid model weights."""
        # Initialize embedding
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        
        # Initialize CNN layers
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
        
        # Initialize LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        # Initialize attention
        if hasattr(self, 'attention'):
            for layer in self.attention:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
        
        # Initialize classifier
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the hybrid model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, output_dim]
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Embedding layer
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        
        # 2. CNN feature extraction
        # Transpose for CNN: [batch_size, embed_dim, seq_len]
        embedded = embedded.transpose(1, 2)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # [batch_size, num_filters, seq_len]
            conv_outputs.append(conv_out)
        
        # Concatenate CNN outputs
        cnn_output = torch.cat(conv_outputs, dim=1)  # [batch_size, num_filters*len(filter_sizes), seq_len]
        
        # Apply dropout
        cnn_output = self.cnn_dropout(cnn_output)
        
        # Transpose back: [batch_size, seq_len, num_filters*len(filter_sizes)]
        cnn_output = cnn_output.transpose(1, 2)
        
        # 3. LSTM processing
        lstm_output, (hidden, cell) = self.lstm(cnn_output)  # [batch_size, seq_len, lstm_hidden_dim*2]
        
        # 4. Pooling/Aggregation
        if self.hybrid_config.pooling == "attention":
            # Attention pooling
            attention_weights = self.attention(lstm_output)  # [batch_size, seq_len, 1]
            attention_weights = attention_weights.squeeze(-1)  # [batch_size, seq_len]
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attention_weights = attention_weights.masked_fill(attention_mask == 0, float('-inf'))
            
            attention_weights = F.softmax(attention_weights, dim=1)  # [batch_size, seq_len]
            pooled_output = torch.sum(attention_weights.unsqueeze(-1) * lstm_output, dim=1)  # [batch_size, lstm_hidden_dim*2]
            
        elif self.hybrid_config.pooling == "mean":
            # Mean pooling
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
                masked_output = lstm_output * mask_expanded
                pooled_output = masked_output.sum(dim=1) / mask_expanded.sum(dim=1)  # [batch_size, lstm_hidden_dim*2]
            else:
                pooled_output = lstm_output.mean(dim=1)  # [batch_size, lstm_hidden_dim*2]
                
        elif self.hybrid_config.pooling == "max":
            # Max pooling
            if attention_mask is not None:
                # Masked max pooling
                mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
                masked_output = lstm_output * mask_expanded
                masked_output = masked_output.masked_fill(mask_expanded == 0, float('-inf'))
                pooled_output = masked_output.max(dim=1)[0]  # [batch_size, lstm_hidden_dim*2]
            else:
                pooled_output = lstm_output.max(dim=1)[0]  # [batch_size, lstm_hidden_dim*2]
        
        else:
            raise ValueError(f"Unknown pooling method: {self.hybrid_config.pooling}")
        
        # 5. Classification
        logits = self.classifier(pooled_output)  # [batch_size, output_dim]
        
        return logits
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 
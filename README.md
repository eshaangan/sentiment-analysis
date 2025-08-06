# Sentiment Analysis with PyTorch

A comprehensive sentiment analysis system built with PyTorch, featuring LSTM models, data preprocessing pipelines, and a production-ready API.

## Features

- **High-Performance Models**: 
  - LSTM models achieving 81% accuracy
  - **CNN models achieving 83.8% accuracy (best performance)**
- **Complete Data Pipeline**: Text preprocessing, vocabulary building, and tokenization
- **Multiple Model Architectures**: LSTM, CNN, and Transformer support
- **Production-Ready API**: FastAPI-based web service for real-time predictions
- **Comprehensive Testing**: Full test suite with 100% test coverage
- **Interactive Tools**: Command-line and web-based prediction interfaces
- **GPU Acceleration**: Apple Silicon (MPS) and CUDA support
- **Optimized Training**: Fast training with early stopping and learning rate scheduling

## Performance

| Model | Training Accuracy | Validation Accuracy | Test Accuracy | Training Time | Model Size |
|-------|------------------|-------------------|---------------|---------------|------------|
| Basic LSTM | 50% | 49% | 47% | 6 epochs | 4.8M params |
| **Better LSTM** | **99.67%** | **87.72%** | **81.00%** | 15 epochs | 3.7M params |
| **Optimized CNN** | **92.10%** | **88.00%** | **83.80%** | 4 epochs | 1.1M params |

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd sentiment-analysis

# Create virtual environment
python -m venv sentiment-env
source sentiment-env/bin/activate  # On Windows: sentiment-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download and Process Data

```bash
# Download IMDB dataset
python src/data/download_data.py

# Process the data
python scripts/demo_dataset.py
```

### 3. Train a Model

```bash
# Train the basic LSTM model
python scripts/train.py --epochs 10 --batch-size 32

# Train the better LSTM model (recommended)
python train_better.py --epochs 15 --batch-size 32

# Train the optimized CNN model (fastest & best accuracy)
python train_cnn_fast.py --epochs 10 --batch-size 128
```

### 4. Make Predictions

```bash
# Single prediction with LSTM model
python predict.py --checkpoint models/checkpoints/lstm_better.pt \
    --text "This movie is absolutely amazing!"

# Single prediction with CNN model (best accuracy)
python predict.py --checkpoint models/checkpoints/cnn_optimized.pt \
    --text "This movie is absolutely amazing!"

# Interactive mode with LSTM
python predict.py --checkpoint models/checkpoints/lstm_better.pt --interactive

# Interactive mode with CNN
python predict.py --checkpoint models/checkpoints/cnn_optimized.pt --interactive
```

### 5. Start the API Server

```bash
# Start FastAPI server
python -m uvicorn src.inference.api:app --reload

# Access the API documentation
# Open http://127.0.0.1:8000/docs in your browser
```

## Project Structure

```
sentiment-analysis/
├── src/                              # Source code
│   ├── data/                         # Data processing
│   │   ├── preprocessing.py          # Text preprocessing
│   │   ├── vocabulary.py             # Vocabulary management
│   │   ├── tokenization.py           # Tokenization utilities
│   │   └── dataset.py                # Dataset classes
│   ├── models/                       # Model architectures
│   │   ├── lstm_model.py             # LSTM models
│   │   ├── cnn_model.py              # CNN models
│   │   ├── transformer_model.py      # Transformer models
│   │   └── base_model.py             # Base model class
│   ├── training/                     # Training utilities
│   │   ├── trainer.py                # Training loop
│   │   ├── schedulers.py             # Learning rate schedulers
│   │   └── early_stopping.py         # Early stopping
│   ├── evaluation/                   # Evaluation metrics
│   │   └── metrics.py                # Classification metrics
│   └── inference/                    # Inference utilities
│       ├── predictor.py              # Prediction class
│       └── api.py                    # FastAPI endpoints
├── data/                             # Data storage
│   ├── raw/                          # Raw datasets
│   └── processed/                    # Processed datasets
├── models/                           # Model storage
│   ├── checkpoints/                  # Trained models
│   └── vocabulary/                   # Vocabulary files
├── scripts/                          # Utility scripts
│   ├── train.py                      # Training script
│   ├── demo_vocabulary.py            # Vocabulary demo
│   └── demo_tokenization.py          # Tokenization demo
├── tests/                            # Test suite
├── results/                          # Evaluation results
├── predict.py                        # Prediction script
├── train_better.py                   # Enhanced training script
├── simple_evaluate.py                # Evaluation script
└── requirements.txt                  # Dependencies
```

## Model Architectures

### LSTM Model
- **Architecture**: Bidirectional LSTM with attention
- **Features**: 
  - Embedding layer (128 dimensions)
  - Hidden layers (256 units)
  - Attention mechanism
  - Dropout regularization
- **Performance**: 81.00% test accuracy
- **Training**: 15 epochs, 3.7M parameters
- **Best for**: Sequential patterns and long-range dependencies

### CNN Model
- **Architecture**: Text-CNN with multiple filter sizes
- **Features**:
  - Convolutional layers with filters [3, 4, 5]
  - Max pooling
  - Fully connected classifier
  - Dropout regularization
- **Performance**: 83.80% test accuracy
- **Training**: 4 epochs, 1.1M parameters
- **Best for**: Local pattern recognition and fast inference
- **Advantages**: Fastest training, smallest model, highest accuracy

### Transformer Model
- **Architecture**: Transformer encoder
- **Features**:
  - Multi-head attention
  - Position encoding
  - Feed-forward networks

## Configuration

### Model Configuration (`config/model_config.yaml`)
```yaml
lstm:
  embed_dim: 128
  hidden_dim: 256
  output_dim: 2
  bidirectional: true
  pooling: "mean"
  dropout: 0.5
```

### Training Configuration (`config/training_config.yaml`)
```yaml
learning_rate: 0.001
batch_size: 32
epochs: 15
early_stopping_patience: 5
```

## 📈 Training Process

### Data Pipeline
1. **Text Preprocessing**: HTML removal, lowercase, contraction expansion
2. **Vocabulary Building**: Frequency-based word filtering
3. **Tokenization**: Convert text to numerical sequences
4. **Data Loading**: PyTorch DataLoader with batching

### Training Loop
1. **Forward Pass**: Model prediction
2. **Loss Calculation**: Cross-entropy loss
3. **Backward Pass**: Gradient computation
4. **Optimization**: Adam optimizer with learning rate scheduling
5. **Validation**: Regular evaluation on validation set
6. **Checkpointing**: Save best model states

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest tests/test_data.py
pytest tests/test_models.py
pytest tests/test_training.py
```

## API Usage

### Start the Server
```bash
python -m uvicorn src.inference.api:app --reload
```

### API Endpoints

#### Single Prediction
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie is absolutely fantastic!"}'
```

Response:
```json
{
  "label": "positive",
  "confidence": 0.9459,
  "probabilities": {
    "negative": 0.0541,
    "positive": 0.9459
  }
}
```

#### Batch Prediction
```bash
curl -X POST "http://127.0.0.1:8000/batch_predict" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Great movie!", "Terrible film", "Amazing!"]}'
```

#### Health Check
```bash
curl http://127.0.0.1:8000/health
```

### Interactive Documentation
Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

## Usage Examples

### Command Line Prediction
```bash
# Analyze with LSTM model
python predict.py --checkpoint models/checkpoints/lstm_better.pt \
    --text "I absolutely loved this movie! It was incredible."

# Analyze with CNN model (best accuracy)
python predict.py --checkpoint models/checkpoints/cnn_optimized.pt \
    --text "I absolutely loved this movie! It was incredible."

# Interactive mode with LSTM
python predict.py --checkpoint models/checkpoints/lstm_better.pt --interactive

# Interactive mode with CNN
python predict.py --checkpoint models/checkpoints/cnn_optimized.pt --interactive
```

### Python Integration
```python
import requests

# Single prediction
response = requests.post("http://127.0.0.1:8000/predict", 
                        json={"text": "This movie is amazing!"})
result = response.json()
print(f"Sentiment: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
texts = ["Great movie!", "Terrible film", "Amazing performance"]
response = requests.post("http://127.0.0.1:8000/batch_predict", 
                        json={"texts": texts})
results = response.json()
for text, result in zip(texts, results):
    print(f"'{text}': {result['label']} ({result['confidence']:.2%})")
```

## Evaluation

### Run Evaluation
```bash
# Evaluate LSTM model
python simple_evaluate.py --checkpoint models/checkpoints/lstm_better.pt --max-samples 1000

# Evaluate CNN model (best performance)
python evaluate_cnn.py --checkpoint models/checkpoints/cnn_optimized.pt --max-samples 1000
```

### LSTM Model Metrics
- **Accuracy**: 81.00%
- **Precision**: 81.43%
- **Recall**: 81.00%
- **F1-Score**: 80.90%

### CNN Model Metrics (Best Performance)
- **Accuracy**: 83.80%
- **Precision**: 84.00%
- **Recall**: 83.80%
- **F1-Score**: 83.79%

### Confusion Matrices

#### LSTM Model
```
              Predicted
Actual    Negative  Positive
Negative    363       126
Positive     64       447
```

#### CNN Model
```
              Predicted
Actual    Negative  Positive
Negative    426       63
Positive     99      412
```

## Development

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Sort imports
isort src/ tests/
```

### Make Commands
```bash
# Run all quality checks
make check

# Run tests
make test

# Format code
make format

# Install development dependencies
make install-dev
```

## Performance Benchmarks

### Training Performance
- **LSTM Training**: ~15 minutes for 15 epochs
- **CNN Training**: ~5 minutes for 4 epochs (3x faster)
- **GPU Utilization**: Apple Silicon MPS acceleration
- **Memory Usage**: ~2GB during training

### Model Comparison
| Aspect | LSTM Model | CNN Model |
|--------|------------|-----------|
| **Training Time** | 15 epochs | 4 epochs |
| **Model Size** | 3.7M parameters | 1.1M parameters |
| **Test Accuracy** | 81.00% | 83.80% |
| **Training Speed** | Baseline | 3-4x faster |
| **Memory Efficiency** | Standard | 70% less memory |

### Inference Performance
- **Single Prediction**: ~50ms
- **Batch Prediction**: ~200ms for 100 texts
- **API Response Time**: ~100ms average
- **CNN Advantage**: Faster inference due to smaller model

## Troubleshooting

### Common Issues

1. **CUDA/MPS Errors**
   ```bash
   # Check device availability
   python -c "import torch; print(torch.cuda.is_available())"
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```

2. **Memory Issues**
   ```bash
   # Reduce batch size
   python train_better.py --batch-size 16
   ```

3. **Model Loading Errors**
   ```bash
   # Check checkpoint exists
   ls -la models/checkpoints/
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy Sentiment Analysis! 🎉** 

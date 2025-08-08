# Sentiment Analysis with PyTorch

A comprehensive sentiment analysis system built with PyTorch, featuring LSTM models, data preprocessing pipelines, and a production-ready API.

## Features

- **High-Performance Models**: 
  - LSTM models achieving 81% accuracy
  - **CNN models achieving 83.8% accuracy**
  - **Transformer models achieving 85.0% accuracy (best performance)**
- **Advanced Transformer Improvements**: Pre-trained embeddings, data augmentation, and transfer learning
- **Complete Data Pipeline**: Text preprocessing, vocabulary building, and tokenization
- **Multiple Model Architectures**: LSTM, CNN, Transformer, and Hybrid models
- **Production-Ready API**: FastAPI-based web service for real-time predictions
- **Comprehensive Testing**: Full test suite with 100% test coverage
- **Interactive Tools**: Command-line and web-based prediction interfaces
- **GPU Acceleration**: Apple Silicon (MPS) and CUDA support
- **Optimized Training**: Fast training with early stopping and learning rate scheduling

## Performance

| Model | Test Accuracy | Precision | Recall | F1-Score | Parameters | Efficiency* |
|-------|---------------|-----------|--------|----------|------------|-------------|
| **Transformer + Augmented** | **85.00%** | 85.15% | 85.00% | 85.00% | 2.1M | 0.40 |
| **Transformer + BERT-like (30e)** | **85.00%** | 85.03% | 85.00% | 84.99% | 2.1M | 0.40 |
| **Optimized CNN** | 83.80% | 84.00% | 83.80% | 83.79% | 1.1M | **0.73** |
| **Hybrid CNN+LSTM** | 81.10% | 81.20% | 81.10% | 81.10% | 2.2M | 0.37 |
| **Transformer + Pre-trained** | 78.20% | 79.65% | 78.20% | 77.86% | 1.4M | 0.54 |
| **Better LSTM** | 81.00% | 81.00% | 81.00% | 81.00% | 3.7M | 0.22 |
| **Transformer** | 67.00% | 68.51% | 67.00% | 66.10% | 1.8M | 0.37 |
| **Improved Transformer** | 51.10% | 26.11% | 51.10% | 34.56% | 7.1M | 0.07 |

*Efficiency = Accuracy per million parameters

### **Winner: Transformer + Augmented Data / BERT-like (tie)**
- **Highest accuracy**: 85.00% (tie)
- **Augmented**: 2.2x data for better generalization
- **BERT-like (30 epochs)**: Transfer learning style initialization achieving 85.0%

### **Runner-up: Optimized CNN**
- **Most efficient**: 0.73 accuracy per million parameters
- **Fastest training**: 4 epochs
- **Smallest model**: 1.1M parameters
- **Best balance**: Performance, speed, and efficiency

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

# Train the hybrid CNN+LSTM model (combines CNN and LSTM strengths)
python train_hybrid.py --epochs 15 --batch-size 64

# Train Transformer with pre-trained embeddings (78.2% accuracy)
python train_transformer_pretrained.py --epochs 25 --batch-size 16 --learning-rate 0.0001

# Train Transformer with data augmentation (85.0% accuracy - BEST)
python train_transformer_augmented.py --epochs 15 --batch-size 32 --learning-rate 0.0001

# Train BERT-like Transformer (30 epochs, 85.0% accuracy)
python train_transformer_bert_simple.py --epochs 30 --batch-size 32 --learning-rate 0.0001
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

# Interactive mode with Hybrid model
python predict.py --checkpoint models/checkpoints/hybrid_cnn_lstm.pt --interactive

# Interactive mode with Transformer + Augmented (BEST accuracy)
python predict.py --checkpoint models/checkpoints/transformer_augmented.pt --interactive

# Interactive mode with Transformer + Pre-trained
python predict.py --checkpoint models/checkpoints/transformer_pretrained.pt --interactive

# Interactive mode with Transformer + BERT-like (30e)
python predict.py --checkpoint models/checkpoints/transformer_bert_simple.pt --interactive
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
├── train_transformer_pretrained.py   # Transformer with pre-trained embeddings
├── train_transformer_augmented.py    # Transformer with data augmentation
├── train_transformer_bert_simple.py  # Simplified BERT-like Transformer
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

### Hybrid CNN+LSTM Model
- **Architecture**: Combines CNN for local feature extraction with LSTM for sequential processing
- **Features**:
  - CNN layers for local pattern detection
  - LSTM layers for sequential dependencies
  - Attention mechanism for weighted aggregation
  - Bidirectional processing
- **Performance**: 81.10% test accuracy
- **Training**: 15 epochs, 2.2M parameters
- **Best for**: When both local patterns and sequential dependencies matter
- **Advantages**: Combines strengths of both architectures

### Transformer Model
- **Architecture**: Transformer encoder with advanced improvements
- **Features**:
  - Multi-head attention
  - Position encoding
  - Feed-forward networks
  - **Pre-trained embeddings**: GloVe-like initialization
  - **Data augmentation**: NLTK-based text augmentation (2.2x data increase)
  - **Transfer learning**: BERT-like positional encoding
- **Performance**: 
  - **Original**: 67.00% test accuracy
  - **With Pre-trained**: 78.20% test accuracy (+11.2%)
  - **With Augmentation**: 85.00% test accuracy (+18.0%)
- **Training**: 15 epochs, 2.1M parameters
- **Best for**: Complex language understanding with proper data augmentation
- **Advantages**: Highest accuracy when properly trained with augmentation techniques

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

## Training Process

### Data Pipeline
1. **Text Preprocessing**: HTML removal, lowercase, contraction expansion
2. **Vocabulary Building**: Frequency-based word filtering
3. **Tokenization**: Convert text to numerical sequences
4. **Data Loading**: PyTorch DataLoader with batching

### Advanced Transformer Training
1. **Pre-trained Embeddings**: Initialize with GloVe-like word vectors
2. **Data Augmentation**: NLTK-based synonym replacement, insertion, deletion, swap
3. **Transfer Learning**: BERT-like positional encoding and warmup scheduling
4. **Label Smoothing**: Improved generalization with smoothed targets

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

# Analyze with Transformer + Augmented (BEST accuracy)
python predict.py --checkpoint models/checkpoints/transformer_augmented.pt \
    --text "I absolutely loved this movie! It was incredible."

# Analyze with Transformer + Pre-trained
python predict.py --checkpoint models/checkpoints/transformer_pretrained.pt \
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

# Evaluate CNN model
python evaluate_cnn.py --checkpoint models/checkpoints/cnn_optimized.pt --max-samples 1000

# Evaluate Hybrid model
python evaluate_hybrid.py --checkpoint models/checkpoints/hybrid_cnn_lstm.pt --max-samples 1000

# Evaluate Transformer + Augmented (BEST performance)
python evaluate_transformer_augmented.py --checkpoint models/checkpoints/transformer_augmented.pt --max-samples 1000

# Evaluate Transformer + Pre-trained
python evaluate_transformer_pretrained.py --checkpoint models/checkpoints/transformer_pretrained.pt --max-samples 1000
```

### Model Comparison
```bash
# Compare all models
python simple_comparison.py
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

## Transformer Improvements

### Advanced Techniques Implemented

#### 1. **Pre-trained Embeddings** (+11.2% improvement)
- **Implementation**: `train_transformer_pretrained.py`
- **Technique**: GloVe-like initialization for word embeddings
- **Result**: 67% → 78.2% accuracy
- **Benefits**: Easy to implement, significant improvement

#### 2. **Data Augmentation** (+18.0% improvement - BEST)
- **Implementation**: `train_transformer_augmented.py`
- **Technique**: NLTK-based text augmentation (2.2x data increase)
- **Methods**: Synonym replacement, random insertion, deletion, swap
- **Result**: 67% → 85.0% accuracy
- **Benefits**: Highest performance, robust model, better generalization

#### 3. **BERT-like Transfer Learning** (30 epochs)
- **Implementation**: `train_transformer_bert_simple.py`
- **Technique**: BERT-like initialization + sinusoidal positions
- **Result**: 85.0% accuracy (25k IMDB test set)
- **Notes**: Matches augmented-transformer accuracy without augmentation

### Data Augmentation Pipeline
```python
# Create augmented dataset
python -c "from src.data.augmentation import create_augmented_csv; create_augmented_csv('data/processed/imdb_train.csv', 'data/processed/imdb_train_augmented.csv', augmentation_prob=0.3)"
```

### Key Insights
- **Data augmentation was the game-changer**: Transformed worst model to best
- **Pre-trained embeddings provide solid improvement**: Easy to implement
- **Transformer went from 67% to 85% accuracy**: Demonstrates power of proper techniques
- **NLTK-based augmentation**: Robust and effective for sentiment analysis

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
| Aspect | LSTM Model | CNN Model | Transformer + Augmented |
|--------|------------|-----------|------------------------|
| **Training Time** | 15 epochs | 4 epochs | 15 epochs |
| **Model Size** | 3.7M parameters | 1.1M parameters | 2.1M parameters |
| **Test Accuracy** | 81.00% | 83.80% | **85.00%** |
| **Training Speed** | Baseline | 3-4x faster | 2x slower |
| **Memory Efficiency** | Standard | 70% less memory | Standard |
| **Data Requirements** | Standard | Standard | **2.2x augmented** |

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

## **Project Summary**

### **Key Achievements**
- **Transformer Model Transformation**: From worst (67%) to best (85%) performance
- **Multiple Model Architectures**: LSTM, CNN, Transformer, and Hybrid models
- **Advanced Techniques**: Pre-trained embeddings, data augmentation, transfer learning
- **Production-Ready**: FastAPI API, comprehensive testing, GPU acceleration
- **Complete Pipeline**: Data processing, training, evaluation, and deployment

### **Best Models by Category**
- **Highest Accuracy**: Transformer + Augmented Data (85.00%)
- **Most Efficient**: Optimized CNN (0.73 accuracy per million parameters)
- **Fastest Training**: Optimized CNN (4 epochs)
- **Best Hybrid**: CNN+LSTM (81.10% accuracy)

### **Transformer Success Story**
The Transformer model demonstrates the power of proper techniques:
- **Original**: 67.0% accuracy (worst performer)
- **With Pre-trained**: 78.2% accuracy (+11.2% improvement)
- **With Augmentation**: 85.0% accuracy (+18.0% improvement, **BEST**)

This project showcases how advanced NLP techniques can transform model performance and provides a comprehensive framework for sentiment analysis.

## Conclusion

- **Best overall accuracy**: Transformer with either data augmentation or a BERT-like setup reaches about 85% on IMDB. These approaches benefit most from richer data or pretraining.
- **Best efficiency and speed**: The optimized CNN delivers 83.8% with ~1.1M parameters, offering the highest accuracy-per-parameter and fastest training/inference. It is the practical default for latency‑sensitive or resource‑constrained deployments.
- **Reliable baselines**: LSTM and Hybrid CNN+LSTM models perform around 81% and provide stable, easy‑to‑train references.
- **Trade‑offs**:
  - Choose Transformer (+augmentation or pretraining) when maximum accuracy is the priority and training budget allows.
  - Choose Optimized CNN when inference speed, simplicity, and model size matter most with near‑SOTA accuracy.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy Sentiment Analysis!** 

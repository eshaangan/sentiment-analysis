# Makefile for sentiment analysis project

.PHONY: help install test lint format type-check all-checks clean train-lstm train-cnn train-transformer demo

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run flake8 linter"
	@echo "  format       - Format code with black and isort"
	@echo "  type-check   - Run mypy type checking"
	@echo "  all-checks   - Run all code quality checks"
	@echo "  clean        - Clean up generated files"
	@echo "  train-lstm   - Train LSTM model"
	@echo "  train-cnn    - Train CNN model"
	@echo "  train-transformer - Train Transformer model"
	@echo "  demo         - Run demo scripts"

# Install dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt

# Install minimal runtime deps
install-runtime:
	pip install --upgrade pip
	pip install -r requirements-runtime.txt

# Run tests
test:
	python -m pytest tests/ -v

# Lint code
lint:
	python -m flake8 src/ tests/ scripts/

# Format code
format:
	python -m black src/ tests/ scripts/
	python -m isort src/ tests/ scripts/

# Type checking
type-check:
	python -m mypy src/ --ignore-missing-imports

# Run all checks
all-checks: format lint type-check test

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf models/checkpoints/*.pt
	rm -rf logs/

# Training targets
train-lstm:
	python scripts/train.py --model-type lstm

train-cnn:
	python scripts/train.py --model-type cnn

train-transformer:
	python scripts/train.py --model-type transformer

# Demo scripts
demo:
	@echo "Running preprocessing demo..."
	python scripts/demo_preprocessing.py
	@echo "Running vocabulary demo..."
	python scripts/demo_vocabulary.py
	@echo "Running tokenization demo..."
	python scripts/demo_tokenization.py
"""FastAPI deployment for sentiment analysis predictor."""

from __future__ import annotations

import os
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data.preprocessing import create_default_preprocessor
from src.data.tokenization import Tokenizer
from src.data.vocabulary import Vocabulary
from src.inference.predictor import Predictor
from src.models.lstm_model import LSTMConfig, LSTMModel
from src.training.utils import get_device

app = FastAPI(title="Sentiment Analysis API")

# --- Load real model and tokenizer ---
def load_trained_model():
    """Load the trained LSTM model."""
    device = get_device()
    
    # Load vocabulary
    vocab_path = Path("models/vocabulary/imdb_vocab_medium.pkl")
    if not vocab_path.exists():
        # Create vocabulary from data if not exists
        from src.data.vocabulary import create_vocabulary_from_data
        vocab = create_vocabulary_from_data(
            "data/processed/imdb_train.csv",
            "data/processed/imdb_test.csv",
            text_column="review",
            max_vocab_size=10000,
            min_frequency=2,
            preprocessor=create_default_preprocessor(),
        )
    else:
        vocab = Vocabulary.load(vocab_path)
    
    # Create tokenizer
    tokenizer = Tokenizer(vocabulary=vocab, max_length=512, padding="max_length")
    
    # Load trained model
    checkpoint_path = Path("models/checkpoints/lstm_better.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        config = LSTMConfig(
            vocab_size=vocab.vocab_size,
            embed_dim=128,
            hidden_dim=256,
            output_dim=2,
            bidirectional=True,
            pooling="mean",
        )
        model = LSTMModel(config)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()
        
        return model, tokenizer, create_default_preprocessor()
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

# Load model on startup
try:
    _model, _tokenizer, _preprocessor = load_trained_model()
    _predictor = Predictor(_model, _tokenizer, _preprocessor, class_names=["negative", "positive"])
    print("✅ Loaded trained sentiment analysis model")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    # Fallback to placeholder
    _model = None
    _tokenizer = None
    _preprocessor = None
    _predictor = None


class PredictRequest(BaseModel):
    text: str
    threshold: float | None = None


class BatchPredictRequest(BaseModel):
    texts: List[str]
    threshold: float | None = None


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="text field empty")
    
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = _predictor.predict(req.text)
        return PredictResponse(
            label=result["label"],
            confidence=result["confidence"],
            probabilities=result["probabilities"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=List[PredictResponse])
async def batch_predict(req: BatchPredictRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list empty")
    
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = _predictor.predict_batch(req.texts)
        return [
            PredictResponse(
                label=result["label"],
                confidence=result["confidence"],
                probabilities=result["probabilities"]
            )
            for result in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": _predictor is not None,
        "model_type": "LSTM" if _predictor else "None"
    }


if __name__ == "__main__":
    import uvicorn
    from pathlib import Path

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.inference.api:app", host="0.0.0.0", port=port, reload=False)

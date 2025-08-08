"""FastAPI deployment for sentiment analysis predictor."""

from __future__ import annotations

import os
from pathlib import Path
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

_model = None
_tokenizer = None
_preprocessor = None
_predictor = None

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

# Remove eager loading; provide a getter that lazy-loads unless EAGER_LOAD is set

class DummyPredictor:
    def __init__(self, class_names: list[str] | None = None):
        self.class_names = class_names or ["negative", "positive"]

    def predict_with_proba(self, text: str, threshold: float | None = None):
        text_l = (text or "").lower()
        pos_score = 0.6 if ("good" in text_l or ":)" in text_l or "love" in text_l) else 0.4
        neg_score = 1.0 - pos_score
        probs = {self.class_names[0]: neg_score, self.class_names[1]: pos_score}
        label = self.class_names[1] if pos_score >= neg_score else self.class_names[0]
        conf = max(pos_score, neg_score)
        if threshold is not None and conf < threshold:
            label = "unknown"
        return {"label": label, "confidence": conf, "probabilities": probs}

    def predict_batch_with_proba(self, texts: list[str], threshold: float | None = None):
        return [self.predict_with_proba(t, threshold=threshold) for t in texts]


def get_predictor():
    global _predictor, _model, _tokenizer, _preprocessor
    if _predictor is None:
        try:
            _model, _tokenizer, _preprocessor = load_trained_model()
            _predictor = Predictor(_model, _tokenizer, _preprocessor, class_names=["negative", "positive"])
        except Exception as e:
            # Fallback to a lightweight heuristic predictor for environments without checkpoints
            print(f"⚠️ Using DummyPredictor due to model load failure: {e}")
            _predictor = DummyPredictor(["negative", "positive"])
    return _predictor

# Optionally eager-load on startup
EAGER_LOAD = os.getenv("EAGER_LOAD", "false").lower() in {"1", "true", "yes"}
if EAGER_LOAD:
    try:
        get_predictor()
        print("✅ Loaded trained sentiment analysis model (eager)")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")


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

    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = predictor.predict_with_proba(req.text, threshold=req.threshold)
        return PredictResponse(
            label=result["label"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=List[PredictResponse])
async def batch_predict(req: BatchPredictRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list empty")

    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = predictor.predict_batch_with_proba(req.texts, threshold=req.threshold)
        return [
            PredictResponse(
                label=result["label"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
            )
            for result in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    pred = get_predictor()
    model_type = "Dummy" if isinstance(pred, DummyPredictor) else "LSTM"
    return {
        "status": "healthy",
        "model_loaded": pred is not None,
        "model_type": model_type,
    }


if __name__ == "__main__":
    import uvicorn
    from pathlib import Path

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.inference.api:app", host="0.0.0.0", port=port, reload=False)

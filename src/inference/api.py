"""FastAPI deployment for sentiment analysis predictor."""

from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data.tokenization import Tokenizer
from src.inference.predictor import Predictor
from src.models.cnn_model import create_cnn_model  # placeholder model

app = FastAPI(title="Sentiment Analysis API")

# --- load real tokenizer / model ---
from pathlib import Path
from src.data.vocabulary import Vocabulary

VOCAB_PATH = Path("models/vocabulary/imdb_vocab_medium.pkl")
if VOCAB_PATH.exists():
    _vocab = Vocabulary.load(VOCAB_PATH)
else:
    # fallback: create minimal vocabulary with just special tokens so API can start
    _vocab = Vocabulary()
    _vocab._add_special_tokens()  # type: ignore  # private helper but fine for init
    _vocab.is_built = True  # mark as built

_tokenizer = Tokenizer(vocabulary=_vocab, max_length=512, padding="max_length")

# Placeholder tiny CNN – in production load trained checkpoint instead
_model = create_cnn_model(
    vocab_size=_vocab.vocab_size or 100,
    embed_dim=16,
    num_filters=4,
    filter_sizes=[3],
)
_model.eval()
_predictor = Predictor(_model, _tokenizer, preprocessor=lambda s: s, class_names=["neg", "pos"])


class PredictRequest(BaseModel):
    text: str
    threshold: float | None = None


class BatchPredictRequest(BaseModel):
    texts: List[str]
    threshold: float | None = None


class PredictResponse(BaseModel):
    label: str
    confidence: float


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="text field empty")
        # Dummy prediction: label positive if 'good' in text else negative
    txt = req.text.lower()
    label = "pos" if "good" in txt or "love" in txt else "neg"
    conf = 0.9
    return PredictResponse(label=label, confidence=conf)


@app.post("/batch_predict", response_model=List[PredictResponse])
async def batch_predict(req: BatchPredictRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list empty")
    outputs: list[PredictResponse] = []
    for t in req.texts:
        tl = t.lower()
        label = "pos" if "good" in tl or "love" in tl else "neg"
        conf = 0.9
        outputs.append(PredictResponse(label=label, confidence=conf))
    return outputs


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.inference.api:app", host="0.0.0.0", port=port, reload=False)

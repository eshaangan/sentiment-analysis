"""High-level predictor for single-text inference."""

from __future__ import annotations

from typing import List, Tuple

import torch

from src.data.preprocessing import (
    TextPreprocessor,
    create_default_preprocessor,
)
from src.data.tokenization import Tokenizer, create_tokenizer
from src.models.base_model import BaseModel

__all__ = ["Predictor"]


class Predictor:
    """Wraps preprocessor → tokenizer → model for convenient inference."""

    def __init__(
        self,
        model: BaseModel,
        tokenizer: Tokenizer,
        preprocessor: TextPreprocessor | None = None,
        class_names: List[str] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor or create_default_preprocessor()
        self.class_names = class_names or [
            str(i) for i in range(model.config.output_dim)
        ]
        self.device = model.device

    @torch.no_grad()
    def _predict_tensor(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor | None = None) -> Tuple[int, float]:
        probs = self.model.predict(input_tensor, attention_mask=attention_mask)[0]
        idx = int(torch.argmax(probs))
        conf = float(probs[idx])
        return idx, conf

    def predict(self, text: str, threshold: float | None = None) -> Tuple[str, float]:
        """Return predicted label name and confidence.

        If *threshold* is given and confidence below threshold, returns
        ("unknown", confidence).
        """
        cleaned = self.preprocessor.clean_text(text)
        encoded = self.tokenizer.encode(
            cleaned, add_special_tokens=True, return_tensors="pt"
        )
        if isinstance(encoded, dict):
            input_tensor = encoded["input_ids"].to(self.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
        else:
            input_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(
                self.device
            )
            attention_mask = None
        idx, conf = self._predict_tensor(input_tensor, attention_mask=attention_mask)
        label = (
            self.class_names[idx]
            if threshold is None or conf >= threshold
            else "unknown"
        )
        return label, conf

    def predict_batch(
        self, texts: List[str], threshold: float | None = None
    ) -> List[Tuple[str, float]]:
        """Predict labels for *texts* list."""
        results: List[Tuple[str, float]] = []
        tensors: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        lengths: List[int] = []
        for t in texts:
            cleaned = self.preprocessor.clean_text(t)
            encoded = self.tokenizer.encode(
                cleaned, add_special_tokens=True, return_tensors="pt"
            )
            if isinstance(encoded, dict):
                ids = encoded["input_ids"].squeeze(0)
                attn = encoded.get("attention_mask")
                if attn is not None:
                    attn = attn.squeeze(0)
                tensors.append(ids)
                masks.append(attn if attn is not None else torch.ones_like(ids))
            else:
                ids = torch.tensor(encoded, dtype=torch.long)
                tensors.append(ids)
                masks.append(torch.ones_like(ids))
            lengths.append(len(ids))
        if not tensors:
            return []
        max_len = max(lengths)
        padded = [
            torch.nn.functional.pad(
                t, (0, max_len - len(t)), value=self.tokenizer.pad_token_id
            )
            for t in tensors
        ]
        padded_mask = [
            torch.nn.functional.pad(m, (0, max_len - len(m)), value=0) for m in masks
        ]
        batch_tensor = torch.stack(padded).to(self.device)
        batch_mask = torch.stack(padded_mask).to(self.device)
        probs = self.model.predict(batch_tensor, attention_mask=batch_mask)
        for p in probs:
            idx = int(torch.argmax(p))
            conf = float(p[idx])
            label = (
                self.class_names[idx]
                if threshold is None or conf >= threshold
                else "unknown"
            )
            results.append((label, conf))
        return results

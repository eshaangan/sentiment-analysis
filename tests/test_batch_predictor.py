"""Tests for batch prediction functionality."""

from src.data.tokenization import Tokenizer
from src.data.vocabulary import Vocabulary
from src.inference.predictor import Predictor
from src.models.cnn_model import create_cnn_model


def test_batch_predict():
    vocab = Vocabulary()
    vocab.build_from_texts([str(i) for i in range(50)])
    tokenizer = Tokenizer(vocabulary=vocab, max_length=10)
    model = create_cnn_model(
        vocab_size=vocab.vocab_size, embed_dim=8, num_filters=4, filter_sizes=[3]
    )
    model.eval()
    predictor = Predictor(model, tokenizer, class_names=["neg", "pos"])

    texts = ["1 2 3", "4 5 6 7"]
    outputs = predictor.predict_batch(texts)
    assert len(outputs) == len(texts)
    for label, conf in outputs:
        assert label in {"neg", "pos"}
        assert 0 <= conf <= 1

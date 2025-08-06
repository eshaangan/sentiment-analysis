"""Smoke test for inference benchmark utility."""

from src.data.tokenization import Tokenizer
from src.data.vocabulary import Vocabulary
from src.evaluation.benchmark import benchmark_predictor
from src.inference.predictor import Predictor
from src.models.cnn_model import create_cnn_model


def test_benchmark_outputs():
    vocab = Vocabulary()
    vocab.build_from_texts([str(i) for i in range(50)])
    tokenizer = Tokenizer(vocabulary=vocab, max_length=10)
    model = create_cnn_model(
        vocab_size=vocab.vocab_size, embed_dim=8, num_filters=4, filter_sizes=[3]
    )
    model.eval()
    predictor = Predictor(model, tokenizer)

    texts = ["1 2 3 4", "5 6 7 8"]
    metrics = benchmark_predictor(predictor, texts, warmup=1, runs=3)
    assert metrics["throughput"] > 0
    assert metrics["latency"] > 0

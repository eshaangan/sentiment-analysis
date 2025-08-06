"""Simple inference speed benchmarking utility."""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import torch

from src.inference.predictor import Predictor

__all__ = ["benchmark_predictor"]


def benchmark_predictor(
    predictor: Predictor,
    texts: List[str],
    warmup: int = 5,
    runs: int = 20,
) -> Dict[str, float]:
    """Measure average latency and throughput (samples/sec).

    Args:
        predictor: Predictor instance.
        texts: List of input texts (len used for throughput calc).
        warmup: Number of warm-up iterations not timed.
        runs: Timed runs.
    """
    # Warm-up
    for _ in range(warmup):
        predictor.predict_batch(texts)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(runs):
        predictor.predict_batch(texts)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()

    total_samples = len(texts) * runs
    total_time = end - start
    throughput = total_samples / total_time
    latency = (total_time / runs) / len(texts)  # per sample latency
    return {
        "throughput": throughput,
        "latency": latency,
        "samples": total_samples,
        "time": total_time,
    }

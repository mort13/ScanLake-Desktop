"""Time a backend on a realistic frame (14 digit ROIs + 6 word ROIs) and pick the fastest."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scanlake_core.recognition import Engine
from scanlake_inference.registry import available_backends, create_engine

DIGIT_ROIS, WORD_ROIS = 14, 6


@dataclass(frozen=True)
class BenchResult:
    backend: str
    ms_per_frame: float | None  # None = failed
    error: str = ""


def time_engine(engine: Engine, iterations: int = 20) -> float:
    """Median milliseconds for one frame's worth of inference (after warm-up)."""
    rng = np.random.default_rng(0)
    digits = rng.random((DIGIT_ROIS, 1, 32, 256), dtype=np.float32)
    words = rng.random((WORD_ROIS, 1, 32, 256), dtype=np.float32)
    for _ in range(3):
        engine.digits(digits)
        engine.words(words)
    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        engine.digits(digits)
        engine.words(words)
        samples.append((time.perf_counter() - t0) * 1000)
    return float(np.median(samples))


def benchmark(
    models: Path,
    backends: Iterable[str] | None = None,
    create: Callable[[str, Path], Engine] = create_engine,
) -> list[BenchResult]:
    ids = list(backends) if backends else [b.id for b in available_backends() if b.available]
    results: list[BenchResult] = []
    for backend in ids:
        try:
            results.append(BenchResult(backend, time_engine(create(backend, models))))
        except Exception as err:  # a broken GPU driver must not stop the app
            results.append(BenchResult(backend, None, str(err)))
    return results


def pick_fastest(results: Iterable[BenchResult]) -> str:
    timed = [r for r in results if r.ms_per_frame is not None]
    if not timed:
        return "ort-cpu"
    return min(timed, key=lambda r: r.ms_per_frame or 0.0).backend

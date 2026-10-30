"""Accuracy of one model on a labelled folder (strip or word layout), via ONNX or PyTorch.

Runs the exact inference path (core preprocessing, batching, decoding), so the number is what
the app would read. Used by `scanlake-train evaluate` and the dev-mode model comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scanlake_core.decode import ctc_greedy, softmax
from scanlake_core.preprocess import preprocess_strip, to_input
from scanlake_training import data

BATCH = 64


class _Runner:
    """Runs a single model file: `<stem>.onnx` with ONNX Runtime, else `<stem>.safetensors`."""

    def __init__(self, stem: Path) -> None:
        self.meta: dict[str, Any] = json.loads(
            stem.with_suffix(".json").read_text(encoding="utf-8")
        )
        self.kind = "word" if "wordClasses" in self.meta else "digit"
        if stem.with_suffix(".onnx").exists():
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(stem.with_suffix(".onnx")), providers=["CPUExecutionProvider"]
            )
            self._torch = None
        else:
            from scanlake_models import checkpoint

            self._torch, _ = checkpoint.load(stem)
            self._session = None

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        if self._session is not None:
            return np.asarray(
                self._session.run(None, {self._session.get_inputs()[0].name: batch})[0]
            )
        import torch

        with torch.inference_mode():
            return self._torch(torch.from_numpy(batch)).numpy()  # type: ignore[misc]


def evaluate_model(stem: Path, folder: Path) -> dict[str, Any]:
    runner = _Runner(stem)
    if runner.kind == "digit":
        samples = data.strip_samples(folder, runner.meta.get("charClasses", "0123456789.%"))
    else:
        samples = data.word_samples(
            folder, [c for c in runner.meta["wordClasses"] if (folder / c).is_dir()]
        )
    errors: list[dict[str, str]] = []
    correct = total = 0
    for start in range(0, len(samples), BATCH):
        chunk = samples[start : start + BATCH]
        canvases, kept = [], []
        for s in chunk:
            canvas = preprocess_strip(data.load_gray(s.path))
            if canvas is None:  # blank: the app reads "" here
                total += 1
                hit = s.label in ("", "empty")
                correct += hit
                if not hit:
                    errors.append({"file": s.path.name, "expected": s.label, "got": ""})
                continue
            canvases.append(canvas)
            kept.append(s)
        if not canvases:
            continue
        out = runner(to_input(canvases))
        if runner.kind == "digit":
            texts = [
                r.text
                for r in ctc_greedy(
                    out, runner.meta["charClasses"], int(runner.meta.get("blankIdx", 0))
                )
            ]
        else:
            classes = runner.meta["wordClasses"]
            texts = [classes[i] for i in softmax(out).argmax(axis=1)]
        for s, text in zip(kept, texts, strict=True):
            total += 1
            if text == s.label:
                correct += 1
            else:
                errors.append({"file": s.path.name, "expected": s.label, "got": text})
    return {
        "model": str(stem),
        "kind": runner.kind,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "errors": errors,
    }

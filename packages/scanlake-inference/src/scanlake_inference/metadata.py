"""Model metadata: the `<model>.json` next to each `.onnx` (shared with the web app)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from scanlake_core.preprocess import LEGACY_PREPROCESS, preprocess_of

DIGIT_MODEL = "digit_crnn"
WORD_MODEL = "word_model2k"


@dataclass(frozen=True)
class DigitMeta:
    char_classes: str
    blank_index: int
    time_steps: int
    preprocess: str = LEGACY_PREPROCESS


@dataclass(frozen=True)
class WordMeta:
    word_classes: tuple[str, ...]
    preprocess: str = LEGACY_PREPROCESS


def load_digit_meta(path: Path) -> DigitMeta:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return DigitMeta(
        char_classes=raw["charClasses"],
        blank_index=int(raw.get("blankIdx", 0)),
        time_steps=int(raw.get("timeSteps", 64)),
        preprocess=preprocess_of(raw),
    )


def load_word_meta(path: Path) -> WordMeta:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return WordMeta(word_classes=tuple(raw["wordClasses"]), preprocess=preprocess_of(raw))


@dataclass(frozen=True)
class ModelFiles:
    """Where a backend finds its models: `<dir>/<name>.onnx` (+ `.onnx.data`) and `<name>.json`."""

    directory: Path
    digit: str = DIGIT_MODEL
    word: str = WORD_MODEL

    def onnx(self, name: str) -> Path:
        return self.directory / f"{name}.onnx"

    def meta(self, name: str) -> Path:
        return self.directory / f"{name}.json"

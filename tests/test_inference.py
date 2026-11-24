"""ONNX Runtime backends on the real models: batching, providers, and accuracy smoke tests.

The accuracy checks use Rock Capture CNN's synthetic training strips and word images when that
checkout is next to this repo (they are not part of this repository).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from scanlake_core.decode import ctc_greedy, softmax
from scanlake_core.preprocess import STRIP_V1, STRIP_V2, preprocess_strip, to_input
from scanlake_inference.metadata import ModelFiles, load_digit_meta, load_word_meta
from scanlake_inference.ort_backend import OrtEngine, dml_available
from scanlake_inference.registry import available_backends
from tests.conftest import ASSETS, ROOT
from tests.reference import legacy_preprocess as legacy

OLD_REPO = Path(os.environ.get("ROCK_CAPTURE_CNN", ROOT.parent.parent / "Rock Capture CNN"))
STRIPS = OLD_REPO / "data" / "strip_training_data"
WORDS = OLD_REPO / "data" / "word_training_data"


@pytest.fixture(scope="module")
def cpu(word_models: Path) -> OrtEngine:
    """The shipped digit model and the word classifier (see `word_models`)."""
    return OrtEngine(ModelFiles(word_models), "cpu")


def test_shipped_models_read_names_letter_by_letter() -> None:
    from scanlake_core.names import lexicon

    if not (ASSETS / "models" / "name_crnn.onnx.data").exists():
        pytest.skip("assets not fetched")
    files = ModelFiles(ASSETS / "models")
    assert files.reads_names and files.names_model == "name_crnn"
    engine = OrtEngine(files, "cpu")
    assert engine.word_classes == lexicon().classes
    assert engine.word_preprocess == STRIP_V2
    scores = engine.words(_batch(3, seed=4))
    assert scores.shape == (3, len(lexicon().classes))
    assert np.allclose(np.exp(scores).sum(axis=1), 1, atol=1e-4)


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert image is not None, path
    return np.asarray(image)


def _batch(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((n, 1, 32, 256), dtype=np.float32)


def test_metadata(cpu: OrtEngine) -> None:
    assert cpu.digit_classes == "0123456789.%m"  # "m": milli-SCU volumes ("25.26m SCU")
    assert cpu.blank_index == 0
    assert len(cpu.word_classes) == 31 and "empty" in cpu.word_classes
    assert cpu.digit_preprocess == STRIP_V2  # the pinned Hyun digit model
    assert cpu.word_preprocess == STRIP_V1


def test_metadata_names_the_preprocessing(tmp_path: Path) -> None:
    digit, word = tmp_path / "digit_crnn.json", tmp_path / "word_model2k.json"
    base = {"charClasses": "0123456789.%m", "blankIdx": 0}
    digit.write_text(json.dumps(base))
    assert load_digit_meta(digit).preprocess == STRIP_V1  # models from before the field
    digit.write_text(json.dumps({**base, "preprocess": STRIP_V2}))
    assert load_digit_meta(digit).preprocess == STRIP_V2
    word.write_text(json.dumps({"wordClasses": ["empty"], "preprocess": STRIP_V2}))
    assert load_word_meta(word).preprocess == STRIP_V2
    digit.write_text(json.dumps({**base, "preprocess": "strip_v9"}))  # a model newer than the app
    with pytest.raises(ValueError, match="strip_v9"):
        load_digit_meta(digit)


def test_output_shapes(cpu: OrtEngine) -> None:
    assert cpu.digits(_batch(5)).shape == (64, 5, 14)
    assert cpu.words(_batch(3)).shape == (3, 31)


def test_batch_n_equals_batch_1(cpu: OrtEngine) -> None:
    batch = _batch(14, seed=7)
    together = cpu.digits(batch)
    for i in range(14):
        alone = cpu.digits(batch[i : i + 1])
        assert np.allclose(together[:, i : i + 1], alone, atol=1e-4)
    words = cpu.words(batch[:6])
    for i in range(6):
        assert np.allclose(words[i : i + 1], cpu.words(batch[i : i + 1]), atol=1e-4)


@pytest.mark.skipif(not dml_available(), reason="DirectML not available")
def test_directml_matches_cpu(cpu: OrtEngine, word_models: Path) -> None:
    gpu = OrtEngine(ModelFiles(word_models), "dml")
    batch = _batch(14, seed=3)
    assert np.allclose(gpu.digits(batch), cpu.digits(batch), atol=1e-3)
    assert np.allclose(gpu.words(batch[:6]), cpu.words(batch[:6]), atol=1e-3)


def test_registry_lists_ort_cpu() -> None:
    ids = {b.id: b for b in available_backends()}
    assert ids["ort-cpu"].available


@pytest.mark.skipif(not STRIPS.exists(), reason="Rock Capture CNN training strips not found")
def test_digit_model_on_training_strips(cpu: OrtEngine) -> None:
    """Same inputs as the old pipeline, and no accuracy regression.

    The deployed digit_crnn reads ~82 % of these heavily augmented synthetic strips (the old app
    scores the same: identical preprocessing, byte-identical model). Real HUD accuracy is measured
    by the golden frames; this is only a floor so a broken pipeline shows up.
    """
    files = sorted(STRIPS.glob("*.png"))
    sample = random.Random(42).sample(files, 300)
    canvases, labels = [], []
    for path in sample:
        gray = _read_gray(path)
        canvas = preprocess_strip(gray, version=STRIP_V1)
        reference = legacy.preprocess(gray)
        assert (canvas is None) == (reference is None)
        if canvas is not None:
            assert np.array_equal(canvas, reference)  # type: ignore[arg-type]
            canvases.append(canvas)
            labels.append(path.stem.rsplit("_", 1)[0])
    results = ctc_greedy(cpu.digits(to_input(canvases)), cpu.digit_classes)
    accuracy = np.mean([r.text == label for r, label in zip(results, labels, strict=True)])
    assert accuracy > 0.78, accuracy


@pytest.mark.skipif(not WORDS.exists(), reason="Rock Capture CNN word images not found")
def test_word_model_reads_training_words(cpu: OrtEngine) -> None:
    rng = random.Random(42)
    canvases, labels = [], []
    for cls in cpu.word_classes:
        folder = WORDS / cls
        files = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
        for path in rng.sample(files, min(8, len(files))):
            canvas = preprocess_strip(_read_gray(path), version=cpu.word_preprocess)
            if canvas is not None:
                canvases.append(canvas)
                labels.append(cls)
    probs = softmax(cpu.words(to_input(canvases)))
    predicted = [cpu.word_classes[i] for i in probs.argmax(axis=1)]
    accuracy = np.mean([p == label for p, label in zip(predicted, labels, strict=True)])
    assert accuracy > 0.95, accuracy

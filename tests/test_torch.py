"""PyTorch backend: the shipped weights match the shipped ONNX files, and legacy conversion.

The pinned assets ship each model twice: `.onnx` (ONNX Runtime, the web app) and `.safetensors`
(PyTorch, dev mode). Both must compute the same. Rock Capture CNN's `.pth` checkpoints (when
that checkout is next to this repo) still convert: the word checkpoint is the shipped word model;
the digit one predates the "m" and keeps its 12-character alphabet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scanlake_inference.metadata import ModelFiles  # noqa: E402
from scanlake_inference.ort_backend import OrtEngine  # noqa: E402
from scanlake_inference.torch_backend import TorchEngine  # noqa: E402
from scanlake_models import checkpoint  # noqa: E402
from scanlake_models.digit_crnn import CHAR_CLASSES, LEGACY_CHAR_CLASSES  # noqa: E402
from tests.conftest import ASSETS  # noqa: E402
from tests.test_inference import OLD_REPO  # noqa: E402

pytestmark = pytest.mark.torch
LEGACY = OLD_REPO / "data" / "models"
SHIPPED = ASSETS / "models"
BATCH = np.random.default_rng(5).random((14, 1, 32, 256), dtype=np.float32)


@pytest.fixture(scope="module")
def legacy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not (LEGACY / "crnn_model.pth").exists() or not (LEGACY / "word_model2k.pth").exists():
        pytest.skip("Rock Capture CNN checkpoints not found")
    folder = tmp_path_factory.mktemp("models")
    checkpoint.convert_legacy(LEGACY / "crnn_model.pth", folder / "digit_crnn")
    checkpoint.convert_legacy(LEGACY / "word_model2k.pth", folder / "word_model2k")
    return folder


def test_shipped_torch_weights_match_the_onnx_files() -> None:
    if not (SHIPPED / "digit_crnn.safetensors").exists():
        pytest.skip("assets not fetched")
    pt = TorchEngine(ModelFiles(SHIPPED), "cpu")
    ort = OrtEngine(ModelFiles(SHIPPED), "cpu")
    assert pt.digit_classes == ort.digit_classes == CHAR_CLASSES
    assert pt.word_classes == ort.word_classes
    assert np.abs(pt.digits(BATCH) - ort.digits(BATCH)).max() < 1e-3
    assert np.abs(pt.words(BATCH[:6]) - ort.words(BATCH[:6])).max() < 1e-3
    assert (pt.digits(BATCH).argmax(2) == ort.digits(BATCH).argmax(2)).all()


def test_legacy_conversion(legacy: Path) -> None:
    _, words = checkpoint.load(legacy / "word_model2k")
    assert words["wordClasses"][10] == "empty" and words["numClasses"] == 31
    _, digits = checkpoint.load(legacy / "digit_crnn")
    assert digits["charClasses"] == LEGACY_CHAR_CLASSES and digits["numClasses"] == 13
    # The word model never changed: the old checkpoint is still the shipped ONNX file
    pt = TorchEngine(ModelFiles(legacy), "cpu")
    ort = OrtEngine(ModelFiles(SHIPPED), "cpu")
    assert np.abs(pt.words(BATCH[:6]) - ort.words(BATCH[:6])).max() < 1e-3


def test_registry_creates_torch_engine() -> None:
    from scanlake_inference.registry import available_backends, create_engine

    if not (SHIPPED / "digit_crnn.safetensors").exists():
        pytest.skip("assets not fetched")
    assert next(b for b in available_backends() if b.id == "torch-cpu").available
    engine = create_engine("torch-cpu", SHIPPED)
    assert engine.words(np.zeros((2, 1, 32, 256), np.float32)).shape == (2, 31)
    assert engine.digits(np.zeros((2, 1, 32, 256), np.float32)).shape == (64, 2, 14)

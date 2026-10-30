"""PyTorch backend: legacy checkpoint conversion and parity with the ONNX exports.

Converts Rock Capture CNN's `crnn_model.pth` / `word_model2k.pth` (when that checkout is next to
this repo) and requires the PyTorch outputs to match the ONNX models the web app ships, which
also proves those checkpoints are the ones the ONNX files were exported from.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scanlake_inference.metadata import ModelFiles  # noqa: E402
from scanlake_inference.ort_backend import OrtEngine  # noqa: E402
from scanlake_inference.torch_backend import TorchEngine  # noqa: E402
from scanlake_models import checkpoint  # noqa: E402
from tests.conftest import ASSETS  # noqa: E402
from tests.test_inference import OLD_REPO  # noqa: E402

pytestmark = pytest.mark.torch
LEGACY = OLD_REPO / "data" / "models"


@pytest.fixture(scope="module")
def models(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not (LEGACY / "crnn_model.pth").exists() or not (LEGACY / "word_model2k.pth").exists():
        pytest.skip("Rock Capture CNN checkpoints not found")
    folder = tmp_path_factory.mktemp("models")
    for name in ("digit_crnn.json", "word_model2k.json"):
        shutil.copy(ASSETS / "models" / name, folder / name)
    checkpoint.convert_legacy(LEGACY / "crnn_model.pth", folder / "digit_crnn")
    checkpoint.convert_legacy(LEGACY / "word_model2k.pth", folder / "word_model2k")
    return folder


def test_conversion_keeps_metadata(models: Path) -> None:
    _, meta = checkpoint.load(models / "word_model2k")
    assert meta["wordClasses"][10] == "empty" and meta["numClasses"] == 31
    _, digit = checkpoint.load(models / "digit_crnn")
    assert digit["charClasses"] == "0123456789.%" and digit["blankIdx"] == 0


def test_torch_matches_onnx(models: Path) -> None:
    pt = TorchEngine(ModelFiles(models), "cpu")
    ort = OrtEngine(ModelFiles(ASSETS / "models"), "cpu")
    assert pt.word_classes == ort.word_classes and pt.digit_classes == ort.digit_classes
    batch = np.random.default_rng(5).random((14, 1, 32, 256), dtype=np.float32)
    digit_diff = np.abs(pt.digits(batch) - ort.digits(batch)).max()
    word_diff = np.abs(pt.words(batch[:6]) - ort.words(batch[:6])).max()
    assert digit_diff < 1e-3, digit_diff
    assert word_diff < 1e-3, word_diff
    assert (pt.digits(batch).argmax(2) == ort.digits(batch).argmax(2)).all()


def test_registry_creates_torch_engine(models: Path) -> None:
    from scanlake_inference.registry import available_backends, create_engine

    assert next(b for b in available_backends() if b.id == "torch-cpu").available
    engine = create_engine("torch-cpu", models)
    assert engine.words(np.zeros((2, 1, 32, 256), np.float32)).shape == (2, 31)

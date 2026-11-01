"""Training package: splits, synthetic data, 1-epoch smoke training, ONNX export round trip."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from scanlake_core.preprocess import resize_pad  # noqa: E402
from scanlake_training import data, synth  # noqa: E402
from scanlake_training.cli import main as cli  # noqa: E402
from scanlake_training.evaluate import evaluate_model  # noqa: E402
from scanlake_training.export import export_onnx, verify  # noqa: E402
from scanlake_training.tasks import CrnnTask, WordTask  # noqa: E402
from scanlake_training.trainer import TrainConfig, train  # noqa: E402

pytestmark = pytest.mark.torch


@pytest.fixture(scope="module")
def strips(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("strips")
    assert synth.generate_strips(out, per_format=24, seed=1) == 120
    return out


@pytest.fixture(scope="module")
def words(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("words")
    counts = synth.generate_words(out, per_class=3, seed=1)
    assert len(counts) == 31 and "empty" in counts
    return out


def test_split_is_deterministic_and_stable(strips: Path) -> None:
    samples = data.strip_samples(strips)
    train1, val1 = data.split(samples, 0.25)
    train2, val2 = data.split(list(reversed(samples)), 0.25)
    assert {s.path for s in val1} == {s.path for s in val2}
    assert not {s.path for s in train1} & {s.path for s in val1}
    _, val3 = data.split(samples[:40], 0.25)  # fewer files: the survivors keep their side
    assert {s.path for s in val3} <= {s.path for s in val1}


def test_synth_strips_parse_and_preprocess(strips: Path) -> None:
    samples = data.strip_samples(strips)
    assert len(samples) == 120
    formats = {s.label.count(".") + 2 * s.label.endswith("%") + 4 * s.label.endswith("m")
               for s in samples}  # fmt: skip
    assert formats == {0, 1, 2, 3, 5}  # integer, decimal, percent, decimal %, decimal m
    canvas = resize_pad(data.load_gray(samples[0].path))
    assert canvas.shape == (32, 256) and canvas.max() > 100


def test_glyph_cut_from_a_screenshot(tmp_path: Path) -> None:
    """A rectangle cut from a frame (background included, alpha 0/255, on a larger transparent
    canvas) becomes a white glyph with the shape in alpha, cropped to the ink."""
    import cv2

    canvas = np.zeros((60, 50, 4), np.uint8)
    canvas[20:30, 10:23] = (25, 25, 25, 255)  # the cut: dark HUD background ...
    canvas[22:30, 11:22:3, :3] = 250  # ... with bright stems, 8 px tall
    canvas[22:24, 11:22, :3] = 250
    cv2.imencode(".png", canvas)[1].tofile(str(tmp_path / "m.png"))
    glyph = synth.load_glyph(tmp_path / "m.png")
    assert glyph.shape == (8, 11, 4) and (glyph[:, :, :3] == 255).all()
    assert glyph[0, 0, 3] > 230 and glyph[5, 1, 3] < 10  # stem vs. gap
    original = synth.load_glyph(synth.resource_dir() / "glyphs" / "3.png")
    assert original.shape == (11, 8, 4)  # soft-alpha glyphs are used as they are


def test_lowercase_m_sits_on_the_baseline() -> None:
    import random

    root = synth.resource_dir()
    glyphs = {ch: synth.load_glyph(root / "glyphs" / n) for ch, n in synth.GLYPH_FILES.items()}
    flat = np.full((64, 64, 3), 10, np.uint8)
    strip = synth.compose_strip("0m", glyphs, flat, random.Random(0)).astype(int)
    columns = np.nonzero((strip > 60).any(axis=0))[0]
    gap = np.nonzero(np.diff(columns) > 1)[0][0]
    digit, letter = strip[:, : columns[gap] + 1], strip[:, columns[gap + 1] :]
    # ink = brighter than half way from the background to the glyph's own peak
    rows = [np.nonzero((part > (10 + part.max()) / 2).any(axis=1))[0] for part in (digit, letter)]
    assert abs(rows[0].max() - rows[1].max()) <= 1  # same baseline
    peaks = [10 + 245 * int(glyphs[str(d)][:, :, 3].max()) / 255 for d in range(10)]
    assert min(peaks) - 2 <= letter.max() <= max(peaks) + 2  # as bright as the digits
    height = [r.max() - r.min() + 1 for r in rows]
    assert 0.6 < height[1] / height[0] < 0.85  # x-height, as on the HUD (8 of 11 px)


def test_a_wider_alphabet_starts_from_the_old_model(tmp_path: Path) -> None:
    import torch

    from scanlake_models import checkpoint
    from scanlake_models.digit_crnn import LEGACY_CHAR_CLASSES

    old_task = CrnnTask(LEGACY_CHAR_CLASSES)
    old = old_task.model()
    checkpoint.save(old, tmp_path / "old", old_task.meta({}))
    task = CrnnTask()
    assert task.alphabet == LEGACY_CHAR_CLASSES + "m"
    model = task.model()
    task.init_from(model, tmp_path / "old")
    new_state, old_state = model.state_dict(), old.state_dict()
    assert torch.equal(new_state["cnn.0.weight"], old_state["cnn.0.weight"])
    assert torch.equal(new_state["fc.weight"][:13], old_state["fc.weight"])  # blank + 12
    assert new_state["fc.weight"].shape[0] == 14
    with pytest.raises(ValueError):
        CrnnTask("0123456789").init_from(CrnnTask("0123456789").model(), tmp_path / "old")


def test_synth_is_reproducible(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    synth.generate_strips(a, 2, seed=7)
    synth.generate_strips(b, 2, seed=7)
    for pa, pb in zip(sorted(a.iterdir()), sorted(b.iterdir()), strict=True):
        assert pa.name == pb.name and pa.read_bytes() == pb.read_bytes()


def test_word_dataset(words: Path) -> None:
    classes = data.word_classes(words)
    ds = data.WordDataset(data.word_samples(words, classes), classes, augment=True)
    image, label = ds[0]
    assert tuple(image.shape) == (1, 32, 256) and 0 <= int(label) < len(classes)


def _events() -> tuple[list[dict[str, Any]], Any]:
    events: list[dict[str, Any]] = []
    return events, events.append


def test_crnn_smoke_train_export_and_evaluate(strips: Path, tmp_path: Path) -> None:
    train_s, val_s = data.split(data.strip_samples(strips), 0.25)
    events, report = _events()
    out = tmp_path / "run" / "digit_crnn"
    result = train(
        CrnnTask(),
        data.StripDataset(train_s, augment=True),
        data.StripDataset(val_s),
        out,
        TrainConfig(epochs=1, batch_size=16),
        report,
    )
    assert result.epochs_run == 1 and out.with_suffix(".safetensors").exists()
    assert [e["event"] for e in events] == ["start", "epoch", "done"]
    import json

    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["charClasses"] == "0123456789.%m" and meta["timeSteps"] == 64 and meta["epoch"] == 1

    onnx_stem = tmp_path / "onnx" / "digit_crnn"
    export_onnx(out, onnx_stem)
    assert (onnx_stem.parent / "digit_crnn.onnx.data").exists()
    check = verify(out, onnx_stem)
    assert check.ok, check

    via_onnx = evaluate_model(onnx_stem, strips)
    via_torch = evaluate_model(out, strips)
    assert via_onnx["total"] == via_torch["total"] == 120
    assert abs(via_onnx["accuracy"] - via_torch["accuracy"]) <= 1 / 120


def test_word_smoke_train(words: Path, tmp_path: Path) -> None:
    classes = data.word_classes(words)
    samples = data.word_samples(words, classes)
    events, report = _events()
    out = tmp_path / "word_model"
    train(
        WordTask(classes),
        data.WordDataset(samples, classes, augment=True),
        data.WordDataset(samples, classes),
        out,
        TrainConfig(epochs=1, batch_size=32),
        report,
    )
    assert events[-1]["event"] == "done" and out.with_suffix(".safetensors").exists()
    export_onnx(out, tmp_path / "onnx" / "word_model")
    assert verify(out, tmp_path / "onnx" / "word_model").ok


def test_cli_train_json(strips: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "cli" / "digit_crnn"
    code = cli(
        [
            "train",
            "crnn",
            "--data",
            str(strips),
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--out",
            str(out),
            "--json",
        ]
    )
    assert code == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("{")]
    import json

    assert json.loads(lines[-1])["event"] == "done"
    assert np.isfinite(json.loads(lines[1])["loss"])

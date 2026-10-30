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
    assert synth.generate_strips(out, per_format=24, seed=1) == 96
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
    assert len(samples) == 96
    formats = {s.label.count(".") + 2 * s.label.endswith("%") for s in samples}
    assert formats == {0, 1, 2, 3}  # integer, decimal, percent, decimal percent
    canvas = resize_pad(data.load_gray(samples[0].path))
    assert canvas.shape == (32, 256) and canvas.max() > 100


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
    assert meta["charClasses"] == "0123456789.%" and meta["timeSteps"] == 64 and meta["epoch"] == 1

    onnx_stem = tmp_path / "onnx" / "digit_crnn"
    export_onnx(out, onnx_stem)
    assert (onnx_stem.parent / "digit_crnn.onnx.data").exists()
    check = verify(out, onnx_stem)
    assert check.ok, check

    via_onnx = evaluate_model(onnx_stem, strips)
    via_torch = evaluate_model(out, strips)
    assert via_onnx["total"] == via_torch["total"] == 96
    assert abs(via_onnx["accuracy"] - via_torch["accuracy"]) <= 1 / 96


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

"""Dev mode tools: training crops, the labeler queue, model overrides, the training window."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from scanlake_core.domain.capture import Capture, FieldValue
from scanlake_core.geometry import Rect
from scanlake_core.profile.model import HudProfile
from scanlake_core.recognition import RoiReading
from scanlake_core.schema_map import slot_layout
from scanlake_desktop.assets import AssetSet
from scanlake_desktop.capture.source import write_image
from scanlake_desktop.dev.labeler import STRIPS, WORDS, LabelQueue
from scanlake_desktop.services.crops import collect, hud_text, word_class
from tests.conftest import ASSETS

WORD_CLASSES = ("aluminium", "empty", "inert_materials", "quartz")


def test_train_cli_runs_after_freeze_support(monkeypatch: pytest.MonkeyPatch) -> None:
    import multiprocessing
    import sys
    import types

    from scanlake_desktop.__main__ import main

    calls: list[str] = []
    cli = types.ModuleType("scanlake_training.cli")
    cli.main = lambda argv: calls.append(f"train {argv}") or 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scanlake_training.cli", cli)
    monkeypatch.setattr(multiprocessing, "freeze_support", lambda: calls.append("freeze"))
    assert main(["--train-cli", "export", "a", "b"]) == 0
    assert calls == ["freeze", "train ['export', 'a', 'b']"]


def test_hud_text_and_word_class() -> None:
    assert hud_text("amount", "42.09", "42.09%") == "42.09%"
    assert hud_text("amount", "42.10", "42.09%") == "42.10%"  # corrected: rebuilt with %
    assert hud_text("mass", "11826", "11828") == "11826"
    assert hud_text("instability", "4.58", "") == "4.58"
    assert word_class("Aluminum", WORD_CLASSES) == "aluminium"  # deposit spelling
    assert word_class("Inert Materials", WORD_CLASSES) == "inert_materials"
    assert word_class("", WORD_CLASSES) == "empty"
    assert word_class("Unobtainium", WORD_CLASSES) is None


def _reading(name: str, mode: str, raw: str) -> RoiReading:
    image = np.full((19, 41, 3), 200, np.uint8)
    return RoiReading(name, mode, Rect(0, 0, 41, 19), raw, 0.9, raw_text=raw, image=image)


def test_collect_writes_labelled_crops(tmp_path: Path, profiles: dict[str, HudProfile]) -> None:
    layout = slot_layout(profiles["mole"])
    ocr = FieldValue.from_ocr
    capture = Capture(
        capture_id="abcdef0123",
        session_id="s",
        cluster_id="k",
        timestamp="t",
        scan={
            "deposit": ocr("Quartz", 0.9),
            "mass": ocr("11826", 0.9).edited("11828"),
            "resistance": ocr("12", 0.9),
            "instability": ocr("4.58", 0.9),
            "volume": FieldValue(),
        },
        rows=[
            {
                "type": ocr("Inert Materials", 0.9),
                "amount": ocr("58.40", 0.9),
                "quality": ocr("512", 0.9),
            },
            {"type": FieldValue(), "amount": FieldValue(), "quality": FieldValue()},
        ],
    )
    readings = {
        "deposit": _reading("deposit", "word_cnn", "quartz"),
        "mass": _reading("mass", "digit_crnn", "11826"),
        "resistance": _reading("resistance", "digit_crnn", "12%"),
        "volume": _reading("volume", "digit_crnn", ""),
        "material1": _reading("material1", "word_cnn", "inert_materials"),
        "amount1": _reading("amount1", "digit_crnn", "58.40%"),
        "material2": _reading("material2", "word_cnn", "empty"),
    }
    n = collect(capture, layout, readings, WORD_CLASSES, tmp_path)
    strips = sorted(p.name for p in (tmp_path / "inbox" / "strips").iterdir())
    assert strips == [
        "11828_abcdef01mass.png",
        "12%_abcdef01resistance.png",
        "58.40%_abcdef01amount1.png",
    ]
    words = sorted(
        str(p.relative_to(tmp_path / "inbox" / "words")).replace("\\", "/")
        for p in (tmp_path / "inbox" / "words").rglob("*.png")
    )
    assert words == [
        "empty/abcdef01-material2.png",
        "inert_materials/abcdef01-material1.png",
        "quartz/abcdef01-deposit.png",
    ]
    assert n == 6


def test_label_queue(tmp_path: Path) -> None:
    image = np.zeros((19, 41, 3), np.uint8)
    write_image(tmp_path / "inbox" / "strips" / "11826_abc1mass.png", image)
    write_image(tmp_path / "inbox" / "strips" / "7_abc1qual1.png", image)
    write_image(tmp_path / "inbox" / "words" / "quartz" / "abc1-deposit.png", image)

    strips = LabelQueue(tmp_path, STRIPS)
    assert [i.label for i in strips.items] == ["11826", "7"]
    with pytest.raises(ValueError):
        strips.accept(strips.items[0], "11a26")
    target = strips.accept(strips.items[0], "11828")  # corrected
    assert target == tmp_path / "real" / "strips" / "11828_abc1mass.png" and target.exists()
    strips.delete(strips.items[0])
    assert strips.items == [] and not any((tmp_path / "inbox" / "strips").iterdir())

    words = LabelQueue(tmp_path, WORDS)
    assert words.items[0].label == "quartz"
    moved = words.accept(words.items[0], "iron")
    assert moved == tmp_path / "real" / "words" / "iron" / "abc1-deposit.png"


def test_model_override(tmp_path: Path) -> None:
    assets = AssetSet(ASSETS, tmp_path)
    assert assets.models_dir == ASSETS / "models"
    bundled_tag = assets.model_versions()
    user = tmp_path / "models"
    user.mkdir()
    for name in ("digit_crnn", "word_model2k"):
        for suffix in (".onnx", ".onnx.data", ".json"):
            shutil.copyfile(ASSETS / "models" / f"{name}{suffix}", user / f"{name}{suffix}")
    assert assets.models_dir == user
    assert assets.model_versions() == bundled_tag  # same files, same hashes
    assert assets.model_stamp().startswith(str(user))


@pytest.mark.torch
def test_training_window_end_to_end(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    from scanlake_desktop.dev.training_window import TrainingWindow
    from scanlake_training import synth

    synth.generate_strips(tmp_path / "synth" / "strips", per_format=12, seed=3)
    window = TrainingWindow(tmp_path, ASSETS / "models", tmp_path / "user_models", True)
    qtbot.addWidget(window)
    window.epochs.setValue(1)
    window.patience.setValue(0)
    window.device.setCurrentText("cpu")

    window.start()
    qtbot.waitUntil(lambda: window.process is None, timeout=180_000)
    assert len(window.chart.accuracy) == 1, window.log.toPlainText()[-2000:]
    assert window.export_button.isEnabled()

    window.export()  # export, then evaluate the new and the current model
    qtbot.waitUntil(lambda: "new model" in window.status.text(), timeout=240_000)
    assert window.use_button.isEnabled()

    installed: list[str] = []
    window.model_installed.connect(installed.append)
    window.use_model()
    assert installed == ["digit_crnn"]
    for name in ("digit_crnn", "word_model2k"):
        assert (tmp_path / "user_models" / f"{name}.onnx.data").exists()

"""The whole app, offscreen: real capture thread and ONNX models, replaying a synthetic HUD."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from scanlake_core.profile.model import HudProfile
from scanlake_core.vision.anchors import load_templates
from tests.conftest import ASSETS
from tests.test_inference import STRIPS, WORDS
from tests.test_recognition import PATTERNS, _pick_strips, _pick_word

pytestmark = pytest.mark.skipif(
    not (STRIPS.exists() and WORDS.exists()), reason="Rock Capture CNN training data not found"
)


def _hud_frame(hud: HudProfile, engine: object) -> np.ndarray:
    import random

    from scanlake_core.schema_map import field_slots

    rng = random.Random(3)
    capture = hud.main
    templates = load_templates(capture.anchors, ASSETS / "anchors")
    region = capture.search_region
    frame = np.full((region.h + 40, region.w + 40, 3), 8, dtype=np.uint8)
    for anchor in capture.anchors:
        t = templates[anchor.name]
        x, y = round(20 + anchor.ref_x), round(20 + anchor.ref_y)
        frame[y : y + t.shape[0], x : x + t.shape[1]] = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
    columns = {s.roi: s.column for s in field_slots(hud)}
    words = ["quartz", "iron", "gold", "inert_materials", "copper", "laranite"]
    for roi in capture.rois:
        if roi.recognition_mode == "word_cnn":
            image = _pick_word(engine, words.pop(), rng)  # type: ignore[arg-type]
        else:
            image = _pick_strips(engine, PATTERNS[columns[roi.name]], 1, rng)[0]  # type: ignore[arg-type]
        x, y = round(20 + roi.ref_x), round(20 + roi.ref_y)
        resized = cv2.resize(image, (roi.width, roi.height), interpolation=cv2.INTER_AREA)
        frame[y : y + roi.height, x : x + roi.width] = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    return frame


@pytest.fixture
def app_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from scanlake_desktop import paths

    monkeypatch.setenv("SCANLAKE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SCANLAKE_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("SCANLAKE_NO_HOTKEYS", "1")
    monkeypatch.setenv("SCANLAKE_FORMAT_URL", str(ASSETS / "format" / "format.json"))
    for fn in (paths.config_dir, paths.data_dir, paths.portable_root):
        fn.cache_clear()
    return tmp_path


def test_stage_edit_and_save(qtbot, app_env: Path, profiles: dict[str, HudProfile]) -> None:  # type: ignore[no-untyped-def]
    from scanlake_desktop.app import DesktopApp
    from scanlake_desktop.capture.source import write_image
    from scanlake_desktop.settings import Settings
    from scanlake_inference.metadata import ModelFiles
    from scanlake_inference.ort_backend import OrtEngine

    engine = OrtEngine(ModelFiles(ASSETS / "models"), "cpu")
    frame_path = app_env / "replay" / "frame.png"
    write_image(frame_path, _hud_frame(profiles["mole"], engine))

    settings = Settings(backend="ort-cpu", auto_upload=False, ship="mole", beep=False, fps=15)
    app = DesktopApp(settings, dev=True, replay=[frame_path])
    qtbot.addWidget(app.window)
    try:

        def settled() -> bool:
            s = app.last_state
            return (
                s is not None
                and s.locked
                and all(v.stable for name, v in s.votes.items() if v.text)
                and sum(v.stable for v in s.votes.values()) >= 15
            )

        qtbot.waitUntil(settled, timeout=15000)
        assert app.last_state is not None and app.last_state.backend == "ort-cpu"

        app.stage()
        assert app.staged
        scan, rows = app.window.form.values()
        state = app.last_state
        assert scan["mass"].text == state.votes["mass"].text
        assert scan["mass"].source == "ocr" and scan["mass"].confidence

        # The synthetic amounts do not add up to 100 %: saving must be refused
        app.save()
        assert app.sessions.session is not None and app.sessions.captures() == []

        # Fix the materials by hand and save
        form = app.window.form
        values = {"type": ["Quartz", "Iron", "", "", ""], "amount": ["60.00", "40.00", "", "", ""],
                  "quality": ["500", "20", "", "", ""]}  # fmt: skip
        for r, widgets in enumerate(form._row_widgets):
            for column, widget in widgets.items():
                text = values[column][r]
                if hasattr(widget, "setCurrentText"):
                    widget.setCurrentText(text)  # type: ignore[union-attr]
                else:
                    widget.setText(text)  # type: ignore[union-attr]
        scan, _ = form.values()
        if scan["deposit"].text == "":
            form._scan_widgets["deposit"].setCurrentText("Quartz")  # type: ignore[union-attr]
        mass = int(scan["mass"].text or 0)
        volume = float(scan["volume"].text or 0)
        if not mass or mass < 100 * volume:
            form._scan_widgets["mass"].setText(str(int(100 * volume) + 1000))  # type: ignore[union-attr]
        for column in ("resistance", "instability", "volume"):
            if not scan[column].text:
                form._scan_widgets[column].setText("1")  # type: ignore[union-attr]

        app.save()
        saved = app.sessions.captures()
        assert len(saved) == 1, app.window.form._issues.text()
        capture = saved[0]
        assert capture.rows[0]["type"].source == "manual"
        assert capture.rows[0]["type"].saved_confidence == 1.0
        assert capture.context["ship"] == "mole" and capture.context["system"] == "Stanton"
        assert app.window.scans.rowCount() == 1
        assert not app.staged
    finally:
        app.stop()
    assert os.listdir(app_env / "data")

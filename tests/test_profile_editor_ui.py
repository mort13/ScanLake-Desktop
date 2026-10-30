"""The profile editor window, offscreen: locate the HUD, move a ROI, test-read, save."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from scanlake_core.format import ParquetFormat
from scanlake_core.profile.model import HudProfile
from scanlake_core.vision.anchors import load_templates
from scanlake_desktop.assets import AssetSet
from tests.conftest import ASSETS


def _frame(hud: HudProfile, offset: tuple[int, int] = (30, 25)) -> np.ndarray:
    capture = hud.main
    templates = load_templates(capture.anchors, ASSETS / "anchors")
    region = capture.search_region
    frame = np.full((region.h + 60, region.w + 60, 3), 12, dtype=np.uint8)
    for anchor in capture.anchors:
        t = templates[anchor.name]
        x, y = round(offset[0] + anchor.ref_x), round(offset[1] + anchor.ref_y)
        frame[y : y + t.shape[0], x : x + t.shape[1]] = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
    return frame


@pytest.fixture
def editor(qtbot, tmp_path: Path, fmt: ParquetFormat, profiles: dict[str, HudProfile]):  # type: ignore[no-untyped-def]
    from scanlake_desktop.dev.profile_editor.editor_window import ProfileEditor
    from scanlake_inference.registry import create_engine

    assets = AssetSet(ASSETS, tmp_path)
    ships = [(e.key, e.label, e.file) for e in assets.manifest()]
    frame = _frame(profiles["mole"])
    window = ProfileEditor(assets, tmp_path, ships, lambda: fmt, lambda: frame,
                           lambda: create_engine("ort-cpu", ASSETS / "models"))  # fmt: skip
    qtbot.addWidget(window)
    window.ship.setCurrentIndex(window.ship.findData("mole"))
    window.use_live_frame()
    return window


def test_locates_and_draws(editor) -> None:  # type: ignore[no-untyped-def]
    assert "HUD located" in editor.status.text()
    assert editor.scale == pytest.approx(1.0, rel=0.01)
    assert editor.roi_list.count() == 20
    assert editor.coverage_note.text().startswith("This profile fills every required column")


def test_move_select_and_save(editor, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    editor.select_roi("mass")
    assert editor.f_w.value() == 41 and editor.f_column.currentText() == "mass"
    # The canvas reports the new top-left in frame pixels; offset (30, 25) at scale 1
    editor._roi_moved("mass", 30 + 133.0, 25 + 78.0)
    assert editor.profile.roi("mass")["ref_x"] == pytest.approx(133.0, abs=0.6)
    assert editor.profile.dirty
    applied: list[str] = []
    editor.applied.connect(applied.append)
    editor.save_and_apply()
    assert applied == ["mole"]
    saved = tmp_path / "profiles" / "mole_relative_anchors_crnn.json"
    assert saved.exists() and not editor.profile.dirty


def test_test_tab_reads_every_roi(editor) -> None:  # type: ignore[no-untyped-def]
    editor.run_test()
    assert editor.test_table.rowCount() == 20

"""The overlay over the game: what it shows where; never shown unless kept out of capture."""

from __future__ import annotations

import time
from typing import Any

import pytest

from scanlake_core.geometry import Rect
from scanlake_core.recognition import RoiReading
from scanlake_core.voting import Stability, Vote
from scanlake_desktop.capture.source import Monitor
from scanlake_desktop.worker.pipeline_worker import LiveState

REGION = Rect(100, 50, 300, 400)


def _state(locked: bool = True) -> LiveState:
    readings = {
        "mass": RoiReading("mass", "crnn", Rect(40, 60, 120, 20), "11826", 0.99),
        "inst": RoiReading("inst", "crnn", Rect(40, 90, 80, 20), "4.58", 0.6),
        "material5": RoiReading("material5", "word_cnn", Rect(10, 300, 100, 20), "", 1.0),
        "deposit": RoiReading("deposit", "word_cnn", Rect(10, 20, 100, 20), None, 0.0),
    }
    votes = {
        "mass": Vote("11826", 0.98, 1.0, 6, Stability.STABLE),
        "inst": Vote("4.58", 0.55, 0.6, 2, Stability.UNSTABLE),
        "material5": Vote("", 1.0, 1.0, 6, Stability.STABLE),  # unused material row
        "deposit": Vote("", 0.0, 0.0, 0, Stability.EMPTY),
    }
    return LiveState(
        t=time.time(), ship="golem", locked=locked, scale=1.0, fps=10.0, backend="ort-cpu",
        region=REGION, black=False, votes=votes, readings=readings,
    )  # fmt: skip


def test_labels_follow_the_rois_and_the_votes() -> None:
    from scanlake_desktop.ui.game_overlay import labels_for
    from scanlake_desktop.ui.style import OK, WARN

    labels = {label.text: label for label in labels_for(_state(), dpr=1.0)}
    assert set(labels) == {"11826", "4.58"}  # blank rows and unread fields get nothing
    assert labels["11826"].color == OK and labels["4.58"].color == WARN
    box = labels["11826"].box
    assert (box.x(), box.y(), box.width(), box.height()) == (40, 60, 120, 20)
    # 200 % Windows scaling: Qt draws in logical pixels
    hidpi = {label.text: label for label in labels_for(_state(), dpr=2.0)}["11826"].box
    assert (hidpi.x(), hidpi.y(), hidpi.width()) == (20, 30, 60)
    assert labels_for(_state(locked=False), dpr=1.0) == []


def test_one_line_per_hud_row() -> None:
    from PySide6.QtCore import QRectF

    from scanlake_desktop.ui.game_overlay import Label, panel_rows

    def label(text: str, x: float, y: float) -> Label:
        return Label(QRectF(x, y, 80, 20), text, "#fff")

    rows = panel_rows(
        [label("512", 260, 202), label("quartz", 20, 200), label("42.09", 150, 199),
         label("11826", 150, 40), label("iron", 20, 230)]
    )  # fmt: skip
    assert [[lb.text for lb in row] for row in rows] == [
        ["11826"],
        ["quartz", "42.09", "512"],
        ["iron"],
    ]


def _staged(status: str = "Staged · F10 saves · F9 restages") -> Any:
    from scanlake_desktop.ui.game_overlay import Staged

    state = _state()
    return Staged(
        values={"mass": ("11900", "manual"), "inst": ("4.58", "warn"), "material1": ("x", "")},
        readings=state.readings,
        region=REGION,
        status=status,
    )


def test_staged_values_by_roi(profiles: dict[str, Any]) -> None:
    from scanlake_core.domain.capture import FieldValue
    from scanlake_core.schema_map import slot_layout
    from scanlake_desktop.services.staging import staged_by_roi

    layout = slot_layout(profiles["golem"])
    mass = next(s for s in layout.scan if s.column == "mass")
    amount = next(s for s in layout.rows[1] if s.column == "amount")
    blank = next(s for s in layout.rows[2] if s.column == "amount")
    cells = {
        (-1, "mass"): (FieldValue("11900", source="manual"), "manual"),
        (1, "amount"): (FieldValue.from_ocr("42.09", 0.6), "error"),
        (2, "amount"): (FieldValue(), ""),
    }
    by_roi = staged_by_roi(layout, cells)
    assert by_roi[mass.roi] == ("11900", "manual")
    assert by_roi[amount.roi] == ("42.09", "error")
    assert blank.roi not in by_roi


def test_staged_labels_stay_where_the_values_were_read() -> None:
    from scanlake_desktop.ui.game_overlay import labels_for
    from scanlake_desktop.ui.style import MANUAL, WARN

    staged = _staged()
    labels = {label.text: label for label in labels_for(_state(), 1.0, staged)}
    assert set(labels) == {"11900", "4.58"}  # material1 was never read: no place for it
    assert labels["11900"].color == MANUAL and labels["4.58"].color == WARN
    lost = {label.text: label for label in labels_for(_state(locked=False), 1.0, staged)}
    assert lost["11900"].box.y() == 60  # HUD gone: still where it was staged


def test_staged_values_replace_the_live_ones_until_cleared(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtGui import QGuiApplication

    from scanlake_desktop.ui import game_overlay
    from scanlake_desktop.ui.game_overlay import GameOverlay

    monkeypatch.setattr(game_overlay, "STALE_S", 0.1)
    screen = QGuiApplication.primaryScreen().geometry()
    overlay = GameOverlay(exclude=lambda widget: True)
    qtbot.addWidget(overlay)
    overlay.set_monitor(Monitor(1, 0, 0, screen.width(), screen.height(), primary=True))
    overlay.update_state(_state())
    overlay.set_staged(_staged())
    assert sorted(label.text for label in overlay._labels) == ["11900", "4.58"]

    overlay.update_state(_state(locked=False))  # HUD gone, no more updates: staged stays
    qtbot.wait(800)
    assert overlay.isVisible() and len(overlay._labels) == 2

    overlay.set_staged(None)  # saved or cleared: back to live (nothing, HUD gone)
    assert not overlay.isVisible()


def test_never_shown_when_it_cannot_be_kept_out_of_the_capture(qtbot: Any) -> None:
    from scanlake_desktop.ui.game_overlay import GameOverlay

    overlay = GameOverlay(exclude=lambda widget: False)
    qtbot.addWidget(overlay)
    overlay.update_state(_state())
    overlay.flash("Saved", "ok")
    assert not overlay.available and not overlay.isVisible()


def test_covers_the_region_and_goes_when_the_hud_does(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtGui import QGuiApplication

    from scanlake_desktop.ui import game_overlay
    from scanlake_desktop.ui.game_overlay import RIGHT_MARGIN, GameOverlay

    monkeypatch.setattr(game_overlay, "FLASH_MS", 50)
    monkeypatch.setattr(game_overlay, "STALE_S", 0.2)
    screen = QGuiApplication.primaryScreen().geometry()
    overlay = GameOverlay(exclude=lambda widget: True)
    qtbot.addWidget(overlay)
    overlay.set_monitor(Monitor(1, 0, 0, screen.width(), screen.height(), primary=True))

    overlay.update_state(_state())
    assert overlay.isVisible()
    g = overlay.geometry()
    assert (g.x(), g.y(), g.height()) == (REGION.x, REGION.y, REGION.h)
    assert g.width() == min(REGION.w + RIGHT_MARGIN, screen.width() - REGION.x)
    image = overlay.grab().toImage()
    assert image.pixelColor(40, 70).alpha() > 0  # the outline of the mass ROI is drawn

    overlay.update_state(_state(locked=False))  # HUD lost: nothing left to show
    assert not overlay.isVisible()
    overlay.flash("New rock #2")  # messages show without a lock, then go
    assert overlay.isVisible()
    qtbot.waitUntil(lambda: not overlay.isVisible(), timeout=2000)

    overlay.update_state(_state())
    assert overlay.isVisible()
    qtbot.waitUntil(lambda: not overlay.isVisible(), timeout=3000)  # no updates: stale

    overlay.update_state(_state())
    overlay.set_enabled(False)
    assert not overlay.isVisible()
    overlay.update_state(_state())
    assert not overlay.isVisible()


def test_setting_round_trip(qtbot: Any) -> None:
    from scanlake_desktop.settings import Settings
    from scanlake_desktop.ui.settings_dialog import SettingsDialog

    assert Settings().game_overlay
    dialog = SettingsDialog(Settings(game_overlay=False), [Monitor(1, 0, 0, 800, 600)], [], "")
    qtbot.addWidget(dialog)
    assert not dialog.result_settings().game_overlay
    dialog.game_overlay.setChecked(True)
    assert dialog.result_settings().game_overlay

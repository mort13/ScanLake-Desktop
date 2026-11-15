"""The Overview tab, the in-game compass, the calibration dialog and their wiring in the app."""

from __future__ import annotations

import csv
import itertools
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scanlake_core.domain import navigation as nav
from scanlake_core.domain.capture import Capture, FieldValue, utc_timestamp
from scanlake_core.domain.navigation import (
    CalibrationResult,
    CameraCheck,
    ClusterSummary,
    Compass,
    Convention,
    NavFix,
    NavFrame,
    summarize_clusters,
)
from scanlake_core.domain.position import Position, ZonePosition
from scanlake_core.geometry import Rect
from scanlake_desktop.capture.source import Monitor
from scanlake_desktop.worker.pipeline_worker import LiveState, PositionRead
from tests.test_navigation import (
    ENGINE,
    FIELD,
    SHIP,
    SYSTEM,
    TURN,
    _capture,
    _flight,
    _in_field,
    _rz,
)


def _summaries() -> list[ClusterSummary]:
    captures = [
        _capture("a", volume="40", position=_in_field(1000, 0, 0)),
        _capture("a", volume="2.5", position=_in_field(1000, 0, 0)),
        _capture("b", deposit="Gold", volume="0.5", position=None),
    ]
    return summarize_clusters(captures, ["a", "b"])


def _card(panel: Any, cluster_id: str) -> list[str]:
    """What a cluster's card shows: title, content, zone, distance."""
    card = panel.cards[cluster_id]
    return [card.title.text(), card.content.text(), card.zone.text(), card.distance.text()]


def _order(panel: Any) -> list[str]:
    """The cards' cluster ids as laid out, first to last."""
    flow = panel.area.flow
    return [flow.itemAt(i).widget().cluster_id for i in range(flow.count())]


# -- Overview panel -----------------------------------------------------------------------------


def test_overview_lists_the_clusters(qtbot: Any) -> None:
    from scanlake_desktop.ui.overview_panel import OverviewPanel

    panel = OverviewPanel()
    qtbot.addWidget(panel)
    assert panel.cards == {} and "No clusters yet" in panel.hint.text()

    panel.show_clusters(_summaries())
    assert _order(panel) == ["a", "b"]
    assert _card(panel, "a") == ["#1 · Quartz", "2 rocks · 42.50 SCU", FIELD, "—"]
    assert _card(panel, "b") == ["#2 · Gold", "1 rock · 0.500 SCU", "No position", "—"]
    assert f"{FIELD}: 1000.0 / 0.0 / 0.0 m" in panel.cards["a"].zone.toolTip()

    panel.update_position(_in_field(1000, 300, 400))
    assert _card(panel, "a")[-1] == "500 m"
    assert _card(panel, "b")[-1] == "—"  # no position: no distance


def test_overview_orders_and_marks_the_cards(qtbot: Any) -> None:
    from scanlake_core.domain.crew import DONE, NEED_HELP, OPEN, build_cards
    from scanlake_desktop.ui.overview_panel import OverviewPanel

    panel = OverviewPanel()
    qtbot.addWidget(panel)
    marked: list[tuple[str, str]] = []
    panel.status_changed.connect(lambda cluster, status: marked.append((cluster, status)))
    own = [replace(s, last_scan=f"2026-11-10T10:0{i}:00.000Z") for i, s in enumerate(_summaries())]
    panel.show_clusters(build_cards(own, {}))
    assert _order(panel) == ["b", "a"]  # newest scan first
    first = panel.cards["a"]
    panel.update_position(_in_field(1000, 300, 400))

    first.done.click()  # asks; the card changes when the new status is shown
    assert marked == [("a", DONE)] and not first.done.isChecked()
    panel.show_clusters(build_cards(own, {"a": DONE}))
    assert _order(panel) == ["b", "a"] and first.done.isChecked() and first.badge.text() == "DONE"
    assert panel.cards["a"] is first  # the same widget: updated in place, not rebuilt
    assert first.distance.text() == "500 m"  # and its distance stays

    first.done.click()  # again: reopen
    panel.cards["b"].help.click()
    assert marked[1:] == [("a", OPEN), ("b", NEED_HELP)]
    panel.show_clusters(build_cards(own, {"b": NEED_HELP}, target="a"))
    assert _order(panel) == ["b", "a"]
    assert panel.cards["b"].badge.text() == "NEED HELP"
    assert panel.cards["a"].enroute.text() == "→ You"

    panel.show_clusters(build_cards(own, {"b": DONE}, target="a"))
    assert _order(panel) == ["a", "b"]  # on the way before done


def test_overview_shows_other_members_clusters(qtbot: Any) -> None:
    from scanlake_core.domain.crew import SharedCluster
    from scanlake_desktop.ui.overview_panel import OverviewPanel

    panel = OverviewPanel()
    qtbot.addWidget(panel)
    mine, theirs = _summaries()[0], replace(_summaries()[0], cluster_id="r", owner="<b>Bo</b>")
    panel.show_clusters(
        [SharedCluster(mine), SharedCluster(theirs, mine=False, owner_id="u", enroute=("Al",))]
    )
    remote = panel.cards["r"]
    assert remote.title.text() == "<b>Bo</b> · #1 · Quartz"  # as typed, never as markup
    assert remote.title.textFormat().name == "PlainText"
    assert remote.enroute.text() == "→ Al"

    panel.click("r")  # another member's cluster is a target like any other
    assert panel.target_id == "r" and panel.target.text() == "<b>Bo</b> #1 · Quartz"

    panel.set_shared("shared · reconnecting…", offline=True)
    assert "reconnecting" in panel.title.text()
    assert not remote.done.isEnabled() and panel.cards["a"].done.isEnabled()
    panel.set_shared("shared with 1")
    assert remote.done.isEnabled()


def test_overview_picks_and_drops_the_target(qtbot: Any) -> None:
    from scanlake_desktop.ui.overview_panel import OverviewPanel

    panel = OverviewPanel()
    qtbot.addWidget(panel)
    picked: list[str] = []
    dropped: list[bool] = []
    panel.target_selected.connect(picked.append)
    panel.target_cleared.connect(lambda: dropped.append(True))
    panel.show_clusters(_summaries())

    panel.click("b")  # no position: cannot be a target
    assert picked == [] and panel.target_id is None
    assert "no position" in panel.hint.text()

    panel.click("a")
    assert picked == ["a"] and panel.target_id == "a"
    assert panel.stop.isEnabled() and panel.target.text() == "#1 · Quartz"
    assert panel.cards["a"].is_target and not panel.cards["b"].is_target
    assert panel.cards["a"].badge.text() == "TARGET"

    summary = panel.summaries[0]
    frame = NavFrame(FIELD, (1000.0, 2000.0, 0.0), summary.centres[FIELD])
    panel.show_fix(NavFix(summary, frame, closing_speed=85.0, age=0.2))
    assert panel.distance.text() == "2.00 km"
    assert panel.offset.text() == "x +0.00 / y -2.00 / z +0.00 km"
    assert panel.closing.text() == "▼ 85 m/s closing"
    assert panel.eta.text() == "0:24"
    assert panel.frame.text() == FIELD
    assert panel.position_age.text() == "Read 0.2 s ago"
    assert panel.course.text().startswith("—")  # not moving
    assert panel.camera.text().startswith("not checked in this zone")
    panel.show_fix(
        NavFix(summary, frame, closing_speed=-3.0, age=12.0,
               camera=CameraCheck(nav.SYSTEM, True), course_off=120.0)
    )  # fmt: skip
    assert panel.closing.text() == "▲ 3 m/s opening" and panel.eta.text() == "—"
    assert panel.position_age.text().startswith("Last read 12 s ago")
    assert panel.course.text() == "120° off the cluster  (moving away)"
    assert panel.camera.text() == "checked here: CamDir is in the solar system's frame"

    panel.click("a")  # again: stop
    assert dropped == [True] and panel.target_id is None and not panel.stop.isEnabled()
    assert not panel.cards["a"].is_target

    panel.click("a")
    panel.show_clusters(_summaries()[1:])  # the target's cluster is gone
    assert dropped == [True, True] and panel.target_id is None


# -- compass overlay ----------------------------------------------------------------------------


def _overlay(qtbot: Any, available: bool = True) -> Any:
    from PySide6.QtGui import QGuiApplication

    from scanlake_desktop.ui.nav_overlay import NavOverlay

    screen = QGuiApplication.primaryScreen().geometry()
    overlay = NavOverlay(exclude=lambda widget: available)
    qtbot.addWidget(overlay)
    overlay.set_monitor(Monitor(1, 0, 0, screen.width(), screen.height(), primary=True))
    return overlay


def _fix(
    compass: Compass | None,
    age: float | None = 0.1,
    checked: bool = True,
    course: float | None = None,
) -> NavFix:
    summary = _summaries()[0]
    frame = NavFrame(FIELD, (0.0, 0.0, 0.0), summary.centres[FIELD])
    return NavFix(
        summary, frame, closing_speed=40.0, age=age, compass=compass,
        camera=CameraCheck(nav.SYSTEM, checked), course_off=course,
    )  # fmt: skip


def test_compass_draws_a_dot_ahead_and_an_x_behind(qtbot: Any) -> None:
    from scanlake_desktop.ui.nav_overlay import RING_R
    from scanlake_desktop.ui.style import HUD_MUTED, HUD_OK

    overlay = _overlay(qtbot)
    overlay.show_fix(None)
    assert not overlay.isVisible()

    overlay.show_fix(_fix(Compass((0.5, 0.25), behind=False, off_angle=34.0)))
    assert overlay.isVisible()
    centre = overlay.ring_centre()
    point = overlay.marker_point()
    assert (point.x(), point.y()) == (centre.x() + RING_R / 2, centre.y() - RING_R / 4)
    image = overlay.grab().toImage()
    x, y = round(point.x()), round(point.y())
    assert image.pixelColor(x + 3, y).red() > 200  # inside the filled dot
    assert image.pixelColor(x, y).red() > 200

    overlay.show_fix(_fix(Compass((0.5, 0.25), behind=True, off_angle=146.0)))
    image = overlay.grab().toImage()
    assert image.pixelColor(x, y).red() > 200  # where the X's strokes cross
    assert image.pixelColor(x + 3, y).red() < 100  # between its strokes: the dark ring fill

    overlay.show_fix(_fix(Compass((0.01, 0.0), behind=False, off_angle=0.6)))  # aligned: green
    c = overlay.ring_centre()
    image = overlay.grab().toImage()
    assert image.pixelColor(round(c.x() + RING_R), round(c.y())).name() == HUD_OK

    overlay.show_fix(_fix(Compass((0.01, 0.0), behind=False, off_angle=0.6), age=None))
    image = overlay.grab().toImage()  # no position: grey
    assert image.pixelColor(round(c.x() + RING_R), round(c.y())).name() == HUD_MUTED

    overlay.show_fix(_fix(None))  # not calibrated: ring and distance only
    assert overlay.marker_point() is None and overlay.isVisible()

    overlay.set_enabled(False)
    assert not overlay.isVisible()


def test_compass_marks_an_unchecked_zone_and_shows_only_the_dot(qtbot: Any) -> None:
    from scanlake_desktop.ui.nav_overlay import DOT_R, NORMAL, RING_R

    overlay = _overlay(qtbot)
    overlay.show_fix(_fix(Compass((0.5, 0.25), behind=False, off_angle=34.0), checked=False))
    point = overlay.marker_point()
    x, y = round(point.x()), round(point.y())
    image = overlay.grab().toImage()
    assert image.pixelColor(x, y).red() < 100  # hollow: the dark ring fill shows inside
    assert image.pixelColor(x + DOT_R - 1, y).red() > 150  # its outline

    overlay.show_fix(_fix(None, course=3.0))  # no compass: "?", the course is not drawn
    c = overlay.ring_centre()
    image = overlay.grab().toImage()
    assert image.pixelColor(round(c.x() + RING_R), round(c.y())).name() == NORMAL


def test_debug_readout(qtbot: Any) -> None:
    from scanlake_desktop.ui.nav_overlay import DEBUG_WIDTH, HEIGHT, WIDTH
    from scanlake_desktop.ui.overview_panel import DEBUG_ROWS, OverviewPanel, debug_lines

    summary = _summaries()[0]
    distance = NavFrame(FIELD, (1000.0, 300.0, 400.0), summary.centres[FIELD])
    way = NavFrame(SYSTEM, (1e9 + 1000, 2e9 + 300, -1e7 + 400), summary.centres[SYSTEM])
    fix = NavFix(
        summary, distance, age=0.3, compass=Compass((0.6, 0.0), False, 36.87),
        camera=CameraCheck(nav.SYSTEM, True), cam_dir=(39.0, -8.0, -64.0),
        compass_frame=way, calibrated=True,
    )  # fmt: skip
    lines = debug_lines(fix)
    assert lines[:2] == ["Read    0.3 s ago", "CamDir  39 -8 -64"]
    assert f"Frame   {FIELD}" in lines and "Ship    1000.0 / 300.0 / 400.0 m" in lines
    assert "Offset  0.0 / -300.0 / -400.0 m  = 500 m" in lines
    assert f"Compass {SYSTEM} (solar system frame, checked)" in lines
    assert any(line.startswith("        offset") for line in lines)  # another frame than Frame
    assert lines[-1] == "Camera  ahead +0.80 right +0.60 up +0.00  37° off"
    assert len(lines) <= DEBUG_ROWS
    for broken, reason in (
        (replace(fix, compass=None, calibrated=False), "not calibrated"),
        (replace(fix, compass=None, cam_dir=None), "no CamDir"),
        (
            replace(fix, compass=None, compass_frame=None),
            "the cluster has no position in that frame",
        ),
    ):  # why the ring shows "?"
        assert debug_lines(broken)[-1] == f"Camera  no dot: {reason}"
    assert debug_lines(replace(fix, cam_dir=None))[1] == "CamDir  not read in the last 2 s"

    overlay = _overlay(qtbot)
    overlay.show_fix(fix)
    assert overlay.width() == WIDTH
    overlay.set_debug(True)  # wider and taller; the ring keeps its place on screen
    assert overlay.width() == DEBUG_WIDTH and overlay.height() > HEIGHT + 100
    assert overlay.ring_centre().x() == DEBUG_WIDTH / 2
    qtbot.waitExposed(overlay)
    image = overlay.grab().toImage()
    top = overlay.ring_centre().y() + 32 + 40
    assert any(
        image.pixelColor(x, round(top)).red() > 150 for x in range(10, 300)
    )  # the readout's text

    panel = OverviewPanel()
    qtbot.addWidget(panel)
    toggled: list[bool] = []
    panel.debug_toggled.connect(toggled.append)
    panel.show_fix(fix)
    assert panel.debug_text.isHidden()
    panel.debug.setChecked(True)
    assert toggled == [True] and not panel.debug_text.isHidden()
    assert panel.debug_text.text() == "\n".join(lines)
    panel.set_debug(False)  # from the setting: no signal
    assert toggled == [True] and panel.debug_text.isHidden()


def test_compass_never_shown_when_it_cannot_be_kept_out_of_the_capture(qtbot: Any) -> None:
    overlay = _overlay(qtbot, available=False)
    overlay.show_fix(_fix(Compass((0.0, 0.0), False, 0.0)))
    assert not overlay.isVisible()


def test_compass_sits_where_the_settings_say(qtbot: Any) -> None:
    from PySide6.QtGui import QGuiApplication

    overlay = _overlay(qtbot)
    screen = QGuiApplication.primaryScreen().geometry()
    overlay.set_anchor(0.25, 0.5)
    overlay.show_fix(_fix(None))
    centre = overlay.mapToGlobal(overlay.ring_centre().toPoint())
    assert abs(centre.x() - screen.width() * 0.25) <= 1
    assert abs(centre.y() - screen.height() * 0.5) <= 1


# -- calibration dialog -------------------------------------------------------------------------


def _as_position(sample: Any) -> Position:
    """The flight (along the camera's axes in the system's frame) inside a turned field."""
    x, y, z = sample.pos
    return Position(
        zones=(
            ZonePosition(0, SHIP, -0.01, 9.72, 0.41),
            ZonePosition(1, FIELD, *_rz(sample.pos, -TURN)),
            ZonePosition(2, SYSTEM, 1e9 + x, 2e9 + y, z),
        ),
        cam_dir=sample.cam_dir,
    )


def test_calibration_records_and_fits(qtbot: Any, tmp_path: Path) -> None:
    from scanlake_desktop.ui.nav_calibration import CalibrationDialog

    ticks = itertools.count()
    dialog = CalibrationDialog(lambda: tmp_path, clock=lambda: float(next(ticks)))
    qtbot.addWidget(dialog)
    done: list[CalibrationResult] = []
    dialog.calibrated.connect(done.append)
    dialog.add(_in_field(0, 0, 0, cam_dir=(1, 2, 3)), 1.0)  # not open yet: not recording
    assert dialog.samples == []

    dialog.show()
    samples = _flight(ENGINE, np.random.default_rng(11))
    for s in samples:
        dialog.add(_as_position(s), s.t)
        dialog.add(_as_position(s), s.t)  # the same reading again: skipped
    dialog.add(Position(zones=(ZonePosition(0, FIELD, 1, 2, 3),)), 9999.0)  # no CamDir
    assert len(dialog.counted) == len(samples)
    assert len(dialog.samples) == 2 * len(samples)  # in the field's frame and the system's
    assert dialog.finish.isEnabled()

    dialog.fit()
    qtbot.waitUntil(lambda: dialog.fit_result is not None, timeout=10000)
    assert len(done) == 1 and done[0].convention is not None
    assert done[0].frame == SYSTEM  # not the field's turned frame
    assert not dialog.recording  # done: positions no longer recorded
    files = list(tmp_path.glob("calibration-*.csv"))
    assert len(files) == 1
    with files[0].open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0][:2] == ["t", "cam_dir_x"] and len(rows) == 2 * len(samples) + 1


def test_calibration_explains_a_failed_fit(qtbot: Any, tmp_path: Path) -> None:
    from scanlake_desktop.ui.nav_calibration import CalibrationDialog

    dialog = CalibrationDialog(lambda: tmp_path)
    qtbot.addWidget(dialog)
    dialog.show()
    for k in range(20):  # hovering
        dialog.add(_in_field(0, 0, 0, cam_dir=(10, 20, 30)), 1.0 + k * 0.25)
    assert not dialog.finish.isEnabled()
    dialog.fit()
    qtbot.waitUntil(lambda: not dialog.fitting, timeout=10000)
    assert dialog.fit_result is None and dialog.recording
    assert "No steady flight" in dialog.status.text()
    assert str(tmp_path) in dialog.status.text()  # where the samples went


# -- settings -----------------------------------------------------------------------------------


def test_compass_settings_round_trip(qtbot: Any) -> None:
    from scanlake_desktop.settings import Settings
    from scanlake_desktop.ui.settings_dialog import SettingsDialog

    start = Settings(nav_compass=False, nav_compass_x=0.3, nav_compass_y=0.7,
                     nav_camdir_convention=str(ENGINE))  # fmt: skip
    dialog = SettingsDialog(start, [Monitor(1, 0, 0, 800, 600)], [], "")
    qtbot.addWidget(dialog)
    result = dialog.result_settings()
    assert (result.nav_compass, result.nav_compass_x, result.nav_compass_y) == (False, 0.3, 0.7)
    assert result.nav_camdir_convention == str(ENGINE)  # learnt, not edited here
    dialog.nav_compass.setChecked(True)
    dialog.nav_compass_y.setValue(55)
    result = dialog.result_settings()
    assert result.nav_compass and result.nav_compass_y == 0.55


# -- the app ------------------------------------------------------------------------------------


def _start_app(app_env: Path) -> Any:
    from scanlake_desktop.app import DesktopApp
    from scanlake_desktop.capture.source import write_image
    from scanlake_desktop.settings import Settings

    frame = app_env / "replay" / "blank.png"
    write_image(frame, np.zeros((1440, 2560, 3), np.uint8))  # no HUD: the worker idles
    return DesktopApp(Settings(backend="ort-cpu", ship="mole", beep=False), replay=[frame])


def _live(position: Position, t: float) -> LiveState:
    return LiveState(
        t=t + 0.1, ship="mole", locked=False, scale=1.0, fps=2.0, backend="ort-cpu",
        region=Rect(0, 0, 10, 10), black=False, votes={}, readings={},
        position=position, position_t=t,
    )  # fmt: skip


def test_app_navigates_to_a_cluster(qtbot: Any, app_env: Path) -> None:
    from scanlake_desktop.worker.pipeline_worker import POSITION_EVERY_S, POSITION_NAV_S

    app = _start_app(app_env)
    qtbot.addWidget(app.window)
    try:
        assert app.window.tabs.tabText(1) == "Overview"
        app.start_session()
        session, cluster = app.sessions.session, app.sessions.cluster_id
        assert session is not None and cluster is not None
        for x in (900.0, 1100.0):
            capture = Capture(
                str(uuid.uuid4()), session.session_id, cluster, utc_timestamp(),
                scan={"deposit": FieldValue("Gold"), "volume": FieldValue("3.5")},
                position=_in_field(x, 0, 0, cam_dir=(0, 0, 0)),
            )  # fmt: skip
            app.db.save_capture(capture)
        app._refresh_session()
        overview = app.window.overview
        assert _card(overview, cluster)[:2] == ["#1 · Gold", "2 rocks · 7.00 SCU"]
        assert app._worker_config().position_every_s == POSITION_EVERY_S

        overview.click(cluster)
        assert app.nav is not None
        assert app._worker_config().position_every_s == POSITION_NAV_S
        now = time.time()
        app.on_live(_live(_in_field(1000, 300, 400, cam_dir=(0, 0, 0)), now))
        assert overview.distance.text() == "500 m"
        assert _card(overview, cluster)[-1] == "500 m"
        # Positions sent on their own (live capture) count at once; an older one a LiveState
        # carries later does not take over
        app.on_position(PositionRead(_in_field(1000, 600, 800), now + 0.1, ()))
        assert overview.distance.text() == "1.00 km"
        app.on_live(_live(_in_field(1000, 300, 400), now + 0.05))
        assert overview.distance.text() == "1.00 km"
        app.on_live(_live(_in_field(1000, 300, 400, cam_dir=(0, 0, 0)), now + 0.15))
        assert overview.distance.text() == "500 m"
        # The game's camera convention is built in: a compass without any calibration
        assert app.settings.nav_camdir_convention == ""
        assert overview.compass_status.text().startswith("Compass ready")
        app.on_live(_live(_in_field(1000, -500, 0, cam_dir=(0, 0, 0)), now + 0.25))
        fix = app.nav_overlay.fix
        assert fix is not None and fix.compass is not None
        assert fix.compass.off_angle == pytest.approx(0, abs=1e-6)  # +y, dead ahead

        # A calibration flown here counts only where it differs from the built-in one
        app._calibrated(CalibrationResult(Convention.parse("xyz/012/---/T"), "Calibrated."))
        assert app.settings.nav_camdir_convention == "xyz/012/---/T"
        assert app.own_convention is None and overview.compass_status.text().startswith(
            "Compass ready"
        )
        odd = Convention((2, 0, 1), (0, 1, 2), (1, -1, 1), True)
        app._calibrated(CalibrationResult(odd, "Calibrated."))
        assert app.nav is not None and app.convention == odd and app.nav.convention == odd
        assert "your own calibration (zxy/012/+-+/T)" in overview.compass_status.text()
        app.settings.nav_camdir_convention = ""  # back to the built-in one
        app._show_compass_status()

        assert not app.nav_overlay.debug and not app.settings.nav_debug
        overview.debug.setChecked(True)  # the readout, in the tab and in game; remembered
        assert app.nav_overlay.debug and app.settings.nav_debug
        assert overview.debug_text.text().startswith("Read")

        overview.stop.click()
        assert app.nav is None and app.nav_overlay.fix is None
        assert app._worker_config().position_every_s == POSITION_EVERY_S

        overview.cards[cluster].done.click()  # marked done: kept, and shown on the card
        assert app.db.cluster_statuses() == {cluster: ("done", False)}
        assert overview.cards[cluster].badge.text() == "DONE"

        overview.click(cluster)
        app.close_session()  # only the active session's clusters: the target goes with it
        assert app.nav is None and overview.cards == {}
    finally:
        app.stop()

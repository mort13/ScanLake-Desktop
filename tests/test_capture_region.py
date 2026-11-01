"""Where the app looks: monitors, automatic region, full-monitor search, manual overlay."""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scanlake_core.geometry import Rect
from scanlake_core.profile.model import HudProfile
from scanlake_core.vision.anchors import load_templates
from scanlake_desktop.capture.source import (
    FileSource,
    Monitor,
    auto_region,
    list_monitors,
    primary_monitor,
    write_image,
)
from scanlake_desktop.worker.pipeline_worker import PipelineWorker, WorkerConfig, _Loop
from tests.conftest import ASSETS
from tests.test_vision import _paste_hud

PRIMARY = Monitor(2, 0, 0, 2560, 1440, primary=True)
LEFT = Monitor(1, -1920, 0, 1920, 1080)


def _fake_mss(monkeypatch: pytest.MonkeyPatch, monitors: list[dict[str, Any]]) -> None:
    class FakeMSS:
        def __init__(self) -> None:
            self.monitors = [{"left": -1920, "top": 0, "width": 4480, "height": 1440}, *monitors]

        def __enter__(self) -> FakeMSS:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setitem(sys.modules, "mss", types.SimpleNamespace(MSS=FakeMSS))


def test_primary_monitor_is_not_always_number_one(monkeypatch: pytest.MonkeyPatch) -> None:
    left = {"left": -1920, "top": 0, "width": 1920, "height": 1080}
    main = {"left": 0, "top": 0, "width": 2560, "height": 1440}
    _fake_mss(monkeypatch, [{**left, "is_primary": False}, {**main, "is_primary": True}])
    monitors = list_monitors()
    assert primary_monitor(monitors).index == 2
    assert monitors[1].label() == "Monitor 2: 2560x1440 (primary)"
    _fake_mss(monkeypatch, [left, main])  # older mss: no is_primary -> the one at (0, 0)
    assert primary_monitor(list_monitors()).index == 2


def _config(hud: HudProfile, monitor: Monitor, manual: Rect | None = None) -> WorkerConfig:
    return WorkerConfig(
        hud=hud, ship="golem", templates=load_templates(hud.main.anchors, ASSETS / "anchors"),
        models_dir=Path("."), backend="ort-cpu", auto_backend=None, monitor=monitor,
        manual_region=manual,
    )  # fmt: skip


def test_empty_regions_are_never_grabbed(profiles: dict[str, HudProfile]) -> None:
    loop = _Loop(PipelineWorker())
    loop.config = _config(profiles["golem"], PRIMARY, manual=Rect(100, 100, 0, 0))
    loop.found_region = Rect(-925, 1195, 0, 0)  # what the search used to "find" on a desktop
    expected = auto_region(profiles["golem"].main.search_region, PRIMARY)
    assert loop._current_region() == expected
    loop.config = _config(profiles["golem"], PRIMARY, manual=Rect(100, 100, 400, 300))
    assert loop._current_region() == Rect(100, 100, 400, 300)


def test_a_failing_frame_does_not_stop_capturing(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, profiles: dict[str, HudProfile]
) -> None:
    from scanlake_desktop.worker import pipeline_worker

    monkeypatch.setattr(pipeline_worker, "SEARCH_FPS", 100.0)
    worker = PipelineWorker()
    loop = _Loop(worker)
    loop.config = replace(_config(profiles["golem"], PRIMARY), fps=100)
    loop.recognizer = object()  # type: ignore[assignment]
    ticks: list[int] = []

    def tick() -> bool:
        ticks.append(1)
        if len(ticks) == 4:
            worker._commands.put(("stop", None))
        raise RuntimeError("Region has zero or negative size")

    monkeypatch.setattr(loop, "_tick", tick)
    problems: list[str] = []
    worker.problem.connect(problems.append)
    loop.run()
    assert len(ticks) == 4  # kept going after the first failure
    assert problems == ["Capture problem: Region has zero or negative size"]  # reported once


@pytest.mark.parametrize("ship", ["golem", "mole", "prospector"])
def test_full_monitor_search(tmp_path: Path, profiles: dict[str, HudProfile], ship: str) -> None:
    hud = profiles[ship]
    loop = _Loop(PipelineWorker())
    loop.config = _config(hud, LEFT)
    desktop = np.random.default_rng(7).integers(0, 255, size=(1080, 1920, 3), dtype=np.uint8)
    write_image(tmp_path / "desktop.png", desktop)
    loop.source = FileSource([tmp_path / "desktop.png"])
    loop._wide_search(LEFT.scale)
    assert loop.found_region is None  # noise is not a HUD

    region = hud.main.search_region
    write_image(tmp_path / "game.png", _paste_hud(hud, LEFT.scale, (900, 300), (1920, 1080)))
    loop.source = FileSource([tmp_path / "game.png"])
    loop._wide_search(LEFT.scale)
    found = loop.found_region
    assert found is not None
    hud_x = LEFT.left + 900
    assert found.x <= hud_x and found.right >= hud_x + round(region.w * LEFT.scale)
    assert found.y <= 300 and found.bottom >= 300 + round(region.h * LEFT.scale)


def test_manual_region_can_be_dragged_from_the_settings_dialog(qtbot: Any) -> None:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtWidgets import QApplication

    from scanlake_desktop.settings import Settings
    from scanlake_desktop.ui.region_overlay import RegionOverlay
    from scanlake_desktop.ui.settings_dialog import SettingsDialog

    dialog = SettingsDialog(Settings(), [PRIMARY], [], "")
    qtbot.addWidget(dialog)
    dialog.setModal(True)  # like exec(): every other window is blocked
    dialog.show()
    dialog._pick_region()
    overlay = next(w for w in QApplication.topLevelWidgets() if isinstance(w, RegionOverlay))
    qtbot.waitExposed(overlay)
    qtbot.mousePress(overlay, Qt.MouseButton.LeftButton, pos=QPoint(50, 40))
    qtbot.mouseMove(overlay, QPoint(250, 190))
    qtbot.mouseRelease(overlay, Qt.MouseButton.LeftButton, pos=QPoint(250, 190))
    # Before: the overlay had no owner and was deleted as soon as it opened, so no region came
    # back. (Qt's test events skip the modal check, so this cannot test the overlay's modality.)
    assert dialog._manual_region is not None
    r = dialog._manual_region
    assert (r.x, r.y) == (50, 40)
    assert r.w == pytest.approx(200, abs=1) and r.h == pytest.approx(150, abs=1)
    assert dialog.result_settings().region_mode == "manual"

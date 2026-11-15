"""The compass calibration flight: records CamDir and position, then fits the convention.

The game's CamDir angles have no documented convention. Flying straight ahead moves the ship
along the camera's forward axis and strafing up along its up axis, so a short flight tells
`fit_convention` how the angles build the camera's orientation. Each read is recorded in the
local zone's frame and the solar system's, and the fit finds which one the angles are in, so
it works in an asteroid field too (its frame is turned against the solar system's). The raw
samples are always written to a CSV (one row per read and frame), so a failed fit can be
analysed later.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from scanlake_core.domain.navigation import (
    MIN_DIRECTIONS,
    MIN_FORWARD,
    MIN_UP,
    CalibrationResult,
    CalibrationSample,
    distinct_directions,
    fit_convention,
    motion_samples,
    nav_frames,
)
from scanlake_core.domain.position import Position, is_system_zone
from scanlake_desktop import tasks
from scanlake_desktop.ui.style import ERROR

INSTRUCTIONS = (
    "<b>Not needed normally</b>: how the game's camera angles work is built in. Fly this only "
    "if the compass dot keeps pointing the wrong way.<br><br>"
    "<b>Before you start</b>: sit in the pilot seat, first-person view, no freelook, with the "
    "game's debug info on (r_DisplayInfo).<ol>"
    "<li>Fly straight ahead faster than 20 m/s for about 5 seconds.</li>"
    "<li>Turn to a clearly different direction (pitch and roll too) and fly straight again. "
    "Do 4 to 6 such legs.</li>"
    "<li>Strafe straight up (vertical thrusters only) for about 3 seconds, twice, in "
    "different attitudes.</li></ol>"
    "Then press <b>Finish</b>. Turning is fine: only steady stretches are used."
)
HEADER = ["t", "cam_dir_x", "cam_dir_y", "cam_dir_z", "zone", "pos_x", "pos_y", "pos_z"]
COUNT_EVERY_S = 0.5  # the counters scan every sample; no need to on each position


def write_samples(path: Path, samples: list[CalibrationSample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(HEADER)
        for s in samples:
            out.writerow([f"{s.t:.3f}", *s.cam_dir, s.zone, *s.pos])


class CalibrationDialog(QDialog):
    calibrated = Signal(object)  # CalibrationResult with a convention

    def __init__(
        self,
        csv_dir: Callable[[], Path],
        parent: QWidget | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibrate compass")
        self.setModal(False)
        self._csv_dir = csv_dir
        self._clock = clock
        self.samples: list[CalibrationSample] = []  # each read in its local and system frame
        self.counted: list[CalibrationSample] = []  # each read once, for the counters
        self._last_t = 0.0
        self._last_count = 0.0
        self.fit_result: CalibrationResult | None = None
        self.fitting = False
        self._csv_path: Path | None = None

        intro = QLabel(INSTRUCTIONS)
        intro.setWordWrap(True)
        self.counters = QLabel()
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.finish = QPushButton("Finish")
        self.finish.setEnabled(False)
        self.finish.clicked.connect(self.fit)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.addButton(self.finish, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(self.reject)

        box = QVBoxLayout(self)
        box.addWidget(intro)
        box.addWidget(self.counters)
        box.addWidget(self.status)
        box.addWidget(buttons)
        self.resize(460, 0)
        self._count(force=True)

    @property
    def recording(self) -> bool:
        return self.isVisible() and not self.fitting and self.fit_result is None

    def add(self, position: Position | None, position_t: float) -> None:
        """One live position; repeats and positions without CamDir are skipped."""
        if not self.recording or position is None or position.cam_dir is None:
            return
        if position_t <= self._last_t:
            return
        frames = nav_frames(position)
        if not frames:
            return
        self._last_t = position_t
        # The local zone's frame and the solar system's: the fit finds which one CamDir is in
        local = frames[0]
        system = next((z for z in frames if is_system_zone(z.zone)), None)
        for z in (local, system):
            if z is None or (z is system and system is local):
                continue
            sample = CalibrationSample(position_t, position.cam_dir, z.zone, (z.x, z.y, z.z))
            self.samples.append(sample)
            if z is local:
                self.counted.append(sample)
        self._count()

    def _count(self, force: bool = False) -> None:
        now = self._clock()
        if not force and now - self._last_count < COUNT_EVERY_S:
            return
        self._last_count = now
        spans = motion_samples(self.counted)
        directions = distinct_directions(s.direction for s in spans)
        self.counters.setText(
            f"Positions read: {len(self.counted)}   ·   steady stretches: {len(spans)}"
            f"   ·   directions: {directions}"
        )
        ready = len(spans) >= MIN_FORWARD + MIN_UP and directions >= MIN_DIRECTIONS + 1
        self.finish.setEnabled(ready and not self.fitting)
        if not self.counted:
            self.status.setText("Waiting for positions with CamDir from the debug text…")
        elif not ready:
            self.status.setText("Keep flying the legs.")
        else:
            self.status.setText("Enough data: press Finish (more legs make the fit surer).")
        self.status.setStyleSheet("")

    def fit(self) -> None:
        if self.fitting:
            return
        self.fitting = True
        self.finish.setEnabled(False)
        self.status.setText("Fitting…")
        samples = list(self.samples)
        path = self._csv_dir() / time.strftime("calibration-%Y%m%d-%H%M%S.csv")
        try:
            write_samples(path, samples)
            self._csv_path = path
        except OSError:
            self._csv_path = None  # the fit still runs
        tasks.run_in_background(lambda: fit_convention(samples), self._fitted)

    def _fitted(self, result: object) -> None:
        self.fitting = False
        if isinstance(result, Exception):
            result = CalibrationResult(None, f"The fit failed: {result}")
        assert isinstance(result, CalibrationResult)
        where = f"\nSamples saved to {self._csv_path}" if self._csv_path else ""
        if result.convention is None:
            self.status.setText(f"{result.message}{where}")
            self.status.setStyleSheet(f"color: {ERROR}")
            self.finish.setEnabled(True)  # more legs, then try again
            return
        self.fit_result = result
        detail = (
            f"{result.message} {result.forward} forward and {result.up} up stretches,"
            f" {result.directions} directions."
        )
        self.status.setText(detail + where)
        self.status.setStyleSheet("")
        self.calibrated.emit(result)

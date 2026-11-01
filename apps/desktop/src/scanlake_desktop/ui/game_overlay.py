"""A click-through window over the game: what is being read, beside each HUD row, and messages.

It lies over the capture region, so it would end up in the app's own screen grab and spoil the
recognition. On Windows it is therefore excluded from screen capture (SetWindowDisplayAffinity,
WDA_EXCLUDEFROMCAPTURE, Windows 10 2004 and later); where that fails it never shows. Side effect:
it is also missing from screenshots and stream captures (OBS, Discord). The game must run in
borderless window mode, as for capturing: exclusive fullscreen draws over every other window.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QGuiApplication,
    QPainter,
    QPaintEvent,
    QPen,
    QShowEvent,
)
from PySide6.QtWidgets import QWidget

from scanlake_core.geometry import Rect
from scanlake_core.recognition import RoiReading
from scanlake_core.voting import Stability
from scanlake_desktop.capture.source import Monitor
from scanlake_desktop.ui.region_overlay import screen_for
from scanlake_desktop.ui.style import ERROR, MANUAL, OK, WARN
from scanlake_desktop.worker.pipeline_worker import LiveState

log = logging.getLogger(__name__)

WDA_EXCLUDEFROMCAPTURE = 0x11
STALE_S = 1.5  # no live update for this long (paused, app hidden): the labels go
FLASH_MS = 2500
RIGHT_MARGIN = 160  # logical px beyond the capture region, room for labels of the right column
STABILITY_COLORS = {Stability.STABLE: OK, Stability.UNSTABLE: WARN}
# Staged values, by the review form's highlight of the cell
STAGED_COLORS = {"": "#f0f0f0", "warn": WARN, "error": ERROR, "manual": MANUAL}
MESSAGE_COLORS = {"info": "#f0f0f0", "ok": OK, "warn": WARN, "error": ERROR}


def exclude_from_capture(widget: QWidget) -> bool:
    """Keeps the window out of screen captures (mss, screenshots, streams); False if impossible."""
    if sys.platform != "win32":
        return False
    import ctypes

    try:
        hwnd = int(widget.winId())
        return bool(ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))
    except (OSError, AttributeError):
        return False


@dataclass(frozen=True)
class Label:
    box: QRectF  # the ROI, in overlay coordinates (logical px from the capture region's corner)
    text: str
    color: str


@dataclass(frozen=True)
class Staged:
    """What F9 staged, shown instead of the live votes until it is saved or cleared."""

    values: Mapping[str, tuple[str, str]]  # roi -> (text, the review form's highlight)
    readings: Mapping[str, RoiReading]  # where each ROI was when staged
    region: Rect  # the capture region when staged
    status: str  # the line shown while staged


def labels_for(state: LiveState, dpr: float, staged: Staged | None = None) -> list[Label]:
    """The value of every ROI that has one, at the ROI's place.

    Live: the votes, while the HUD is locked. Staged: the staged values, at the ROIs' current
    place, or where they were when staged if the HUD is gone.
    """

    def box(r: Rect) -> QRectF:
        return QRectF(r.x / dpr, r.y / dpr, r.w / dpr, r.h / dpr)

    if staged is not None:
        readings = state.readings if state.locked else staged.readings
        return [
            Label(box(readings[roi].rect), text, STAGED_COLORS.get(highlight, STAGED_COLORS[""]))
            for roi, (text, highlight) in staged.values.items()
            if roi in readings
        ]
    if not state.locked:
        return []
    labels = []
    for name, reading in state.readings.items():
        vote = state.votes.get(name)
        if vote is None or not vote.text or vote.stability not in STABILITY_COLORS:
            continue  # nothing read yet, or a blank slot (an unused material row)
        labels.append(Label(box(reading.rect), vote.text, STABILITY_COLORS[vote.stability]))
    return labels


def panel_rows(labels: list[Label]) -> list[list[Label]]:
    """Labels grouped by HUD row (vertical overlap), top to bottom, each row left to right.

    The values are shown as one line per row in a column beside the panel: next to each ROI
    they would cover the neighbouring column (material | amount | quality).
    """
    rows: list[list[Label]] = []
    for label in sorted(labels, key=lambda lb: lb.box.center().y()):
        last = rows[-1][-1].box if rows else None
        if last is not None and abs(label.box.center().y() - last.center().y()) < last.height() / 2:
            rows[-1].append(label)
        else:
            rows.append([label])
    return [sorted(row, key=lambda lb: lb.box.x()) for row in rows]


class GameOverlay(QWidget):
    def __init__(self, exclude: Callable[[QWidget], bool] = exclude_from_capture) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool  # no taskbar button
            | Qt.WindowType.WindowTransparentForInput,  # clicks go to the game
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)  # the game keeps focus
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._exclude = exclude
        # Excluded before it is ever shown, so not a single frame of it reaches a capture
        self.available = exclude(self)
        if not self.available:
            log.info("game overlay off: this system cannot keep it out of the screen capture")
        self.enabled = True
        self._monitor = Monitor(1, 0, 0, 2560, 1440, primary=True)
        self._region = Rect(0, 0, 0, 0)
        self._state: LiveState | None = None
        self._staged: Staged | None = None
        self._labels: list[Label] = []
        self._message = ("", "info")
        self._last_update = 0.0
        self._font = QFont()
        self._font.setPixelSize(13)
        self._font.setBold(True)
        self._message_timer = QTimer(self, singleShot=True, interval=FLASH_MS)
        self._message_timer.timeout.connect(self._clear_message)
        self._stale_timer = QTimer(self, interval=500)
        self._stale_timer.timeout.connect(self._drop_stale)
        self._stale_timer.start()

    # -- inputs ---------------------------------------------------------------

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self._labels, self._message = [], ("", "info")
        self._refresh()

    def set_monitor(self, monitor: Monitor) -> None:
        self._monitor = monitor
        self._region = Rect(0, 0, 0, 0)  # re-place on the next update

    def update_state(self, state: LiveState) -> None:
        if not (self.available and self.enabled):
            return
        self._last_update = time.monotonic()
        self._state = state
        self._relabel()

    @property
    def staged(self) -> Staged | None:
        return self._staged

    def set_staged(self, staged: Staged | None) -> None:
        """Shows the staged values (None: back to the live readings)."""
        self._staged = staged
        if self.available and self.enabled:
            self._relabel()

    def _relabel(self) -> None:
        state = self._state
        if state is None:
            return
        staged = self._staged
        self._place(staged.region if staged is not None and not state.locked else state.region)
        self._labels = labels_for(state, self._dpr(), staged)
        self._refresh()

    def flash(self, text: str, kind: str = "info") -> None:
        """A short message at the top of the capture region (stage, save, errors)."""
        if not (self.available and self.enabled) or self._region.w == 0:
            return  # placed by the first live update; before that we do not know where
        self._message = (text, kind)
        self._message_timer.start()
        self._refresh()

    # -- placement --------------------------------------------------------------

    def _dpr(self) -> float:
        screen = screen_for(self._monitor)
        return screen.devicePixelRatio() if screen is not None else 1.0

    def _place(self, region: Rect) -> None:
        """Covers the capture region (+ room on the right), in Qt's logical coordinates."""
        if region == self._region:
            return
        self._region = region
        screen = screen_for(self._monitor) or QGuiApplication.primaryScreen()
        dpr = screen.devicePixelRatio()
        origin = screen.geometry().topLeft()
        x = origin.x() + (region.x - self._monitor.left) / dpr
        y = origin.y() + (region.y - self._monitor.top) / dpr
        area = QRect(
            round(x), round(y), round(region.w / dpr) + RIGHT_MARGIN, round(region.h / dpr)
        )
        self.setGeometry(area.intersected(screen.geometry()))

    # -- drawing ----------------------------------------------------------------

    def _refresh(self) -> None:
        content = self._labels or self._message[0] or self._staged is not None
        wanted = self.available and self.enabled and bool(content)
        if wanted and not self.isVisible():
            self.show()
        elif not wanted and self.isVisible():
            self.hide()
        if wanted:
            self.update()

    def _clear_message(self) -> None:
        self._message = ("", "info")
        self._refresh()

    def _drop_stale(self) -> None:
        if self._staged is not None:
            return  # staged values stay until saved or cleared
        if self._labels and time.monotonic() - self._last_update > STALE_S:
            self._labels = []
            self._refresh()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        self._exclude(self)  # again, in case the native window was recreated
        super().showEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self._font)
        metrics = QFontMetricsF(self._font)
        for label in self._labels:  # thin outlines show where each value is read
            outline = QColor(label.color)
            outline.setAlpha(140)
            painter.setPen(QPen(outline, 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(label.box)
        if self._labels:
            right = max(label.box.right() for label in self._labels) + 10
            left = min(label.box.left() for label in self._labels) - 10
            for row in panel_rows(self._labels):
                parts = [(label.text, QColor(label.color)) for label in row]
                pill = self._pill(metrics, "   ".join(text for text, _ in parts))
                x = right if right + pill.width() <= self.width() else left - pill.width()
                pill.moveTo(QPointF(x, row[0].box.center().y() - pill.height() / 2))
                self._draw_pill(painter, metrics, pill, parts)
        text, kind = self._message
        if not text and self._staged is not None:  # a message briefly covers the staged line
            text, kind = self._staged.status, "info"
        if text:
            pill = self._pill(metrics, text)
            pill.moveTo(QPointF(4, 4))
            color = QColor(MESSAGE_COLORS.get(kind, "#f0f0f0"))
            self._draw_pill(painter, metrics, pill, [(text, color)])

    @staticmethod
    def _pill(metrics: QFontMetricsF, text: str) -> QRectF:
        return QRectF(0, 0, metrics.horizontalAdvance(text) + 12, metrics.height() + 4)

    @staticmethod
    def _draw_pill(
        painter: QPainter, metrics: QFontMetricsF, pill: QRectF, parts: list[tuple[str, QColor]]
    ) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 185))
        painter.drawRoundedRect(pill, 4, 4)
        x = pill.x() + 6
        for i, (text, color) in enumerate(parts):
            text = text + ("   " if i < len(parts) - 1 else "")
            painter.setPen(color)
            width = metrics.horizontalAdvance(text)
            box = QRectF(x, pill.y(), width, pill.height())
            painter.drawText(box, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
            x += width

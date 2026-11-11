"""The in-game navigation compass: a ring with a dot that points at the target cluster.

The dot sits in the centre when the cluster is dead ahead and moves towards the side it lies on,
reaching the ring at 90° off; when the cluster is behind, the dot becomes an X (centred: right
behind you). The dot is hollow (the X thin) until straight flight in the zone confirmed the
frame CamDir is in. Distance and closing speed are written under the ring. Without a compass
(not calibrated, no CamDir read for 2 s, or no position of the cluster in CamDir's frame) the
ring shows "?". With the debug readout on, what the navigation works from is listed below.

Like `GameOverlay` it is a click-through window kept out of screen captures (and never shown
where that fails), so it can never spoil the HUD or debug-text reading.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

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

from scanlake_core.domain.navigation import NavFix
from scanlake_desktop.capture.source import Monitor
from scanlake_desktop.ui.game_overlay import exclude_from_capture
from scanlake_desktop.ui.overview_panel import (
    DEBUG_ROWS,
    STALE_S,
    debug_lines,
    format_distance,
    mono_font,
    target_label,
)
from scanlake_desktop.ui.region_overlay import screen_for
from scanlake_desktop.ui.style import HUD_FONT, HUD_MUTED, HUD_OK, HUD_TEXT

log = logging.getLogger(__name__)

WIDTH, HEIGHT = 260, 150  # logical px; room for "12,345 km   ▼ 1,234 m/s"
DEBUG_WIDTH = 600  # with the debug readout: room for its longest lines ...
DEBUG_LINE_H = 15  # ... and their rows
RING_R = 32
DOT_R = 5
ALIGNED_DEG = 5.0
UPDATE_STALE_S = 1.5  # no fix for this long (app paused or hidden): drawn grey
NORMAL = HUD_TEXT


def speed_text(closing: float | None) -> str:
    if closing is None or abs(closing) < 0.5:
        return ""
    return f"{'▼' if closing > 0 else '▲'} {abs(closing):,.0f} m/s"


class NavOverlay(QWidget):
    def __init__(self, exclude: Callable[[QWidget], bool] = exclude_from_capture) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._exclude = exclude
        self.available = exclude(self)  # before it is ever shown
        if not self.available:
            log.info("navigation compass off: this system cannot keep it out of the capture")
        self.enabled = True
        self._monitor = Monitor(1, 0, 0, 2560, 1440, primary=True)
        self._anchor = (0.5, 0.62)
        self.fix: NavFix | None = None
        self._last_update = 0.0
        self._font = QFont(HUD_FONT)
        self._font.setPixelSize(13)
        self._font.setBold(True)
        self.debug = False
        self._mono = mono_font(12)
        self._stale_timer = QTimer(self, interval=500)
        self._stale_timer.timeout.connect(self._repaint_if_stale)
        self._stale_timer.start()

    # -- inputs -------------------------------------------------------------------------------

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        self._refresh()

    def set_monitor(self, monitor: Monitor) -> None:
        self._monitor = monitor
        self._place()

    def set_anchor(self, x: float, y: float) -> None:
        """Where the ring's centre sits, as fractions of the game monitor."""
        self._anchor = (x, y)
        self._place()

    def show_fix(self, fix: NavFix | None) -> None:
        self.fix = fix
        self._last_update = time.monotonic()
        self._refresh()

    def set_debug(self, on: bool) -> None:
        """Lists what the navigation works from under the compass (`debug_lines`)."""
        self.debug = on
        self._place()
        self._refresh()

    # -- placement and visibility ---------------------------------------------------------------

    def size_now(self) -> tuple[int, int]:
        if self.debug:
            return DEBUG_WIDTH, HEIGHT + DEBUG_ROWS * DEBUG_LINE_H + 12
        return WIDTH, HEIGHT

    def _place(self) -> None:
        screen = screen_for(self._monitor) or QGuiApplication.primaryScreen()
        geom = screen.geometry()
        cx = geom.x() + geom.width() * self._anchor[0]
        cy = geom.y() + geom.height() * self._anchor[1]
        w, h = self.size_now()
        self.setGeometry(QRect(round(cx - w / 2), round(cy - self.ring_centre().y()), w, h))

    def _refresh(self) -> None:
        wanted = self.available and self.enabled and self.fix is not None
        if wanted and not self.isVisible():
            self._place()
            self.show()
        elif not wanted and self.isVisible():
            self.hide()
        if wanted:
            self.update()

    def _repaint_if_stale(self) -> None:
        if self.isVisible() and time.monotonic() - self._last_update > UPDATE_STALE_S:
            self.update()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        self._exclude(self)  # again, in case the native window was recreated
        super().showEvent(event)

    # -- drawing --------------------------------------------------------------------------------

    def ring_centre(self) -> QPointF:
        return QPointF(self.size_now()[0] / 2, 30 + RING_R)

    def marker_point(self) -> QPointF | None:
        """Where the dot or X is drawn; None without a compass reading."""
        fix = self.fix
        if fix is None or fix.compass is None:
            return None
        dx, dy = fix.compass.dot
        c = self.ring_centre()
        return QPointF(c.x() + dx * RING_R, c.y() - dy * RING_R)

    def _stale(self) -> bool:
        fix = self.fix
        return (
            fix is None
            or fix.age is None
            or fix.age > STALE_S
            or time.monotonic() - self._last_update > UPDATE_STALE_S
        )

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        fix = self.fix
        if fix is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self._font)
        metrics = QFontMetricsF(self._font)
        stale = self._stale()
        point = self.marker_point()
        compass = fix.compass
        aligned = compass is not None and not compass.behind and compass.off_angle < ALIGNED_DEG
        color = QColor(HUD_MUTED if stale else HUD_OK if aligned else NORMAL)
        c = self.ring_centre()

        painter.setBrush(QColor(0, 0, 0, 120))
        painter.setPen(QPen(color, 2))
        painter.drawEllipse(c, RING_R, RING_R)
        painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 110), 1))
        for dx, dy in ((-4, 0), (4, 0), (0, -4), (0, 4)):  # a faint mark where "aligned" is
            painter.drawLine(c, QPointF(c.x() + dx, c.y() + dy))

        if point is not None and compass is not None:
            checked = fix.camera.checked
            if compass.behind:
                painter.setPen(QPen(color, 2.5 if checked else 1.5))
                for sx in (-1, 1):
                    painter.drawLine(
                        QPointF(point.x() - DOT_R, point.y() - sx * DOT_R),
                        QPointF(point.x() + DOT_R, point.y() + sx * DOT_R),
                    )
            elif checked:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(point, DOT_R, DOT_R)
            else:  # hollow: this zone's camera frame is not confirmed yet
                painter.setPen(QPen(color, 1.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(point, DOT_R - 0.75, DOT_R - 0.75)
        else:
            painter.setPen(color)
            box = QRectF(c.x() - RING_R, c.y() - RING_R, 2 * RING_R, 2 * RING_R)
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, "?")

        self._pill(painter, metrics, target_label(fix.target), QPointF(c.x(), 12), color)
        if stale:
            below = "no position"
        elif fix.frame is None:
            below = "no shared zone"
        else:
            below = "   ".join(
                t for t in (format_distance(fix.frame.distance), speed_text(fix.closing_speed)) if t
            )
        self._pill(painter, metrics, below, QPointF(c.x(), c.y() + RING_R + 16), color)
        if self.debug:
            self._debug(painter, debug_lines(fix), c.y() + RING_R + 32)

    def _debug(self, painter: QPainter, lines: list[str], top: float) -> None:
        painter.setFont(self._mono)
        box = QRectF(4, top, self.size_now()[0] - 8, len(lines) * DEBUG_LINE_H + 8)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 185))
        painter.drawRoundedRect(box, 4, 4)
        painter.setPen(QColor(NORMAL))
        for i, line in enumerate(lines):
            row = QRectF(box.x() + 8, box.y() + 4 + i * DEBUG_LINE_H, box.width() - 16,
                         DEBUG_LINE_H)  # fmt: skip
            painter.drawText(row, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, line)

    @staticmethod
    def _pill(
        painter: QPainter, metrics: QFontMetricsF, text: str, centre: QPointF, color: QColor
    ) -> None:
        width = metrics.horizontalAdvance(text) + 12
        pill = QRectF(centre.x() - width / 2, centre.y() - metrics.height() / 2 - 2,
                      width, metrics.height() + 4)  # fmt: skip
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 185))
        painter.drawRoundedRect(pill, 4, 4)
        painter.setPen(color)
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, text)

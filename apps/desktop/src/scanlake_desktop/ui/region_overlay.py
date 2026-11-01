"""Full-screen overlay to drag the capture region by hand (when the automatic one is wrong)."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QGuiApplication,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QScreen,
)
from PySide6.QtWidgets import QWidget

from scanlake_desktop.capture.source import Monitor

MIN_BOX = 32  # px; the capture loop ignores smaller regions


def screen_for(monitor: Monitor) -> QScreen | None:
    """The QScreen showing an mss monitor (matched on physical size and position order)."""
    screens = QGuiApplication.screens()
    best = None
    for screen in screens:
        dpr = screen.devicePixelRatio()
        size = screen.geometry().size()
        if (
            round(size.width() * dpr) == monitor.width
            and round(size.height() * dpr) == monitor.height
        ):
            if best is None or abs(screen.geometry().x() * dpr - monitor.left) < abs(
                best.geometry().x() * best.devicePixelRatio() - monitor.left
            ):
                best = screen
    return best or QGuiApplication.primaryScreen()


class RegionOverlay(QWidget):
    """Emits the dragged rectangle in the monitor's physical pixels (x, y, w, h).

    Application-modal: it is opened from the (modal) settings dialog, which would otherwise
    block its mouse and keyboard input.
    """

    selected = Signal(int, int, int, int)

    def __init__(self, monitor: Monitor) -> None:
        super().__init__(
            None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._screen = screen_for(monitor)
        if self._screen is not None:
            self.setScreen(self._screen)
            self.setGeometry(self._screen.geometry())
        self._start: QPoint | None = None
        self._end: QPoint | None = None

    def start(self) -> None:
        self.showFullScreen()
        self.raise_()
        self.activateWindow()  # so Esc reaches it
        self.setFocus()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 110))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            self.rect().adjusted(0, 40, 0, 0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "Drag a box around the scan results panel · Esc cancels",
        )
        if self._start and self._end:
            box = QRect(self._start, self._end).normalized()
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(box, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setPen(QColor(80, 200, 120))
            painter.drawRect(box)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._start = self._end = event.position().toPoint()
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._start is not None:
            self._end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._start is None:
            return
        box = QRect(self._start, event.position().toPoint()).normalized()
        if box.width() >= MIN_BOX and box.height() >= MIN_BOX:
            dpr = self._screen.devicePixelRatio() if self._screen else 1.0
            self.selected.emit(
                round(box.x() * dpr), round(box.y() * dpr),
                round(box.width() * dpr), round(box.height() * dpr),
            )  # fmt: skip
        self.close()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.close()

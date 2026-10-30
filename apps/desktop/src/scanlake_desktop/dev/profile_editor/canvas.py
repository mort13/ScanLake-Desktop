"""The frozen frame with the profile drawn on it: ROIs can be moved, new boxes drawn."""

from __future__ import annotations

from enum import Enum

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QMouseEvent, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from scanlake_desktop.dev.debug_window import to_pixmap

ROI_COLOR = QColor(90, 210, 120)
SELECTED_COLOR = QColor(255, 210, 60)
ANCHOR_COLOR = QColor(255, 150, 40)
DRAW_COLOR = QColor(80, 160, 255)


class Mode(Enum):
    SELECT = "select"
    DRAW_ROI = "roi"
    CUT_ANCHOR = "anchor"


class _RoiItem(QGraphicsRectItem):
    def __init__(self, name: str, rect: QRectF, canvas: FrameCanvas) -> None:
        super().__init__(QRectF(0, 0, rect.width(), rect.height()))
        self.name = name
        self.canvas = canvas
        self.setPos(rect.topLeft())
        self.setPen(QPen(ROI_COLOR, 0))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        label = QGraphicsSimpleTextItem(name, self)
        label.setBrush(QBrush(ROI_COLOR))
        label.setScale(0.6)
        label.setPos(0, -9)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: object) -> object:  # noqa: N802
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.setPen(QPen(SELECTED_COLOR if value else ROI_COLOR, 0))
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event: object) -> None:  # noqa: N802
        super().mouseReleaseEvent(event)  # type: ignore[arg-type]
        self.canvas.roi_moved.emit(self.name, self.pos().x(), self.pos().y())


class FrameCanvas(QGraphicsView):
    """Shows a BGR frame; all coordinates in signals are frame pixels."""

    roi_selected = Signal(str)
    roi_moved = Signal(str, float, float)  # name, new top-left x, y
    rect_drawn = Signal(object, object)  # Mode, QRectF

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(20, 22, 26)))
        self.mode = Mode.SELECT
        self._rois: dict[str, _RoiItem] = {}
        self._drag_start: QPointF | None = None
        self._rubber: QGraphicsRectItem | None = None
        self.scene().selectionChanged.connect(self._selection_changed)

    def show_frame(self, frame: np.ndarray) -> None:
        self.scene().clear()
        self._rois.clear()
        self.scene().addPixmap(to_pixmap(frame))
        self.scene().setSceneRect(QRectF(0, 0, frame.shape[1], frame.shape[0]))
        self.resetTransform()
        self.scale(2.0, 2.0)

    def set_overlay(
        self, rois: dict[str, QRectF], anchors: dict[str, QRectF], selected: str | None
    ) -> None:
        for item in list(self.scene().items()):
            if isinstance(item, QGraphicsRectItem) and item is not self._rubber:
                if item.parentItem() is None:
                    self.scene().removeItem(item)
        self._rois.clear()
        for name, rect in anchors.items():
            box = self.scene().addRect(rect, QPen(ANCHOR_COLOR, 0, Qt.PenStyle.DashLine))
            text = QGraphicsSimpleTextItem(name, box)
            text.setBrush(QBrush(ANCHOR_COLOR))
            text.setScale(0.6)
            text.setPos(rect.left(), rect.bottom() + 1)
        for name, rect in rois.items():
            item = _RoiItem(name, rect, self)
            self.scene().addItem(item)
            self._rois[name] = item
        if selected in self._rois:
            self._rois[selected].setSelected(True)

    def set_mode(self, mode: Mode) -> None:
        self.mode = mode
        self.setCursor(
            Qt.CursorShape.ArrowCursor if mode is Mode.SELECT else Qt.CursorShape.CrossCursor
        )
        for item in self._rois.values():
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, mode is Mode.SELECT)

    def _selection_changed(self) -> None:
        chosen = [i for i in self.scene().selectedItems() if isinstance(i, _RoiItem)]
        if chosen:
            self.roi_selected.emit(chosen[0].name)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self.mode is Mode.SELECT:
            super().mousePressEvent(event)
            return
        self._drag_start = self.mapToScene(event.position().toPoint())
        color = DRAW_COLOR if self.mode is Mode.DRAW_ROI else ANCHOR_COLOR
        self._rubber = self.scene().addRect(
            QRectF(self._drag_start, self._drag_start), QPen(color, 0)
        )

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_start is not None and self._rubber is not None:
            here = self.mapToScene(event.position().toPoint())
            self._rubber.setRect(QRectF(self._drag_start, here).normalized())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_start is None or self._rubber is None:
            super().mouseReleaseEvent(event)
            return
        rect = self._rubber.rect()
        self.scene().removeItem(self._rubber)
        self._rubber, self._drag_start = None, None
        if rect.width() >= 3 and rect.height() >= 3:
            self.rect_drawn.emit(self.mode, rect)

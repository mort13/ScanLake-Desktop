"""What the capture loop sees right now: HUD lock, speed, and the voted value of every field."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from scanlake_core.schema_map import SlotLayout
from scanlake_core.voting import Stability
from scanlake_desktop.ui.style import ERROR, MUTED, OK, WARN
from scanlake_desktop.worker.pipeline_worker import LiveState
from scanlake_inference.registry import LABELS

DOT = "●"
STABILITY_COLOR = {Stability.STABLE: OK, Stability.UNSTABLE: WARN, Stability.EMPTY: MUTED}


class LivePanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._status = QLabel("Starting…")
        self._status.setProperty("role", "title")
        self._detail = QLabel()
        self._detail.setProperty("role", "hint")
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Field", "Live value", ""])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._rois: list[str] = []

        box = QVBoxLayout(self)
        box.addWidget(self._status)
        box.addWidget(self._detail)
        box.addWidget(self._table, 1)

    def set_layout(self, layout: SlotLayout) -> None:
        rows: list[tuple[str, str]] = [(s.roi, s.column) for s in layout.scan]
        for i, row in enumerate(layout.rows):
            rows += [(s.roi, f"{s.column} {i + 1}") for s in row]
        self._rois = [roi for roi, _ in rows]
        self._table.setRowCount(len(rows))
        for r, (_, label) in enumerate(rows):
            self._table.setItem(r, 0, QTableWidgetItem(label))
            self._table.setItem(r, 1, QTableWidgetItem(""))
            dot = QTableWidgetItem(DOT)
            dot.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(r, 2, dot)

    def show_message(self, text: str, color: str = MUTED) -> None:
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color}")

    def update_state(self, state: LiveState) -> None:
        if state.black:
            self.show_message("Black screen: run the game in borderless window mode", ERROR)
        elif state.locked:
            stable = sum(v.stable for v in state.votes.values())
            self.show_message(f"HUD locked · {stable}/{len(state.votes)} stable", OK)
        else:
            self.show_message("Looking for the scan HUD…", WARN)
        backend = LABELS.get(state.backend, state.backend)
        ms = sum(state.timings.values()) * 1000
        self._detail.setText(
            f"{backend} · {state.fps:.0f} fps · {ms:.0f} ms/frame · scale {state.scale:.2f}"
        )
        for r, roi in enumerate(self._rois):
            vote = state.votes.get(roi)
            value = self._table.item(r, 1)
            dot = self._table.item(r, 2)
            if vote is None or value is None or dot is None:
                continue
            value.setText(vote.text if vote.text else "")
            tip = f"confidence {vote.confidence:.0%}, {vote.count} votes"
            if len(vote.candidates) > 1:
                tip += " · also: " + ", ".join(
                    f"{t or '∅'} {s:.0%}" for t, s in vote.candidates[1:]
                )
            value.setToolTip(tip)
            dot.setForeground(QBrush(QColor(STABILITY_COLOR[vote.stability])))
            dot.setToolTip(vote.stability.value)
